package com.ridevision.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * Wraps the YOLOv8n TFLite pothole model (best.tflite).
 * Supports flexible output tensor shapes ([1, 5, 8400], [1, 8400, 5], etc.),
 * auto-detects raw logits vs sigmoid probabilities, and handles coordinate scaling.
 */
class PotholeDetector(
    context: Context,
    modelFileName: String = "best.tflite",
    var confidenceThreshold: Float = 0.20f,
    private val iouThreshold: Float = 0.45f
) {
    data class Detection(
        val box: RectF,
        val confidence: Float,
        val severity: String = "moderate"
    )

    data class AnalysisResult(
        val detections: List<Detection>,
        val maxRawScore: Float,
        val outputShape: String
    )

    private val inputSize = 640
    private val interpreter: Interpreter

    init {
        val model: MappedByteBuffer = loadModelFile(context, modelFileName)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(model, options)
    }

    /** Loads the .tflite file from app assets into a memory-mapped buffer. */
    private fun loadModelFile(context: Context, fileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /** Resizes the bitmap to 640x640 and packs normalized (0.0 - 1.0) float RGB values. */
    private fun bitmapToInputBuffer(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val buffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4) // 4 bytes per float
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            val r = (pixel shr 16 and 0xFF) / 255.0f
            val g = (pixel shr 8 and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
        }

        buffer.rewind()
        return buffer
    }

    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + exp(-x))
    }

    /**
     * Executes inference on a frame and returns comprehensive analysis data.
     */
    fun analyze(bitmap: Bitmap): AnalysisResult {
        val origWidth = bitmap.width
        val origHeight = bitmap.height

        val inputBuffer = bitmapToInputBuffer(bitmap)

        val outputTensor = interpreter.getOutputTensor(0)
        val shape = outputTensor.shape() // e.g. [1, 5, 8400] or [1, 8400, 5]
        val shapeString = shape.joinToString("x")

        val candidates = mutableListOf<Detection>()
        var highestScoreFound = 0f

        if (shape.size == 3) {
            val dim1 = shape[1]
            val dim2 = shape[2]

            if (dim1 < dim2) {
                // Shape is [1, C, N], e.g. [1, 5, 8400]
                val numChannels = dim1
                val numAnchors = dim2
                val output = Array(1) { Array(numChannels) { FloatArray(numAnchors) } }
                interpreter.run(inputBuffer, output)

                for (i in 0 until numAnchors) {
                    var maxScore = 0f
                    for (c in 4 until numChannels) {
                        var score = output[0][c][i]
                        // Apply sigmoid if score is raw logit
                        if (score < 0f || score > 1f) {
                            score = sigmoid(score)
                        }
                        if (score > maxScore) maxScore = score
                    }

                    if (maxScore > highestScoreFound) highestScoreFound = maxScore
                    if (maxScore < confidenceThreshold) continue

                    val cx = output[0][0][i]
                    val cy = output[0][1][i]
                    val w = output[0][2][i]
                    val h = output[0][3][i]

                    addCandidate(cx, cy, w, h, maxScore, origWidth, origHeight, candidates)
                }
            } else {
                // Shape is [1, N, C], e.g. [1, 8400, 5]
                val numAnchors = dim1
                val numChannels = dim2
                val output = Array(1) { Array(numAnchors) { FloatArray(numChannels) } }
                interpreter.run(inputBuffer, output)

                for (i in 0 until numAnchors) {
                    var maxScore = 0f
                    for (c in 4 until numChannels) {
                        var score = output[0][i][c]
                        if (score < 0f || score > 1f) {
                            score = sigmoid(score)
                        }
                        if (score > maxScore) maxScore = score
                    }

                    if (maxScore > highestScoreFound) highestScoreFound = maxScore
                    if (maxScore < confidenceThreshold) continue

                    val cx = output[0][i][0]
                    val cy = output[0][i][1]
                    val w = output[0][i][2]
                    val h = output[0][i][3]

                    addCandidate(cx, cy, w, h, maxScore, origWidth, origHeight, candidates)
                }
            }
        }

        val filtered = nonMaxSuppression(candidates)
        return AnalysisResult(filtered, highestScoreFound, shapeString)
    }

    fun detect(bitmap: Bitmap): List<Detection> {
        return analyze(bitmap).detections
    }

    private fun addCandidate(
        cx: Float, cy: Float, w: Float, h: Float,
        conf: Float, origWidth: Int, origHeight: Int,
        candidates: MutableList<Detection>
    ) {
        if (w <= 0f || h <= 0f) return

        val scaleX = if (cx > 2.0f || w > 2.0f) 1f / inputSize else 1f
        val scaleY = if (cy > 2.0f || h > 2.0f) 1f / inputSize else 1f

        val left = ((cx - w / 2f) * scaleX * origWidth).coerceIn(0f, origWidth.toFloat())
        val top = ((cy - h / 2f) * scaleY * origHeight).coerceIn(0f, origHeight.toFloat())
        val right = ((cx + w / 2f) * scaleX * origWidth).coerceIn(0f, origWidth.toFloat())
        val bottom = ((cy + h / 2f) * scaleY * origHeight).coerceIn(0f, origHeight.toFloat())

        val bw = right - left
        val bh = bottom - top
        if (bw <= 1f || bh <= 1f) return

        val areaRatio = (bw * bh) / (origWidth * origHeight.toFloat())

        val severity = when {
            areaRatio > 0.05f || bw > (origWidth * 0.25f) -> "severe"
            areaRatio > 0.015f || bw > (origWidth * 0.12f) -> "moderate"
            else -> "minor"
        }

        candidates.add(Detection(RectF(left, top, right, bottom), conf, severity))
    }

    /** Greedy Non-Maximum Suppression to remove duplicate candidate boxes. */
    private fun nonMaxSuppression(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val kept = mutableListOf<Detection>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            kept.add(best)
            sorted.removeAll { iou(best.box, it.box) > iouThreshold }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = maxOf(a.left, b.left)
        val interTop = maxOf(a.top, b.top)
        val interRight = minOf(a.right, b.right)
        val interBottom = minOf(a.bottom, b.bottom)

        val interArea = maxOf(0f, interRight - interLeft) * maxOf(0f, interBottom - interTop)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val unionArea = areaA + areaB - interArea

        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    fun close() {
        interpreter.close()
    }
}
