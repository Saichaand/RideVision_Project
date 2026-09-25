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

/**
 * Wraps the YOLOv8n TFLite pothole model (best.tflite) exported from the training pipeline.
 * Input: 640x640 RGB image.
 * Output: [1, 5, 8400] tensor (cx, cy, w, h, confidence) for single-class pothole detector.
 *
 * Uses core TensorFlow Lite runtime without dependency on external support packages
 * for maximum compatibility across AGP versions.
 */
class PotholeDetector(
    context: Context,
    modelFileName: String = "best.tflite",
    private val confidenceThreshold: Float = 0.35f,
    private val iouThreshold: Float = 0.45f
) {
    data class Detection(
        val box: RectF,
        val confidence: Float,
        val severity: String = "moderate"
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

    /**
     * Executes inference on a frame.
     * Returned bounding boxes are scaled to the ORIGINAL bitmap's width & height.
     */
    fun detect(bitmap: Bitmap): List<Detection> {
        val origWidth = bitmap.width
        val origHeight = bitmap.height

        val inputBuffer = bitmapToInputBuffer(bitmap)

        // YOLOv8 single-class output shape: [1, 5, 8400]
        val numAnchors = 8400
        val output = Array(1) { Array(5) { FloatArray(numAnchors) } }
        interpreter.run(inputBuffer, output)

        val candidates = mutableListOf<Detection>()

        for (i in 0 until numAnchors) {
            val conf = output[0][4][i]
            if (conf < confidenceThreshold) continue

            val cx = output[0][0][i]
            val cy = output[0][1][i]
            val w = output[0][2][i]
            val h = output[0][3][i]

            // Convert center format (cx, cy, w, h) to original image pixel coordinates
            val left = (cx - w / 2f) / inputSize * origWidth
            val top = (cy - h / 2f) / inputSize * origHeight
            val right = (cx + w / 2f) / inputSize * origWidth
            val bottom = (cy + h / 2f) / inputSize * origHeight

            val bw = right - left
            val bh = bottom - top
            val areaRatio = (bw * bh) / (origWidth * origHeight.toFloat())

            val severity = when {
                areaRatio > 0.05f || bw > 180f -> "severe"
                areaRatio > 0.018f || bw > 90f -> "moderate"
                else -> "minor"
            }

            candidates.add(Detection(RectF(left, top, right, bottom), conf, severity))
        }

        return nonMaxSuppression(candidates)
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
