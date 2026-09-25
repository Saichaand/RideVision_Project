import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.MappedByteBuffer

/**
 * Wraps a YOLOv8n TFLite pothole model exported from the training notebook
 * (best.tflite). Input: 640x640 RGB image. Output: [1, 5, 8400] tensor
 * (x, y, w, h, confidence) per anchor for a single-class model.
 *
 * Usage:
 *   val detector = PotholeDetector(context, "best.tflite")
 *   val results = detector.detect(bitmap)
 *   results.forEach { println("box=${it.box} conf=${it.confidence}") }
 */
class PotholeDetector(
    context: Context,
    modelFileName: String,
    private val confidenceThreshold: Float = 0.45f,
    private val iouThreshold: Float = 0.45f
) {
    data class Detection(val box: RectF, val confidence: Float)

    private val inputSize = 640
    private val interpreter: Interpreter

    init {
        val model: MappedByteBuffer = FileUtil.loadMappedFile(context, modelFileName)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            // Uncomment if you add the GPU delegate dependency for faster inference:
            // addDelegate(GpuDelegate())
        }
        interpreter = Interpreter(model, options)
    }

    /** Runs detection on a single frame. Coordinates in the returned boxes are
     *  normalized to the ORIGINAL bitmap's width/height, not the model's 640x640. */
    fun detect(bitmap: Bitmap): List<Detection> {
        val origWidth = bitmap.width
        val origHeight = bitmap.height

        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(bitmap) }
        val processor = ImageProcessor.Builder()
            .add(ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // scale pixel values to 0-1
            .build()
        val input = processor.process(tensorImage)

        // YOLOv8 export shape: [1, 5, 8400] for a single class (x,y,w,h,conf)
        val output = TensorBuffer.createFixedSize(intArrayOf(1, 5, 8400), DataType.FLOAT32)
        interpreter.run(input.buffer, output.buffer.rewind())

        val raw = output.floatArray
        val numAnchors = 8400
        val candidates = mutableListOf<Detection>()

        for (i in 0 until numAnchors) {
            val conf = raw[4 * numAnchors + i]
            if (conf < confidenceThreshold) continue

            val cx = raw[0 * numAnchors + i]
            val cy = raw[1 * numAnchors + i]
            val w = raw[2 * numAnchors + i]
            val h = raw[3 * numAnchors + i]

            // Convert normalized center-format box back to original image pixel coords
            val left = (cx - w / 2f) / inputSize * origWidth
            val top = (cy - h / 2f) / inputSize * origHeight
            val right = (cx + w / 2f) / inputSize * origWidth
            val bottom = (cy + h / 2f) / inputSize * origHeight

            candidates.add(Detection(RectF(left, top, right, bottom), conf))
        }

        return nonMaxSuppression(candidates)
    }

    /** Standard greedy NMS to remove overlapping duplicate boxes for the same pothole. */
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

/*
Gradle dependencies needed (app/build.gradle):

implementation("org.tensorflow:tensorflow-lite:2.14.0")
implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

Place best.tflite in app/src/main/assets/ (matching the modelFileName you pass in).

Multi-frame confirmation (recommended, from the design discussion):
Track detections per rough location bucket (e.g. round GPS to ~5m grid) across
consecutive frames during a trip. Only surface / auto-file a detection once it
has appeared in 3+ consecutive frames above threshold, to filter shadow/manhole
false positives before they reach the report queue.
*/
