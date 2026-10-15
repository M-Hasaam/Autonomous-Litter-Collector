package com.example.littervision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/** One detection. [box] is normalized (0..1) relative to the upright, cropped camera frame. */
data class Detection(
    val label: String,
    val confidence: Float,
    val box: RectF,
)

/**
 * Runs an Ultralytics YOLO11 detector exported with `training/export_model.py`
 * (float32 TFLite, NHWC input [1, S, S, 3] scaled to 0..1, output [1, 4 + numClasses, N]
 * with rows cx, cy, w, h in input-image pixels followed by per-class scores; no objectness and no built-in NMS).
 *
 * Not thread-safe: call [detect] from a single thread.
 */
class YoloDetector(
    context: Context,
    modelAsset: String = MODEL_ASSET,
    labelsAsset: String = LABELS_ASSET,
    private val confidenceThreshold: Float = 0.25f,
    private val iouThreshold: Float = 0.45f,
) : Closeable {

    private val labels: List<String> = context.assets.open(labelsAsset).bufferedReader().useLines { lines ->
        lines.map { it.trim() }.filter { it.isNotEmpty() }.toList()
    }

    private val interpreter: Interpreter
    private val inputWidth: Int
    private val inputHeight: Int

    // Output layout, validated against the label file when the model is loaded.
    private val channelsFirst: Boolean
    private val numAnchors: Int
    private val numChannels: Int

    private val inputBuffer: ByteBuffer
    private val outputBuffer: Array<Array<FloatArray>>
    private val pixels: IntArray
    private val letterboxBitmap: Bitmap
    private val letterboxCanvas: Canvas
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    init {
        val modelBytes = context.assets.open(modelAsset).use { it.readBytes() }
        val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        modelBuffer.put(modelBytes).rewind()

        val options = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(modelBuffer, options)

        val inShape = interpreter.getInputTensor(0).shape() // [1, H, W, 3]
        require(inShape.size == 4 && inShape[3] == 3) {
            "Expected NHWC input [1,H,W,3] but model has ${inShape.toList()}"
        }
        inputHeight = inShape[1]
        inputWidth = inShape[2]

        val outShape = interpreter.getOutputTensor(0).shape() // [1, 4+nc, N] for Ultralytics exports
        require(outShape.size == 3 && outShape[0] == 1) {
            "Unexpected output shape ${outShape.toList()}; export the model with training/export_model.py"
        }
        val expectedChannels = 4 + labels.size
        channelsFirst = when {
            outShape[1] == expectedChannels -> true
            outShape[2] == expectedChannels -> false
            else -> error(
                "Model output ${outShape.toList()} does not match ${labels.size} labels in $labelsAsset " +
                    "(expected a dimension of $expectedChannels)"
            )
        }
        numChannels = expectedChannels
        numAnchors = if (channelsFirst) outShape[2] else outShape[1]
        outputBuffer = Array(1) { Array(outShape[1]) { FloatArray(outShape[2]) } }

        inputBuffer = ByteBuffer.allocateDirect(4 * inputWidth * inputHeight * 3).order(ByteOrder.nativeOrder())
        pixels = IntArray(inputWidth * inputHeight)
        letterboxBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
        letterboxCanvas = Canvas(letterboxBitmap)
    }

    /**
     * Detects objects in an upright (already rotated and cropped) [frame].
     * Returned boxes are normalized to the frame, so they are independent of the model input size.
     */
    fun detect(frame: Bitmap): List<Detection> {
        val frameW = frame.width.toFloat()
        val frameH = frame.height.toFloat()
        val scale = min(inputWidth / frameW, inputHeight / frameH)
        val newW = frameW * scale
        val newH = frameH * scale
        val padX = (inputWidth - newW) / 2f
        val padY = (inputHeight - newH) / 2f

        // Letterbox with Ultralytics' gray (114) padding, then pack RGB floats in 0..1.
        letterboxCanvas.drawColor(Color.rgb(114, 114, 114))
        letterboxCanvas.drawBitmap(
            frame, null, RectF(padX, padY, padX + newW, padY + newH), paint
        )
        letterboxBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        inputBuffer.rewind()
        for (p in pixels) {
            inputBuffer.putFloat(((p shr 16) and 0xFF) / 255f)
            inputBuffer.putFloat(((p shr 8) and 0xFF) / 255f)
            inputBuffer.putFloat((p and 0xFF) / 255f)
        }
        inputBuffer.rewind()

        interpreter.run(inputBuffer, outputBuffer)
        return decode(outputBuffer[0], scale, padX, padY, frameW, frameH)
    }

    private fun decode(
        out: Array<FloatArray>,
        scale: Float,
        padX: Float,
        padY: Float,
        frameW: Float,
        frameH: Float,
    ): List<Detection> {
        fun value(channel: Int, anchor: Int) = if (channelsFirst) out[channel][anchor] else out[anchor][channel]

        val candidates = ArrayList<Detection>()
        val classIds = ArrayList<Int>()
        for (a in 0 until numAnchors) {
            var bestClass = -1
            var bestScore = confidenceThreshold
            for (c in 4 until numChannels) {
                val s = value(c, a)
                if (s > bestScore) {
                    bestScore = s
                    bestClass = c - 4
                }
            }
            if (bestClass < 0) continue

            val cx = value(0, a)
            val cy = value(1, a)
            val w = value(2, a)
            val h = value(3, a)

            // Undo letterbox -> frame pixels -> normalized frame coordinates.
            val left = ((cx - w / 2f - padX) / scale / frameW).coerceIn(0f, 1f)
            val top = ((cy - h / 2f - padY) / scale / frameH).coerceIn(0f, 1f)
            val right = ((cx + w / 2f - padX) / scale / frameW).coerceIn(0f, 1f)
            val bottom = ((cy + h / 2f - padY) / scale / frameH).coerceIn(0f, 1f)
            if (right <= left || bottom <= top) continue

            candidates += Detection(labels[bestClass], bestScore, RectF(left, top, right, bottom))
            classIds += bestClass
        }
        return nonMaxSuppression(candidates, classIds)
    }

    /** Greedy per-class NMS, highest confidence first. */
    private fun nonMaxSuppression(dets: List<Detection>, classIds: List<Int>): List<Detection> {
        val order = dets.indices.sortedByDescending { dets[it].confidence }
        val kept = ArrayList<Int>()
        for (i in order) {
            var keep = true
            for (j in kept) {
                if (classIds[i] == classIds[j] && iou(dets[i].box, dets[j].box) > iouThreshold) {
                    keep = false
                    break
                }
            }
            if (keep) {
                kept += i
                if (kept.size >= MAX_DETECTIONS) break
            }
        }
        return kept.map { dets[it] }
    }

    private fun iou(a: RectF, b: RectF): Float {
        val iw = min(a.right, b.right) - max(a.left, b.left)
        val ih = min(a.bottom, b.bottom) - max(a.top, b.top)
        if (iw <= 0f || ih <= 0f) return 0f
        val inter = iw * ih
        return inter / (a.width() * a.height() + b.width() * b.height() - inter)
    }

    override fun close() {
        interpreter.close()
        letterboxBitmap.recycle()
    }

    companion object {
        const val MODEL_ASSET = "litter_detector.tflite"
        const val LABELS_ASSET = "labels.txt"
        private const val MAX_DETECTIONS = 50
    }
}
