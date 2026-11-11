package com.example.littervision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/** Detections for one camera frame, in the coordinate space of the upright, cropped frame. */
data class DetectionResult(
    val detections: List<Detection>,
    val frameWidth: Int,
    val frameHeight: Int,
    val inferenceMs: Long,
)

/** What the UI should show: the latest result, or an error if the model could not be loaded. */
sealed interface DetectorState {
    data class Ready(val result: DetectionResult) : DetectorState
    data class Failed(val message: String) : DetectorState
}

/**
 * CameraX analyzer running YOLO11n (LiteRT/TFLite) on the executor it is attached to.
 *
 * The camera must use STRATEGY_KEEP_ONLY_LATEST so frames arriving during inference are dropped
 * instead of queued, and OUTPUT_IMAGE_FORMAT_RGBA_8888 so [ImageProxy.toBitmap] is a cheap copy.
 */
class ObjectDetectorAnalyzer(
    context: Context,
    private val onState: (DetectorState) -> Unit,
) : ImageAnalysis.Analyzer {

    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())

    // Created lazily on the analysis thread so model loading never blocks the UI.
    private var detector: YoloDetector? = null
    private var loadFailed = false

    @Volatile
    private var closed = false

    override fun analyze(imageProxy: ImageProxy) {
        try {
            if (closed) return
            val yolo = detector ?: loadDetector() ?: return

            val start = SystemClock.elapsedRealtime()
            val frame = toUprightFrame(imageProxy)
            val detections = yolo.detect(frame)
            val result = DetectionResult(
                detections = detections,
                frameWidth = frame.width,
                frameHeight = frame.height,
                inferenceMs = SystemClock.elapsedRealtime() - start,
            )
            frame.recycle()
            post(DetectorState.Ready(result))
        } catch (t: Throwable) {
            post(DetectorState.Failed("Detection failed: ${t.message}"))
        } finally {
            imageProxy.close()
        }
    }

    private fun loadDetector(): YoloDetector? {
        if (loadFailed) return null
        return try {
            YoloDetector(appContext).also { detector = it }
        } catch (t: Throwable) {
            loadFailed = true
            post(
                DetectorState.Failed(
                    "Could not load ${YoloDetector.MODEL_ASSET} / ${YoloDetector.LABELS_ASSET}: ${t.message}"
                )
            )
            null
        }
    }

    /** Crops to the visible viewport and rotates so the frame is upright. */
    private fun toUprightFrame(imageProxy: ImageProxy): Bitmap {
        val source = imageProxy.toBitmap() // RGBA_8888, unrotated, full buffer
        val crop = imageProxy.cropRect
        val x = crop.left.coerceIn(0, source.width - 1)
        val y = crop.top.coerceIn(0, source.height - 1)
        val w = crop.width().coerceIn(1, source.width - x)
        val h = crop.height().coerceIn(1, source.height - y)
        val matrix = Matrix().apply { postRotate(imageProxy.imageInfo.rotationDegrees.toFloat()) }
        val upright = Bitmap.createBitmap(source, x, y, w, h, matrix, true)
        if (upright !== source) source.recycle()
        return upright
    }

    private fun post(state: DetectorState) {
        mainHandler.post { if (!closed) onState(state) }
    }

    /** Call after the analyzer is cleared from the camera and its executor is shut down. */
    fun close() {
        closed = true
        mainHandler.removeCallbacksAndMessages(null)
        detector?.close()
        detector = null
    }
}
