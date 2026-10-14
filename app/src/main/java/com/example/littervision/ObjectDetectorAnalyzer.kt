package com.example.littervision

import android.graphics.Rect
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.DetectedObject
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions

/**
 * Data class holding detection results for a single camera frame.
 * Includes cropRect to account for CameraX viewport cropping.
 */
data class DetectionResult(
    val objects: List<DetectedObject>,
    val imageWidth: Int,
    val imageHeight: Int,
    val cropRect: Rect,
    val rotationDegrees: Int
)

/**
 * CameraX ImageAnalysis analyzer using Google ML Kit On-Device Object Detection.
 *
 * Configured in STREAM_MODE with multiple object detection and classification enabled.
 */
class ObjectDetectorAnalyzer(
    private val onResult: (DetectionResult) -> Unit
) : ImageAnalysis.Analyzer {

    private val options = ObjectDetectorOptions.Builder()
        .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
        .enableMultipleObjects()
        .enableClassification()
        .build()

    private val detector: ObjectDetector = ObjectDetection.getClient(options)

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            val cropRect = Rect(imageProxy.cropRect)
            val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)

            detector.process(inputImage)
                .addOnSuccessListener { detectedObjects ->
                    onResult(
                        DetectionResult(
                            objects = detectedObjects,
                            imageWidth = imageProxy.width,
                            imageHeight = imageProxy.height,
                            cropRect = cropRect,
                            rotationDegrees = rotationDegrees
                        )
                    )
                }
                .addOnFailureListener {
                    // Failures are gracefully handled; imageProxy is closed in onCompleteListener below
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        } else {
            imageProxy.close()
        }
    }

    fun close() {
        detector.close()
    }
}
