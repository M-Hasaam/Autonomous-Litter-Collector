package com.example.littervision

import android.graphics.Color as AndroidColor
import android.graphics.Paint as AndroidPaint
import android.graphics.RectF
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.unit.dp
import kotlin.math.max

/**
 * Helper class to smooth bounding box coordinates across video stream frames using EMA.
 */
private class BoxSmoother {
    private val prevBoxes = mutableMapOf<Int, RectF>()

    fun smooth(trackingId: Int?, newBox: RectF): RectF {
        if (trackingId == null) return newBox
        val prev = prevBoxes[trackingId]
        if (prev == null) {
            prevBoxes[trackingId] = newBox
            return newBox
        }
        val alpha = 0.45f // Smooth factor for camera tracking
        val smoothed = RectF(
            prev.left + alpha * (newBox.left - prev.left),
            prev.top + alpha * (newBox.top - prev.top),
            prev.right + alpha * (newBox.right - prev.right),
            prev.bottom + alpha * (newBox.bottom - prev.bottom)
        )
        prevBoxes[trackingId] = smoothed
        return smoothed
    }

    fun cleanup(currentIds: Set<Int>) {
        prevBoxes.keys.retainAll(currentIds)
    }
}

/**
 * Overlay composable that draws accurate bounding boxes and labels over detected objects.
 *
 * Accounts for camera image rotation, CameraX cropRect, PreviewView FILL_CENTER scaling,
 * and optional front-camera mirroring.
 */
@Composable
fun ObjectOverlay(
    detectionResult: DetectionResult?,
    modifier: Modifier = Modifier,
    isFrontFacing: Boolean = false,
) {
    val boxSmoother = remember { BoxSmoother() }

    Canvas(modifier = modifier) {
        val result = detectionResult ?: return@Canvas
        if (result.objects.isEmpty() || result.imageWidth <= 0 || result.imageHeight <= 0) return@Canvas

        val viewWidth = size.width
        val viewHeight = size.height

        val rawWidth = result.imageWidth
        val rawHeight = result.imageHeight
        val crop = result.cropRect

        // Transform cropRect into rotated image coordinate space
        val rotatedCrop = when (result.rotationDegrees) {
            90 -> RectF(
                crop.top.toFloat(),
                (rawWidth - crop.right).toFloat(),
                crop.bottom.toFloat(),
                (rawWidth - crop.left).toFloat()
            )
            180 -> RectF(
                (rawWidth - crop.right).toFloat(),
                (rawHeight - crop.bottom).toFloat(),
                (rawWidth - crop.left).toFloat(),
                (rawHeight - crop.top).toFloat()
            )
            270 -> RectF(
                (rawHeight - crop.bottom).toFloat(),
                crop.left.toFloat(),
                (rawHeight - crop.top).toFloat(),
                crop.right.toFloat()
            )
            else -> RectF(
                crop.left.toFloat(),
                crop.top.toFloat(),
                crop.right.toFloat(),
                crop.bottom.toFloat()
            )
        }

        val visibleWidth = max(rotatedCrop.width(), 1f)
        val visibleHeight = max(rotatedCrop.height(), 1f)

        // PreviewView FILL_CENTER scale & offset relative to the visible crop
        val scaleX = viewWidth / visibleWidth
        val scaleY = viewHeight / visibleHeight
        val scale = max(scaleX, scaleY)

        val offsetX = (viewWidth - visibleWidth * scale) / 2f
        val offsetY = (viewHeight - visibleHeight * scale) / 2f

        val strokeWidthPx = 3.dp.toPx()

        val textPaint = AndroidPaint().apply {
            color = AndroidColor.WHITE
            textSize = 14.dp.toPx()
            isAntiAlias = true
            style = AndroidPaint.Style.FILL
            isFakeBoldText = true
        }

        val textBgPaint = AndroidPaint().apply {
            color = AndroidColor.argb(180, 0, 150, 136) // Semi-transparent teal background for label tag
            style = AndroidPaint.Style.FILL
            isAntiAlias = true
        }

        val boxColor = Color(0xFF00E676) // Bright green box accent

        val currentTrackingIds = result.objects.mapNotNull { it.trackingId }.toSet()
        boxSmoother.cleanup(currentTrackingIds)

        for (obj in result.objects) {
            val rawBox = RectF(obj.boundingBox)

            // Convert raw box relative to rotated crop region
            val boxInCrop = RectF(
                rawBox.left - rotatedCrop.left,
                rawBox.top - rotatedCrop.top,
                rawBox.right - rotatedCrop.left,
                rawBox.bottom - rotatedCrop.top
            )

            val smoothedBox = boxSmoother.smooth(obj.trackingId, boxInCrop)

            val mappedLeft = if (isFrontFacing) {
                (visibleWidth - smoothedBox.right) * scale + offsetX
            } else {
                smoothedBox.left * scale + offsetX
            }

            val mappedRight = if (isFrontFacing) {
                (visibleWidth - smoothedBox.left) * scale + offsetX
            } else {
                smoothedBox.right * scale + offsetX
            }

            val mappedTop = smoothedBox.top * scale + offsetY
            val mappedBottom = smoothedBox.bottom * scale + offsetY

            // Draw bounding box
            drawRoundRect(
                color = boxColor,
                topLeft = Offset(mappedLeft, mappedTop),
                size = Size(
                    width = max(mappedRight - mappedLeft, 1f),
                    height = max(mappedBottom - mappedTop, 1f)
                ),
                cornerRadius = CornerRadius(8.dp.toPx()),
                style = Stroke(width = strokeWidthPx)
            )

            // Format label text and confidence
            val primaryLabel = obj.labels.maxByOrNull { it.confidence }
            val labelText = if (primaryLabel != null) {
                val confidencePct = (primaryLabel.confidence * 100).toInt()
                "${primaryLabel.text} $confidencePct%"
            } else {
                "Object"
            }

            // Calculate label text dimensions for background pill
            val textBounds = android.graphics.Rect()
            textPaint.getTextBounds(labelText, 0, labelText.length, textBounds)
            val textWidth = textBounds.width().toFloat()
            val textHeight = textBounds.height().toFloat()

            val padding = 6.dp.toPx()
            val textTop = max(mappedTop - textHeight - padding * 2, 0f)
            val textBottom = textTop + textHeight + padding * 2

            val bgRect = RectF(
                mappedLeft,
                textTop,
                mappedLeft + textWidth + padding * 2,
                textBottom
            )

            // Draw label background pill and label text
            drawContext.canvas.nativeCanvas.drawRoundRect(bgRect, 8.dp.toPx(), 8.dp.toPx(), textBgPaint)
            drawContext.canvas.nativeCanvas.drawText(
                labelText,
                mappedLeft + padding,
                textBottom - padding,
                textPaint
            )
        }
    }
}
