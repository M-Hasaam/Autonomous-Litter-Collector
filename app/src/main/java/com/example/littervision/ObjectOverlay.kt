package com.example.littervision

import android.graphics.Color as AndroidColor
import android.graphics.Paint as AndroidPaint
import android.graphics.RectF
import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
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
 * Overlay composable that draws YOLO detection boxes, class names and confidence scores.
 *
 * Detection boxes are normalized to the upright, cropped analysis frame, which has the same
 * field of view as the PreviewView, so only FILL_CENTER scaling and front-camera mirroring
 * need to be applied here.
 */
@Composable
fun ObjectOverlay(
    detectionResult: DetectionResult?,
    modifier: Modifier = Modifier,
    isFrontFacing: Boolean = false,
) {
    Canvas(modifier = modifier) {
        val result = detectionResult ?: return@Canvas
        if (result.detections.isEmpty() || result.frameWidth <= 0 || result.frameHeight <= 0) return@Canvas

        val viewWidth = size.width
        val viewHeight = size.height
        val frameWidth = result.frameWidth.toFloat()
        val frameHeight = result.frameHeight.toFloat()

        // PreviewView FILL_CENTER: uniform scale to cover the view, centered.
        val scale = max(viewWidth / frameWidth, viewHeight / frameHeight)
        val offsetX = (viewWidth - frameWidth * scale) / 2f
        val offsetY = (viewHeight - frameHeight * scale) / 2f

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

        for (det in result.detections) {
            val boxLeft = det.box.left * frameWidth
            val boxRight = det.box.right * frameWidth
            val boxTop = det.box.top * frameHeight
            val boxBottom = det.box.bottom * frameHeight

            val mappedLeft = if (isFrontFacing) {
                (frameWidth - boxRight) * scale + offsetX
            } else {
                boxLeft * scale + offsetX
            }
            val mappedRight = if (isFrontFacing) {
                (frameWidth - boxLeft) * scale + offsetX
            } else {
                boxRight * scale + offsetX
            }
            val mappedTop = boxTop * scale + offsetY
            val mappedBottom = boxBottom * scale + offsetY

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

            val labelText = "${det.label} ${(det.confidence * 100).toInt()}%"

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
