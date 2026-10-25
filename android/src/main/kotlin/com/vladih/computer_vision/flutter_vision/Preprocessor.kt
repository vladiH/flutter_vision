package com.vladih.computer_vision.flutter_vision

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayInputStream
import kotlin.math.max
import kotlin.math.min

/** Letterbox transform: inputPixel = imagePixel * scale + pad. */
internal data class Letterbox(
    val imageWidth: Int,
    val imageHeight: Int,
    val scale: Float,
    val padX: Float,
    val padY: Float,
) {
    companion object {
        /** Fits [width]x[height] inside the model input, centred, keeping the aspect ratio. */
        fun fit(width: Int, height: Int, inputWidth: Int, inputHeight: Int): Letterbox {
            val scale = min(inputWidth.toFloat() / width, inputHeight.toFloat() / height)
            return Letterbox(
                width, height, scale,
                (inputWidth - width * scale) / 2f,
                (inputHeight - height * scale) / 2f,
            )
        }

        /** Largest power-of-two subsampling that keeps the decoded image at least [target] times the original. */
        fun sampleSize(target: Float): Int {
            var sample = 1
            while (target * sample * 2 <= 1f) sample *= 2
            return sample
        }
    }
}

/**
 * Builds the model input tensor from an encoded image or a camera frame.
 * Owned by one model and used only from that model's thread.
 */
internal class Preprocessor(
    private val inputWidth: Int,
    private val inputHeight: Int,
    private val channelsFirst: Boolean,
    private val int8: Boolean,
) {
    private val canvasBitmap = Bitmap.createBitmap(inputWidth, inputHeight, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(canvasBitmap)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG or Paint.DITHER_FLAG)
    private val pixels = IntArray(inputWidth * inputHeight)
    private var frameBitmap: Bitmap? = null
    private var framePixels: IntArray? = null

    val floatInput: FloatArray? = if (int8) null else FloatArray(inputWidth * inputHeight * 3)
    val int8Input: ByteArray? = if (int8) ByteArray(inputWidth * inputHeight * 3) else null

    /** Decodes [bytes], applies the EXIF orientation and fills the input. */
    fun fromEncoded(bytes: ByteArray): Letterbox {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw VisionException(VisionException.INVALID_INPUT, "Cannot decode the image (unknown or corrupted format).")
        }
        val orientation = try {
            ExifInterface(ByteArrayInputStream(bytes))
                .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } catch (_: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        val transposed = orientation in TRANSPOSED
        val uprightWidth = if (transposed) bounds.outHeight else bounds.outWidth
        val uprightHeight = if (transposed) bounds.outWidth else bounds.outHeight

        val fit = Letterbox.fit(uprightWidth, uprightHeight, inputWidth, inputHeight)
        val options = BitmapFactory.Options().apply {
            inSampleSize = Letterbox.sampleSize(fit.scale)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
            ?: throw VisionException(VisionException.INVALID_INPUT, "Cannot decode the image.")
        try {
            // The bitmap may be subsampled: express everything in original pixels.
            val sampled = bitmap.width.toFloat() / bounds.outWidth
            val matrix = orientationMatrix(orientation, bitmap.width, bitmap.height)
            matrix.postScale(fit.scale / sampled, fit.scale / sampled)
            matrix.postTranslate(fit.padX, fit.padY)
            draw(bitmap, matrix)
        } finally {
            bitmap.recycle()
        }
        return fit
    }

    /** Converts a camera frame, rotates it upright and fills the input. */
    fun fromFrame(frame: FrameMessage): Letterbox {
        val width = frame.width.toInt()
        val height = frame.height.toInt()
        if (width <= 0 || height <= 0 || frame.planes.isEmpty()) {
            throw VisionException(VisionException.INVALID_INPUT, "Empty camera frame.")
        }
        val rotation = ((frame.rotationDegrees.toInt() % 360) + 360) % 360
        val argb = framePixels?.takeIf { it.size == width * height } ?: IntArray(width * height).also { framePixels = it }
        val planes = frame.planes
        when (frame.format) {
            FrameFormatMessage.NV21 ->
                ImageConverter.nv21(planes[0].bytes, width, height, planes[0].bytesPerRow.toInt(), argb)
            FrameFormatMessage.YUV420 -> when (planes.size) {
                3 -> ImageConverter.yuv420(
                    planes[0].bytes, planes[1].bytes, planes[2].bytes, width, height,
                    planes[0].bytesPerRow.toInt(), planes[1].bytesPerRow.toInt(),
                    max(1, planes[1].bytesPerPixel.toInt()), argb,
                )
                2 -> ImageConverter.nv12(
                    planes[0].bytes, planes[1].bytes, width, height,
                    planes[0].bytesPerRow.toInt(), planes[1].bytesPerRow.toInt(), argb,
                )
                else -> throw VisionException(VisionException.INVALID_INPUT, "YUV420 frames need 2 or 3 planes, got ${planes.size}.")
            }
            FrameFormatMessage.BGRA8888 ->
                ImageConverter.packed32(planes[0].bytes, width, height, planes[0].bytesPerRow.toInt(), true, argb)
            FrameFormatMessage.RGBA8888 ->
                ImageConverter.packed32(planes[0].bytes, width, height, planes[0].bytesPerRow.toInt(), false, argb)
        }
        val bitmap = frameBitmap?.takeIf { it.width == width && it.height == height }
            ?: Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
                frameBitmap?.recycle()
                frameBitmap = it
            }
        bitmap.setPixels(argb, 0, width, 0, 0, width, height)

        val uprightWidth = if (rotation % 180 == 0) width else height
        val uprightHeight = if (rotation % 180 == 0) height else width
        val fit = Letterbox.fit(uprightWidth, uprightHeight, inputWidth, inputHeight)
        val matrix = Matrix().apply {
            postTranslate(-width / 2f, -height / 2f)
            postRotate(rotation.toFloat())
            postTranslate(uprightWidth / 2f, uprightHeight / 2f)
            postScale(fit.scale, fit.scale)
            postTranslate(fit.padX, fit.padY)
        }
        draw(bitmap, matrix)
        return fit
    }

    fun release() {
        canvasBitmap.recycle()
        frameBitmap?.recycle()
        frameBitmap = null
        framePixels = null
    }

    private fun draw(bitmap: Bitmap, matrix: Matrix) {
        canvas.drawColor(PAD_COLOR)
        canvas.drawBitmap(bitmap, matrix, paint)
        canvasBitmap.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        fillTensor(pixels, inputWidth * inputHeight, channelsFirst, floatInput, int8Input)
    }

    companion object {
        /** Ultralytics letterbox grey (114, 114, 114). */
        private const val PAD_COLOR = 0xFF727272.toInt()

        private val TRANSPOSED = setOf(
            ExifInterface.ORIENTATION_TRANSPOSE,
            ExifInterface.ORIENTATION_ROTATE_90,
            ExifInterface.ORIENTATION_TRANSVERSE,
            ExifInterface.ORIENTATION_ROTATE_270,
        )

        /** Maps bitmap pixels to the upright image for an EXIF orientation. */
        fun orientationMatrix(orientation: Int, width: Int, height: Int): Matrix {
            val transposed = orientation in TRANSPOSED
            val outWidth = if (transposed) height else width
            val outHeight = if (transposed) width else height
            return Matrix().apply {
                postTranslate(-width / 2f, -height / 2f)
                when (orientation) {
                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> postScale(-1f, 1f)
                    ExifInterface.ORIENTATION_ROTATE_180 -> postRotate(180f)
                    ExifInterface.ORIENTATION_FLIP_VERTICAL -> postScale(1f, -1f)
                    ExifInterface.ORIENTATION_TRANSPOSE -> { postRotate(90f); postScale(-1f, 1f) }
                    ExifInterface.ORIENTATION_ROTATE_90 -> postRotate(90f)
                    ExifInterface.ORIENTATION_TRANSVERSE -> { postRotate(270f); postScale(-1f, 1f) }
                    ExifInterface.ORIENTATION_ROTATE_270 -> postRotate(270f)
                }
                postTranslate(outWidth / 2f, outHeight / 2f)
            }
        }

        /**
         * Writes ARGB [pixels] as RGB into the tensor: float32 in 0..1, or
         * int8 as pixel - 128 (Ultralytics full-integer models).
         */
        fun fillTensor(pixels: IntArray, count: Int, channelsFirst: Boolean, floats: FloatArray?, bytes: ByteArray?) {
            val inv = 1f / 255f
            for (i in 0 until count) {
                val p = pixels[i]
                val r = (p shr 16) and 0xff
                val g = (p shr 8) and 0xff
                val b = p and 0xff
                val ir: Int
                val ig: Int
                val ib: Int
                if (channelsFirst) {
                    ir = i; ig = count + i; ib = 2 * count + i
                } else {
                    ir = 3 * i; ig = ir + 1; ib = ir + 2
                }
                if (floats != null) {
                    floats[ir] = r * inv
                    floats[ig] = g * inv
                    floats[ib] = b * inv
                } else if (bytes != null) {
                    bytes[ir] = (r - 128).toByte()
                    bytes[ig] = (g - 128).toByte()
                    bytes[ib] = (b - 128).toByte()
                }
            }
        }
    }
}
