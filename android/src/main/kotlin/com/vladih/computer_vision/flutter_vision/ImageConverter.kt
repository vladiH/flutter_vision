package com.vladih.computer_vision.flutter_vision

/**
 * Converts camera frames to ARGB_8888 pixels. Handles row padding and pixel
 * strides, which the previous version ignored (corrupted live frames on many
 * phones).
 */
internal object ImageConverter {
    /** NV21: Y plane followed by interleaved V/U, both with [rowStride]. */
    fun nv21(data: ByteArray, width: Int, height: Int, rowStride: Int, out: IntArray) {
        val stride = if (rowStride >= width) rowStride else width
        val uvStart = stride * height
        val required = uvStart + stride * ((height + 1) / 2 - 1) + ((width + 1) / 2) * 2
        if (data.size < required) {
            throw VisionException(
                VisionException.INVALID_INPUT,
                "NV21 frame too small: ${data.size} bytes for ${width}x$height (row stride $stride).",
            )
        }
        for (y in 0 until height) {
            val yRow = y * stride
            val uvRow = uvStart + (y shr 1) * stride
            val outRow = y * width
            for (x in 0 until width) {
                val uv = uvRow + (x and 1.inv())
                out[outRow + x] = yuvToArgb(
                    data[yRow + x].toInt() and 0xff,
                    data[uv + 1].toInt() and 0xff,
                    data[uv].toInt() and 0xff,
                )
            }
        }
    }

    /**
     * YUV 4:2:0 in separate planes (Android `YUV_420_888`): U and V with their
     * own row stride and pixel stride (1 = planar, 2 = interleaved).
     */
    fun yuv420(
        y: ByteArray, u: ByteArray, v: ByteArray,
        width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, uvPixelStride: Int,
        out: IntArray,
    ) {
        val lastUv = ((height + 1) / 2 - 1) * uvRowStride + ((width + 1) / 2 - 1) * uvPixelStride
        if (y.size < (height - 1) * yRowStride + width || u.size <= lastUv || v.size <= lastUv) {
            throw VisionException(VisionException.INVALID_INPUT, "YUV planes are smaller than the frame size.")
        }
        for (row in 0 until height) {
            val yRow = row * yRowStride
            val uvRow = (row shr 1) * uvRowStride
            val outRow = row * width
            for (col in 0 until width) {
                val uvIndex = uvRow + (col shr 1) * uvPixelStride
                out[outRow + col] = yuvToArgb(
                    y[yRow + col].toInt() and 0xff,
                    u[uvIndex].toInt() and 0xff,
                    v[uvIndex].toInt() and 0xff,
                )
            }
        }
    }

    /** Y plane plus interleaved U/V plane (NV12), as delivered on iOS. */
    fun nv12(
        y: ByteArray, uv: ByteArray, width: Int, height: Int,
        yRowStride: Int, uvRowStride: Int, out: IntArray,
    ) {
        val lastUv = ((height + 1) / 2 - 1) * uvRowStride + ((width + 1) / 2 - 1) * 2 + 1
        if (y.size < (height - 1) * yRowStride + width || uv.size <= lastUv) {
            throw VisionException(VisionException.INVALID_INPUT, "YUV planes are smaller than the frame size.")
        }
        for (row in 0 until height) {
            val yRow = row * yRowStride
            val uvRow = (row shr 1) * uvRowStride
            val outRow = row * width
            for (col in 0 until width) {
                val uvIndex = uvRow + (col and 1.inv())
                out[outRow + col] = yuvToArgb(
                    y[yRow + col].toInt() and 0xff,
                    uv[uvIndex].toInt() and 0xff,
                    uv[uvIndex + 1].toInt() and 0xff,
                )
            }
        }
    }

    /** 4 bytes per pixel. [bgr] selects BGRA (true) or RGBA (false) order. */
    fun packed32(data: ByteArray, width: Int, height: Int, rowStride: Int, bgr: Boolean, out: IntArray) {
        val stride = if (rowStride >= width * 4) rowStride else width * 4
        if (data.size < (height - 1) * stride + width * 4) {
            throw VisionException(VisionException.INVALID_INPUT, "Frame buffer smaller than ${width}x$height.")
        }
        for (row in 0 until height) {
            var i = row * stride
            val outRow = row * width
            for (col in 0 until width) {
                val c0 = data[i].toInt() and 0xff
                val c1 = data[i + 1].toInt() and 0xff
                val c2 = data[i + 2].toInt() and 0xff
                val r = if (bgr) c2 else c0
                val b = if (bgr) c0 else c2
                out[outRow + col] = (0xff shl 24) or (r shl 16) or (c1 shl 8) or b
                i += 4
            }
        }
    }

    /** Full-range BT.601 (JFIF) YUV to opaque ARGB. */
    fun yuvToArgb(y: Int, u: Int, v: Int): Int {
        val cb = u - 128
        val cr = v - 128
        val r = clamp(y + ((91881 * cr) shr 16))
        val g = clamp(y - ((22554 * cb + 46802 * cr) shr 16))
        val b = clamp(y + ((116130 * cb) shr 16))
        return (0xff shl 24) or (r shl 16) or (g shl 8) or b
    }

    private fun clamp(value: Int): Int = if (value < 0) 0 else if (value > 255) 255 else value
}
