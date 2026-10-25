package com.vladih.computer_vision.flutter_vision

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ImageConverterTest {
    private fun rgb(argb: Int) = Triple((argb shr 16) and 0xff, (argb shr 8) and 0xff, argb and 0xff)

    @Test
    fun yuvToArgbPrimaries() {
        assertEquals(Triple(0, 0, 0), rgb(ImageConverter.yuvToArgb(0, 128, 128)))
        assertEquals(Triple(255, 255, 255), rgb(ImageConverter.yuvToArgb(255, 128, 128)))
        // Pure red in full-range BT.601: Y=76, U=85, V=255.
        val (r, g, b) = rgb(ImageConverter.yuvToArgb(76, 85, 255))
        assertTrue(r > 240 && g < 15 && b < 15, "got $r $g $b")
    }

    @Test
    fun nv21WithRowPadding() {
        // 2x2 image, row stride 4: Y rows padded, then one VU row.
        val data = byteArrayOf(
            10, 20, 0, 0,
            30, 40, 0, 0,
            128.toByte(), 128.toByte(), 0, 0,
        )
        val out = IntArray(4)
        ImageConverter.nv21(data, 2, 2, 4, out)
        assertEquals(listOf(10, 20, 30, 40), out.map { rgb(it).first })
    }

    @Test
    fun yuv420PlanarAndInterleavedAgree() {
        val width = 4
        val height = 2
        val y = ByteArray(width * height) { (it * 20).toByte() }
        // Planar U/V (pixel stride 1).
        val u = byteArrayOf(100, 110)
        val v = byteArrayOf(150.toByte(), 160.toByte())
        val planar = IntArray(width * height)
        ImageConverter.yuv420(y, u, v, width, height, width, 2, 1, planar)
        // Interleaved U/V (pixel stride 2), as on most Android devices.
        val ui = byteArrayOf(100, 0, 110)
        val vi = byteArrayOf(150.toByte(), 0, 160.toByte())
        val interleaved = IntArray(width * height)
        ImageConverter.yuv420(y, ui, vi, width, height, width, 4, 2, interleaved)
        assertEquals(planar.toList(), interleaved.toList())
    }

    @Test
    fun bgraAndRgba() {
        val pixel = byteArrayOf(1, 2, 3, 255.toByte())
        val out = IntArray(1)
        ImageConverter.packed32(pixel, 1, 1, 4, bgr = true, out = out)
        assertEquals(Triple(3, 2, 1), rgb(out[0]))
        ImageConverter.packed32(pixel, 1, 1, 4, bgr = false, out = out)
        assertEquals(Triple(1, 2, 3), rgb(out[0]))
    }

    @Test
    fun rejectsShortBuffers() {
        assertFailsWith<VisionException> { ImageConverter.nv21(ByteArray(3), 2, 2, 2, IntArray(4)) }
        assertFailsWith<VisionException> { ImageConverter.packed32(ByteArray(3), 1, 1, 4, true, IntArray(1)) }
    }
}
