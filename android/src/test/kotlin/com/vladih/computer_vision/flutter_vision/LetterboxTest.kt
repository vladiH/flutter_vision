package com.vladih.computer_vision.flutter_vision

import kotlin.test.Test
import kotlin.test.assertEquals

class LetterboxTest {
    @Test
    fun fitsLandscapeImage() {
        val fit = Letterbox.fit(1280, 720, 640, 640)
        assertEquals(0.5f, fit.scale)
        assertEquals(0f, fit.padX)
        assertEquals(140f, fit.padY)
    }

    @Test
    fun fitsPortraitImageInNonSquareInput() {
        val fit = Letterbox.fit(480, 640, 320, 320)
        assertEquals(0.5f, fit.scale)
        assertEquals(40f, fit.padX)
        assertEquals(0f, fit.padY)
    }

    @Test
    fun sampleSizeKeepsAtLeastTheTargetResolution() {
        assertEquals(1, Letterbox.sampleSize(1.5f))
        assertEquals(1, Letterbox.sampleSize(0.6f))
        assertEquals(2, Letterbox.sampleSize(0.4f))
        assertEquals(4, Letterbox.sampleSize(0.16f))
    }

    @Test
    fun fillTensorLayouts() {
        val pixels = intArrayOf(0xff0a141e.toInt(), 0xffffffff.toInt()) // (10,20,30), white
        val eps = 1e-6f
        val nhwc = FloatArray(6)
        Preprocessor.fillTensor(pixels, 2, channelsFirst = false, floats = nhwc, bytes = null)
        assertEquals(10 / 255f, nhwc[0], eps)
        assertEquals(30 / 255f, nhwc[2], eps)
        assertEquals(1f, nhwc[5], eps)

        val nchw = FloatArray(6)
        Preprocessor.fillTensor(pixels, 2, channelsFirst = true, floats = nchw, bytes = null)
        assertEquals(10 / 255f, nchw[0], eps)
        assertEquals(20 / 255f, nchw[2], eps)
        assertEquals(30 / 255f, nchw[4], eps)

        val int8 = ByteArray(6)
        Preprocessor.fillTensor(pixels, 2, channelsFirst = false, floats = null, bytes = int8)
        assertEquals((10 - 128).toByte(), int8[0])
        assertEquals(127.toByte(), int8[5])
    }
}
