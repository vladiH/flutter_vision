package com.vladih.computer_vision.flutter_vision

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class TfliteModelInfoTest {
    private fun model(name: String): ByteBuffer {
        // Tests run from the plugin's android/ directory.
        val file = listOf("../example/assets/$name", "example/assets/$name")
            .map(::File).first { it.isFile }
        return RandomAccessFile(file, "r").use {
            it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length())
        }
    }

    @Test
    fun readsYolo11Detection() {
        val info = TfliteModelInfo.parse(model("yolov11n.tflite"))
        val input = info.inputs.single()
        assertContentEquals(intArrayOf(1, 640, 640, 3), input.shape)
        assertEquals(TfliteModelInfo.FLOAT32, input.type)
        val output = info.outputs.single()
        assertContentEquals(intArrayOf(1, 84, 8400), output.shape)
        assertEquals(TfliteModelInfo.FLOAT32, output.type)
    }

    @Test
    fun readsYolov8Segmentation() {
        val info = TfliteModelInfo.parse(model("yolov8n-seg.tflite"))
        assertEquals(1, info.inputs.size)
        assertEquals(2, info.outputs.size)
        val shapes = info.outputs.map { it.shape.toList() }.toSet()
        assertEquals(setOf(listOf(1, 116, 8400), listOf(1, 160, 160, 32)), shapes)
    }

    @Test
    fun rejectsOtherFiles() {
        val notAModel = ByteBuffer.wrap("hello world, not a model".toByteArray())
        assertFailsWith<VisionException> { TfliteModelInfo.parse(notAModel) }
    }
}
