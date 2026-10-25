package com.vladih.computer_vision.flutter_vision

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Input and output tensors of a `.tflite` model, read directly from the
 * FlatBuffer. The LiteRT Kotlin API needs tensor names but cannot list them,
 * and does not expose quantization parameters.
 */
internal class TfliteModelInfo(
    /** Signature to run, or "" when the model has no signatures. */
    val signatureKey: String,
    val inputs: List<TensorDesc>,
    val outputs: List<TensorDesc>,
) {
    class TensorDesc(
        val name: String,
        val shape: IntArray,
        val type: Int,
        val scale: Float,
        val zeroPoint: Long,
    ) {
        val elementCount: Int get() = shape.fold(1) { acc, d -> acc * d }
    }

    companion object {
        // TensorType values from the TFLite schema.
        const val FLOAT32 = 0
        const val FLOAT16 = 1
        const val INT32 = 2
        const val UINT8 = 3
        const val INT64 = 4
        const val INT8 = 9

        fun parse(source: ByteBuffer): TfliteModelInfo {
            val fb = FlatBuffer(source.duplicate().order(ByteOrder.LITTLE_ENDIAN))
            if (source.capacity() < 8 || fb.identifier() != "TFL3") {
                throw VisionException(
                    VisionException.UNSUPPORTED_MODEL,
                    "Not a TensorFlow Lite model (.tflite).",
                )
            }
            val model = fb.root()
            val subgraphs = fb.vector(model, 2)
                ?: throw VisionException(VisionException.UNSUPPORTED_MODEL, "The model has no subgraph.")

            val signatures = fb.vector(model, 7)
            if (signatures != null && fb.vectorLength(signatures) > 0) {
                val signature = fb.tableAt(signatures, 0)
                val key = fb.string(signature, 2) ?: ""
                val subgraphIndex = fb.uint(signature, 4)
                val subgraph = fb.tableAt(subgraphs, subgraphIndex)
                val tensors = fb.vector(subgraph, 0)
                    ?: throw VisionException(VisionException.UNSUPPORTED_MODEL, "The model has no tensors.")
                fun maps(field: Int): List<TensorDesc> {
                    val vec = fb.vector(signature, field) ?: return emptyList()
                    return (0 until fb.vectorLength(vec)).map { i ->
                        val map = fb.tableAt(vec, i)
                        val name = fb.string(map, 0) ?: ""
                        tensor(fb, fb.tableAt(tensors, fb.uint(map, 1)), name)
                    }
                }
                return TfliteModelInfo(key, maps(0), maps(1))
            }

            val subgraph = fb.tableAt(subgraphs, 0)
            val tensors = fb.vector(subgraph, 0)
                ?: throw VisionException(VisionException.UNSUPPORTED_MODEL, "The model has no tensors.")
            fun indices(field: Int): List<TensorDesc> {
                val vec = fb.vector(subgraph, field) ?: return emptyList()
                return (0 until fb.vectorLength(vec)).map { i ->
                    tensor(fb, fb.tableAt(tensors, fb.intAt(vec, i)), null)
                }
            }
            return TfliteModelInfo("", indices(1), indices(2))
        }

        private fun tensor(fb: FlatBuffer, table: Int, signatureName: String?): TensorDesc {
            val shapeVec = fb.vector(table, 0)
            val shape = if (shapeVec == null) IntArray(0) else
                IntArray(fb.vectorLength(shapeVec)) { fb.intAt(shapeVec, it) }
            val type = fb.byte(table, 1, FLOAT32.toByte()).toInt()
            val tensorName = fb.string(table, 3) ?: ""
            var scale = 0f
            var zeroPoint = 0L
            val quant = fb.table(table, 4)
            if (quant != null) {
                fb.vector(quant, 2)?.let { if (fb.vectorLength(it) > 0) scale = fb.floatAt(it, 0) }
                fb.vector(quant, 3)?.let { if (fb.vectorLength(it) > 0) zeroPoint = fb.longAt(it, 0) }
            }
            return TensorDesc(signatureName ?: tensorName, shape, type, scale, zeroPoint)
        }
    }
}

/** Minimal read-only FlatBuffers accessor. Positions are absolute offsets. */
private class FlatBuffer(private val bb: ByteBuffer) {
    fun identifier(): String = String(ByteArray(4) { bb.get(4 + it) }, Charsets.US_ASCII)

    fun root(): Int = bb.getInt(0)

    private fun field(table: Int, index: Int): Int {
        val vtable = table - bb.getInt(table)
        val vtableSize = bb.getShort(vtable).toInt() and 0xffff
        val entry = 4 + index * 2
        if (entry >= vtableSize) return 0
        val offset = bb.getShort(vtable + entry).toInt() and 0xffff
        return if (offset == 0) 0 else table + offset
    }

    private fun indirect(position: Int): Int = position + bb.getInt(position)

    fun table(table: Int, index: Int): Int? {
        val f = field(table, index)
        return if (f == 0) null else indirect(f)
    }

    fun vector(table: Int, index: Int): Int? = table(table, index)

    fun vectorLength(vector: Int): Int = bb.getInt(vector)

    fun tableAt(vector: Int, i: Int): Int {
        if (i < 0 || i >= vectorLength(vector)) {
            throw VisionException(VisionException.UNSUPPORTED_MODEL, "Corrupted model file.")
        }
        return indirect(vector + 4 + i * 4)
    }

    fun intAt(vector: Int, i: Int): Int = bb.getInt(vector + 4 + i * 4)
    fun floatAt(vector: Int, i: Int): Float = bb.getFloat(vector + 4 + i * 4)
    fun longAt(vector: Int, i: Int): Long = bb.getLong(vector + 4 + i * 8)

    fun string(table: Int, index: Int): String? {
        val f = field(table, index)
        if (f == 0) return null
        val s = indirect(f)
        val length = bb.getInt(s)
        return String(ByteArray(length) { bb.get(s + 4 + it) }, Charsets.UTF_8)
    }

    fun uint(table: Int, index: Int): Int {
        val f = field(table, index)
        return if (f == 0) 0 else bb.getInt(f)
    }

    fun byte(table: Int, index: Int, default: Byte): Byte {
        val f = field(table, index)
        return if (f == 0) default else bb.get(f)
    }
}
