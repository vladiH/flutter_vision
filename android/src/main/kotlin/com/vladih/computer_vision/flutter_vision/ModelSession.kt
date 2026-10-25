package com.vladih.computer_vision.flutter_vision

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import io.flutter.embedding.engine.plugins.FlutterPlugin
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException

/**
 * One loaded model. Loading, inference and release all run on the model's own
 * thread, in order, so a model is never closed while it is running and GPU
 * work always stays on the thread that created it.
 */
internal class ModelSession(
    val handle: Long,
    private val context: Context,
    private val assets: FlutterPlugin.FlutterAssets,
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "flutter_vision_model_$handle")
    }
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var closed = false

    private var environment: Environment? = null
    private var model: CompiledModel? = null
    private var inputBuffers: Map<String, TensorBuffer> = emptyMap()
    private var outputBuffers: Map<String, TensorBuffer> = emptyMap()
    private var preprocessor: Preprocessor? = null
    private lateinit var info: TfliteModelInfo

    fun load(request: LoadModelRequest, callback: (Result<ModelInfoMessage>) -> Unit) =
        submit(callback, VisionException.LOAD_FAILED) { loadOnThread(request) }

    fun runImage(bytes: ByteArray, callback: (Result<InferenceResultMessage>) -> Unit) =
        submit(callback, VisionException.INFERENCE_FAILED) {
            infer { preprocessor!!.fromEncoded(bytes) }
        }

    fun runFrame(frame: FrameMessage, callback: (Result<InferenceResultMessage>) -> Unit) =
        submit(callback, VisionException.INFERENCE_FAILED) {
            infer { preprocessor!!.fromFrame(frame) }
        }

    /** Releases the model after the calls already queued. */
    fun close(callback: (Result<Unit>) -> Unit) {
        if (closed) {
            callback(Result.success(Unit))
            return
        }
        closed = true
        try {
            executor.execute {
                release()
                mainHandler.post { callback(Result.success(Unit)) }
            }
        } catch (_: RejectedExecutionException) {
            callback(Result.success(Unit))
        }
        executor.shutdown()
    }

    private fun <T> submit(
        callback: (Result<T>) -> Unit,
        errorCode: String,
        work: () -> T,
    ) {
        if (closed) {
            callback(Result.failure(VisionException(VisionException.DISPOSED, "The model was disposed.").toFlutterError()))
            return
        }
        try {
            executor.execute {
                val result = try {
                    Result.success(work())
                } catch (t: Throwable) {
                    Log.w(TAG, "Model $handle failed", t)
                    Result.failure(VisionException.from(t, errorCode))
                }
                mainHandler.post { callback(result) }
            }
        } catch (_: RejectedExecutionException) {
            callback(Result.failure(VisionException(VisionException.DISPOSED, "The model was disposed.").toFlutterError()))
        }
    }

    private fun loadOnThread(request: LoadModelRequest): ModelInfoMessage {
        val source = resolveModel(request)
        info = TfliteModelInfo.parse(source.bytes)

        val input = info.inputs.singleOrNull()
            ?: throw VisionException(VisionException.UNSUPPORTED_MODEL, "Expected one image input, the model has ${info.inputs.size}.")
        val inputType = elementType(input)
        if (inputType == ElementTypeMessage.UNSUPPORTED) {
            throw VisionException(VisionException.UNSUPPORTED_MODEL, "The model input must be float32 or int8 (got TFLite type ${input.type}).")
        }
        val shape = input.shape
        val channelsFirst = shape.size == 4 && shape[1] == 3 && shape[3] != 3
        if (shape.size != 4 || (!channelsFirst && shape[3] != 3)) {
            throw VisionException(VisionException.UNSUPPORTED_MODEL, "Expected an image input [1, H, W, 3] or [1, 3, H, W], got ${shape.contentToString()}.")
        }
        val inputHeight = if (channelsFirst) shape[2] else shape[1]
        val inputWidth = if (channelsFirst) shape[3] else shape[2]
        for (output in info.outputs) {
            if (elementType(output) == ElementTypeMessage.UNSUPPORTED) {
                throw VisionException(
                    VisionException.UNSUPPORTED_MODEL,
                    "Output '${output.name}' must be float32 or int8 (got TFLite type ${output.type}). " +
                        "Export the model with float outputs.",
                )
            }
        }

        preprocessor = Preprocessor(inputWidth, inputHeight, channelsFirst, inputType == ElementTypeMessage.INT8)
        val env = Environment.create(context.applicationContext).also { environment = it }

        // Try the preferred accelerators first; every attempt keeps the CPU as
        // fallback for the operations the accelerator cannot run.
        val preference = request.accelerators.ifEmpty { listOf(AcceleratorMessage.CPU) }
        var lastError: Throwable? = null
        for (i in preference.indices) {
            val accelerators = preference.subList(i, preference.size).map(::toLiteRt).toMutableSet()
            accelerators.add(Accelerator.CPU)
            try {
                val options = CompiledModel.Options(accelerators).apply {
                    cpuOptions = CompiledModel.CpuOptions(request.numThreads.toInt().coerceAtLeast(1), null, null)
                }
                val compiled = if (source.assetPath != null) {
                    CompiledModel.create(context.assets, source.assetPath, options, env)
                } else {
                    CompiledModel.create(source.file!!.absolutePath, options, env)
                }
                model = compiled
                createBuffers(compiled)
                warmUp()
                return ModelInfoMessage(
                    handle = handle,
                    input = TensorInfoMessage(input.shape.map { it.toLong() }, inputType),
                    outputs = info.outputs.map { TensorInfoMessage(it.shape.map { d -> d.toLong() }, elementType(it)) },
                    accelerator = preference[i],
                )
            } catch (t: Throwable) {
                Log.w(TAG, "Cannot run model on $accelerators, trying the next option", t)
                lastError = t
                releaseModel()
            }
        }
        throw VisionException(
            VisionException.LOAD_FAILED,
            "LiteRT could not load the model: ${lastError?.message ?: "unknown error"}",
            lastError,
        )
    }

    private fun createBuffers(compiled: CompiledModel) {
        val signature = info.signatureKey
        inputBuffers = info.inputs.associate { it.name to compiled.createInputBuffer(it.name, signature) }
        outputBuffers = LinkedHashMap<String, TensorBuffer>().apply {
            for (output in info.outputs) put(output.name, compiled.createOutputBuffer(output.name, signature))
        }
    }

    /** One run with a blank input: surfaces accelerator failures at load time. */
    private fun warmUp() {
        writeInput()
        model!!.run(inputBuffers, outputBuffers, info.signatureKey)
    }

    private fun writeInput() {
        val buffer = inputBuffers.values.single()
        val prep = preprocessor!!
        prep.floatInput?.let { buffer.writeFloat(it) }
        prep.int8Input?.let { buffer.writeInt8(it) }
    }

    private fun infer(prepare: () -> Letterbox): InferenceResultMessage {
        val compiled = model ?: throw VisionException(VisionException.DISPOSED, "The model was disposed.")
        val start = SystemClock.elapsedRealtimeNanos()
        val letterbox = prepare()
        writeInput()
        val prepared = SystemClock.elapsedRealtimeNanos()
        compiled.run(inputBuffers, outputBuffers, info.signatureKey)
        val outputs = info.outputs.map { desc -> readOutput(desc, outputBuffers.getValue(desc.name)) }
        val done = SystemClock.elapsedRealtimeNanos()
        return InferenceResultMessage(
            outputs = outputs,
            imageWidth = letterbox.imageWidth.toLong(),
            imageHeight = letterbox.imageHeight.toLong(),
            scale = letterbox.scale.toDouble(),
            padX = letterbox.padX.toDouble(),
            padY = letterbox.padY.toDouble(),
            preprocessMicros = (prepared - start) / 1000,
            inferenceMicros = (done - prepared) / 1000,
        )
    }

    /** Output as little-endian float32 bytes; int8 outputs are dequantized. */
    private fun readOutput(desc: TfliteModelInfo.TensorDesc, buffer: TensorBuffer): ByteArray {
        val floats: FloatArray = if (desc.type == TfliteModelInfo.INT8) {
            val raw = buffer.readInt8()
            FloatArray(raw.size) { (raw[it] - desc.zeroPoint) * desc.scale }
        } else {
            buffer.readFloat()
        }
        val bytes = ByteBuffer.allocate(floats.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bytes.asFloatBuffer().put(floats)
        return bytes.array()
    }

    /** Where LiteRT reads the model from, plus a view of its bytes for the header. */
    private class ModelSource(val assetPath: String?, val file: File?, val bytes: ByteBuffer)

    /**
     * Uncompressed assets are read in place from the APK. Compressed assets
     * (when the app did not mark .tflite as noCompress) are copied once to the
     * cache directory.
     */
    private fun resolveModel(request: LoadModelRequest): ModelSource {
        if (!request.isAsset) {
            val file = File(request.modelPath)
            if (!file.isFile) {
                throw VisionException(VisionException.NOT_FOUND, "Model file not found: ${request.modelPath}")
            }
            return ModelSource(null, file, map(file))
        }
        val assetPath = assets.getAssetFilePathByName(request.modelPath)
        try {
            context.assets.openFd(assetPath).use { fd ->
                FileInputStream(fd.fileDescriptor).use { stream ->
                    val bytes = stream.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                    return ModelSource(assetPath, null, bytes)
                }
            }
        } catch (_: FileNotFoundException) {
            // Compressed asset, or missing: handled below.
        }
        val stream = try {
            context.assets.open(assetPath)
        } catch (e: FileNotFoundException) {
            throw VisionException(
                VisionException.NOT_FOUND,
                "Asset '${request.modelPath}' not found. Check that it is listed under flutter/assets in pubspec.yaml.",
                e,
            )
        }
        val cached = File(File(context.cacheDir, "flutter_vision"), assetPath.replace('/', '_'))
        cached.parentFile?.mkdirs()
        val temp = File(cached.parentFile, cached.name + ".tmp")
        stream.use { input -> temp.outputStream().use { input.copyTo(it, 1 shl 16) } }
        if (!temp.renameTo(cached)) {
            temp.copyTo(cached, overwrite = true)
            temp.delete()
        }
        return ModelSource(null, cached, map(cached))
    }

    private fun map(file: File): ByteBuffer =
        RandomAccessFile(file, "r").use { it.channel.map(FileChannel.MapMode.READ_ONLY, 0, it.length()) }

    private fun releaseModel() {
        for (buffer in inputBuffers.values + outputBuffers.values) runCatching { buffer.close() }
        inputBuffers = emptyMap()
        outputBuffers = emptyMap()
        model?.let { runCatching { it.close() } }
        model = null
    }

    private fun release() {
        releaseModel()
        environment?.let { runCatching { it.close() } }
        environment = null
        preprocessor?.release()
        preprocessor = null
    }

    private fun elementType(desc: TfliteModelInfo.TensorDesc): ElementTypeMessage = when (desc.type) {
        TfliteModelInfo.FLOAT32 -> ElementTypeMessage.FLOAT32
        TfliteModelInfo.INT8 -> if (desc.scale > 0f) ElementTypeMessage.INT8 else ElementTypeMessage.UNSUPPORTED
        else -> ElementTypeMessage.UNSUPPORTED
    }

    private fun toLiteRt(accelerator: AcceleratorMessage): Accelerator = when (accelerator) {
        AcceleratorMessage.CPU -> Accelerator.CPU
        AcceleratorMessage.GPU -> Accelerator.GPU
        AcceleratorMessage.NPU -> Accelerator.NPU
    }

    companion object {
        private const val TAG = "FlutterVision"
    }
}
