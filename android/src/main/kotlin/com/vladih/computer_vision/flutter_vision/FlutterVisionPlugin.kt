package com.vladih.computer_vision.flutter_vision

import io.flutter.embedding.engine.plugins.FlutterPlugin
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

/** Entry point. Keeps one [ModelSession] per loaded model. */
class FlutterVisionPlugin : FlutterPlugin, FlutterVisionHostApi {
    private var binding: FlutterPlugin.FlutterPluginBinding? = null
    private val sessions = ConcurrentHashMap<Long, ModelSession>()
    private val nextHandle = AtomicLong(1)

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        this.binding = binding
        FlutterVisionHostApi.setUp(binding.binaryMessenger, this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        FlutterVisionHostApi.setUp(binding.binaryMessenger, null)
        for (session in sessions.values) session.close {}
        sessions.clear()
        this.binding = null
    }

    override fun loadModel(request: LoadModelRequest, callback: (Result<ModelInfoMessage>) -> Unit) {
        val binding = this.binding ?: run {
            callback(Result.failure(FlutterError(VisionException.LOAD_FAILED, "Plugin not attached to an engine.", null)))
            return
        }
        val handle = nextHandle.getAndIncrement()
        val session = ModelSession(handle, binding.applicationContext, binding.flutterAssets)
        sessions[handle] = session
        session.load(request) { result ->
            if (result.isFailure) {
                sessions.remove(handle)
                session.close {}
            }
            callback(result)
        }
    }

    override fun runOnImage(handle: Long, encodedImage: ByteArray, callback: (Result<InferenceResultMessage>) -> Unit) {
        val session = sessions[handle] ?: return callback(disposed())
        session.runImage(encodedImage, callback)
    }

    override fun runOnFrame(handle: Long, frame: FrameMessage, callback: (Result<InferenceResultMessage>) -> Unit) {
        val session = sessions[handle] ?: return callback(disposed())
        session.runFrame(frame, callback)
    }

    override fun disposeModel(handle: Long, callback: (Result<Unit>) -> Unit) {
        val session = sessions.remove(handle) ?: return callback(Result.success(Unit))
        session.close(callback)
    }

    private fun <T> disposed(): Result<T> =
        Result.failure(FlutterError(VisionException.DISPOSED, "The model was disposed or never loaded.", null))
}
