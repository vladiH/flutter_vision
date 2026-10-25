package com.vladih.computer_vision.flutter_vision

/** Error with a code understood by the Dart side (see FlutterVisionException). */
internal class VisionException(
    val code: String,
    override val message: String,
    cause: Throwable? = null,
) : Exception(message, cause) {
    fun toFlutterError(): FlutterError = FlutterError(code, message, cause?.toString())

    companion object {
        const val NOT_FOUND = "not_found"
        const val UNSUPPORTED_MODEL = "unsupported_model"
        const val LOAD_FAILED = "load_failed"
        const val INVALID_INPUT = "invalid_input"
        const val INFERENCE_FAILED = "inference_failed"
        const val DISPOSED = "disposed"

        fun from(error: Throwable, fallbackCode: String): FlutterError = when (error) {
            is VisionException -> error.toFlutterError()
            is FlutterError -> error
            is OutOfMemoryError -> FlutterError(fallbackCode, "Out of memory: ${error.message}", null)
            else -> FlutterError(fallbackCode, error.message ?: error.toString(), error.toString())
        }
    }
}
