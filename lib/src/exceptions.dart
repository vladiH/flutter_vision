import 'package:flutter/services.dart';

/// Error raised by flutter_vision.
class FlutterVisionException implements Exception {
  const FlutterVisionException(this.code, this.message);

  /// Builds an exception from an error reported by the native side.
  factory FlutterVisionException.fromPlatform(PlatformException error) {
    return FlutterVisionException(
      error.code,
      error.message ?? 'Unknown native error',
    );
  }

  /// Machine readable reason. One of the constants in this class or a code
  /// reported by the native engine.
  final String code;
  final String message;

  /// The model or labels file does not exist.
  static const String notFound = 'not_found';

  /// The model cannot be used (unknown output layout, unsupported tensor type,
  /// labels that do not match the model...).
  static const String unsupportedModel = 'unsupported_model';

  /// LiteRT failed to load or compile the model.
  static const String loadFailed = 'load_failed';

  /// The image or camera frame could not be read.
  static const String invalidInput = 'invalid_input';

  /// Inference failed.
  static const String inferenceFailed = 'inference_failed';

  /// The model was used after [YoloModel.dispose].
  static const String disposed = 'disposed';

  @override
  String toString() => 'FlutterVisionException($code): $message';
}
