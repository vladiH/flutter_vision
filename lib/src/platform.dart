import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'exceptions.dart';
import 'messages.g.dart';

/// Bridge to the native engine. Replaceable in tests.
abstract class VisionPlatform {
  static VisionPlatform _instance = _PigeonVisionPlatform();

  static VisionPlatform get instance => _instance;

  @visibleForTesting
  static set instance(VisionPlatform platform) => _instance = platform;

  Future<ModelInfoMessage> loadModel(LoadModelRequest request);
  Future<InferenceResultMessage> runOnImage(int handle, Uint8List bytes);
  Future<InferenceResultMessage> runOnFrame(int handle, FrameMessage frame);
  Future<void> disposeModel(int handle);
}

class _PigeonVisionPlatform implements VisionPlatform {
  final FlutterVisionHostApi _api = FlutterVisionHostApi();

  Future<T> _guard<T>(Future<T> Function() call) async {
    try {
      return await call();
    } on PlatformException catch (e) {
      throw FlutterVisionException.fromPlatform(e);
    }
  }

  @override
  Future<ModelInfoMessage> loadModel(LoadModelRequest request) =>
      _guard(() => _api.loadModel(request));

  @override
  Future<InferenceResultMessage> runOnImage(int handle, Uint8List bytes) =>
      _guard(() => _api.runOnImage(handle, bytes));

  @override
  Future<InferenceResultMessage> runOnFrame(int handle, FrameMessage frame) =>
      _guard(() => _api.runOnFrame(handle, frame));

  @override
  Future<void> disposeModel(int handle) =>
      _guard(() => _api.disposeModel(handle));
}
