// Contract between the Dart API and the native (Kotlin / Swift) engines.
//
// Regenerate the bindings after editing this file:
//   dart run pigeon --input pigeons/messages.dart
//
// The native side only loads models, prepares the input tensor (decode,
// rotate, letterbox, normalize) and runs inference. All YOLO decoding, NMS,
// masks and coordinate mapping happen in Dart so both platforms return
// identical results.
import 'package:pigeon/pigeon.dart';

@ConfigurePigeon(
  PigeonOptions(
    dartOut: 'lib/src/messages.g.dart',
    dartPackageName: 'flutter_vision',
    kotlinOut:
        'android/src/main/kotlin/com/vladih/computer_vision/flutter_vision/Messages.g.kt',
    kotlinOptions: KotlinOptions(
      package: 'com.vladih.computer_vision.flutter_vision',
    ),
    swiftOut: 'ios/flutter_vision/Sources/flutter_vision/Messages.g.swift',
    copyrightHeader: 'pigeons/copyright.txt',
  ),
)
/// Hardware used to run a model.
enum AcceleratorMessage { cpu, gpu, npu }

/// Element type of a model tensor, as reported by LiteRT.
enum ElementTypeMessage { float32, int8, unsupported }

/// Pixel layout of a camera frame.
enum FrameFormatMessage { nv21, yuv420, bgra8888, rgba8888 }

class LoadModelRequest {
  LoadModelRequest({
    required this.modelPath,
    required this.isAsset,
    required this.accelerators,
    required this.numThreads,
  });

  /// Flutter asset key (for example `assets/yolo11n.tflite`) or absolute path.
  String modelPath;
  bool isAsset;

  /// Accelerators to try, in order of preference. CPU is always the fallback.
  List<AcceleratorMessage> accelerators;
  int numThreads;
}

class TensorInfoMessage {
  TensorInfoMessage({required this.shape, required this.elementType});

  List<int> shape;
  ElementTypeMessage elementType;
}

class ModelInfoMessage {
  ModelInfoMessage({
    required this.handle,
    required this.input,
    required this.outputs,
    required this.accelerator,
  });

  /// Identifier of the loaded model on the native side.
  int handle;
  TensorInfoMessage input;
  List<TensorInfoMessage> outputs;

  /// Accelerator the model was actually compiled for.
  AcceleratorMessage accelerator;
}

class PlaneMessage {
  PlaneMessage({
    required this.bytes,
    required this.bytesPerRow,
    required this.bytesPerPixel,
  });

  Uint8List bytes;
  int bytesPerRow;

  /// Distance in bytes between two consecutive pixels of this plane.
  int bytesPerPixel;
}

class FrameMessage {
  FrameMessage({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
    required this.rotationDegrees,
  });

  /// Size of the frame as delivered by the camera, before rotation.
  int width;
  int height;
  FrameFormatMessage format;
  List<PlaneMessage> planes;

  /// Clockwise rotation (0, 90, 180 or 270) that makes the frame upright.
  int rotationDegrees;
}

class InferenceResultMessage {
  InferenceResultMessage({
    required this.outputs,
    required this.imageWidth,
    required this.imageHeight,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.preprocessMicros,
    required this.inferenceMicros,
  });

  /// One entry per model output: float32 values, little endian, in the
  /// tensor's own element order.
  List<Uint8List> outputs;

  /// Size of the upright image the input tensor was built from.
  int imageWidth;
  int imageHeight;

  /// Letterbox transform: inputPixel = imagePixel * scale + pad.
  double scale;
  double padX;
  double padY;

  int preprocessMicros;
  int inferenceMicros;
}

@HostApi()
abstract class FlutterVisionHostApi {
  @asyncCallback
  ModelInfoMessage loadModel(LoadModelRequest request);

  /// Runs the model on an encoded image (JPEG, PNG, WebP, HEIC...).
  /// EXIF orientation is applied before inference.
  @asyncCallback
  InferenceResultMessage runOnImage(int handle, Uint8List encodedImage);

  @asyncCallback
  InferenceResultMessage runOnFrame(int handle, FrameMessage frame);

  @asyncCallback
  void disposeModel(int handle);
}
