import 'dart:async';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:flutter_vision/src/messages.g.dart';
import 'package:flutter_vision/src/platform.dart';

class FakePlatform implements VisionPlatform {
  FakePlatform({
    this.outputType = ElementTypeMessage.float32,
    this.outputShapes = const [
      [1, 6, 4]
    ],
  });

  final ElementTypeMessage outputType;
  final List<List<int>> outputShapes;
  final List<int> disposed = [];
  LoadModelRequest? lastRequest;
  FrameMessage? lastFrame;
  Completer<void>? gate;
  int frames = 0;

  @override
  Future<ModelInfoMessage> loadModel(LoadModelRequest request) async {
    lastRequest = request;
    return ModelInfoMessage(
      handle: 7,
      input: TensorInfoMessage(
          shape: [1, 640, 640, 3], elementType: ElementTypeMessage.float32),
      outputs: [
        for (final s in outputShapes)
          TensorInfoMessage(shape: s, elementType: outputType),
      ],
      accelerator: AcceleratorMessage.gpu,
    );
  }

  InferenceResultMessage _result() {
    // One box of class 1 at anchor 2: centre (0.5, 0.5), size 0.25.
    final data = Float32List(6 * 4);
    data[0 * 4 + 2] = 0.5;
    data[1 * 4 + 2] = 0.5;
    data[2 * 4 + 2] = 0.25;
    data[3 * 4 + 2] = 0.25;
    data[5 * 4 + 2] = 0.8;
    return InferenceResultMessage(
      outputs: [data.buffer.asUint8List()],
      imageWidth: 1280,
      imageHeight: 1280,
      scale: 0.5,
      padX: 0,
      padY: 0,
      preprocessMicros: 1000,
      inferenceMicros: 2000,
    );
  }

  @override
  Future<InferenceResultMessage> runOnImage(int handle, Uint8List bytes) async {
    expect(handle, 7);
    return _result();
  }

  @override
  Future<InferenceResultMessage> runOnFrame(
      int handle, FrameMessage frame) async {
    lastFrame = frame;
    frames++;
    if (gate != null) await gate!.future;
    return _result();
  }

  @override
  Future<void> disposeModel(int handle) async => disposed.add(handle);
}

void main() {
  late FakePlatform platform;

  setUp(() {
    platform = FakePlatform();
    VisionPlatform.instance = platform;
  });

  test('loads, detects in image coordinates and disposes', () async {
    final model = await YoloModel.load(
      modelPath: 'assets/model.tflite',
      labels: const ['cat', 'dog'],
      accelerator: Accelerator.gpu,
    );
    expect(platform.lastRequest!.accelerators,
        [AcceleratorMessage.gpu, AcceleratorMessage.cpu]);
    expect(platform.lastRequest!.isAsset, isTrue);
    expect(model.accelerator, Accelerator.gpu);
    expect(model.format, YoloFormat.anchorFree);
    expect(model.inputSize, const Size(640, 640));

    final result = await model.detectImage(Uint8List(0));
    final d = result.detections.single;
    expect(d.label, 'dog');
    // 0.375 * 640 / 0.5 = 480 .. 0.625 * 640 / 0.5 = 800.
    expect(d.box, const Rect.fromLTRB(480, 480, 800, 800));
    expect(result.imageSize, const Size(1280, 1280));
    expect(result.inferenceTime, const Duration(milliseconds: 2));

    await model.dispose();
    expect(platform.disposed, [7]);
    expect(() => model.detectImage(Uint8List(0)),
        throwsA(isA<FlutterVisionException>()));
  });

  test('drops frames while the previous one is running', () async {
    final model =
        await YoloModel.load(modelPath: 'm', labels: const ['a', 'b']);
    platform.gate = Completer<void>();
    final frame = CameraFrame(
      width: 4,
      height: 2,
      format: FrameFormat.bgra8888,
      planes: [FramePlane(bytes: Uint8List(32), bytesPerRow: 16)],
      rotation: 90,
    );
    final first = model.detectFrame(frame);
    expect(model.isBusy, isTrue);
    expect(await model.detectFrame(frame), isNull);
    platform.gate!.complete();
    expect((await first)!.detections, hasLength(1));
    expect(platform.frames, 1);
    expect(platform.lastFrame!.rotationDegrees, 90);
    expect(platform.lastFrame!.format, FrameFormatMessage.bgra8888);
    expect(model.isBusy, isFalse);
  });

  test('rejects quantized outputs and frees the native model', () async {
    VisionPlatform.instance =
        platform = FakePlatform(outputType: ElementTypeMessage.int8);
    await expectLater(
      YoloModel.load(modelPath: 'm', labels: const ['a', 'b']),
      throwsA(isA<FlutterVisionException>()
          .having((e) => e.code, 'code', FlutterVisionException.unsupportedModel)),
    );
    expect(platform.disposed, [7]);
  });

  test('rejects labels that do not match and frees the native model',
      () async {
    await expectLater(
      YoloModel.load(modelPath: 'm', labels: const ['a', 'b', 'c']),
      throwsA(isA<FlutterVisionException>()),
    );
    expect(platform.disposed, [7]);
  });

  test('labelsPath and labels are exclusive', () {
    expect(
      () => YoloModel.load(modelPath: 'm', labelsPath: 'x', labels: const []),
      throwsArgumentError,
    );
  });
}
