import 'dart:io';

import 'package:flutter/services.dart';

import 'camera_frame.dart';
import 'exceptions.dart';
import 'messages.g.dart';
import 'platform.dart';
import 'postprocess/model_layout.dart';
import 'postprocess/postprocessor.dart';
import 'types.dart';

/// A loaded YOLO model.
///
/// ```dart
/// final model = await YoloModel.load(
///   modelPath: 'assets/yolo11n.tflite',
///   labelsPath: 'assets/labels.txt',
/// );
/// final result = await model.detectImage(bytes);
/// for (final d in result.detections) {
///   print('${d.label} ${d.confidence} ${d.box}');
/// }
/// await model.dispose();
/// ```
///
/// Several models can be loaded at the same time. Each one runs on its own
/// native thread; calls to the same model run one after another.
class YoloModel {
  YoloModel._({
    required int handle,
    required this.labels,
    required this.layout,
    required this.inputLayout,
    required this.accelerator,
    required List<List<int>> outputShapes,
    required Postprocessor postprocessor,
  })  : _handle = handle,
        _outputShapes = outputShapes,
        _postprocessor = postprocessor;

  final int _handle;
  final List<List<int>> _outputShapes;
  final Postprocessor _postprocessor;
  bool _disposed = false;
  bool _frameInFlight = false;

  /// Class names, one per class index.
  final List<String> labels;

  /// How the model outputs are read.
  final ModelLayout layout;
  final InputLayout inputLayout;

  /// Hardware the model is running on (after any fallback).
  final Accelerator accelerator;

  YoloTask get task => layout.task;
  YoloFormat get format => layout.format;

  /// Size of the model input, for example 640x640.
  Size get inputSize =>
      Size(inputLayout.width.toDouble(), inputLayout.height.toDouble());

  /// True while a [detectFrame] call is running.
  bool get isBusy => _frameInFlight;

  bool get isDisposed => _disposed;

  /// Loads a `.tflite` YOLO model.
  ///
  /// - [labelsPath]: text file with one class name per line. Use [labels] to
  ///   pass the names directly instead.
  /// - [source]: whether the paths are Flutter assets or files on the device.
  /// - [task] / [format]: detected from the model; set them only to check the
  ///   model is what you expect or when detection fails.
  /// - [accelerator]: preferred hardware. Falls back to CPU when the model
  ///   cannot run on it; read [YoloModel.accelerator] to see the result.
  /// - [numThreads]: CPU threads used for inference.
  static Future<YoloModel> load({
    required String modelPath,
    String? labelsPath,
    List<String>? labels,
    ModelSource source = ModelSource.asset,
    YoloTask? task,
    YoloFormat? format,
    Accelerator accelerator = Accelerator.cpu,
    int numThreads = 4,
  }) async {
    if (labelsPath != null && labels != null) {
      throw ArgumentError('Pass either labelsPath or labels, not both.');
    }
    final names = labels ??
        (labelsPath == null
            ? const <String>[]
            : parseLabels(await _readText(labelsPath, source)));

    final info = await VisionPlatform.instance.loadModel(LoadModelRequest(
      modelPath: modelPath,
      isAsset: source == ModelSource.asset,
      accelerators: _preference(accelerator),
      numThreads: numThreads,
    ));

    try {
      if (info.input.elementType == ElementTypeMessage.unsupported) {
        throw FlutterVisionException(
          FlutterVisionException.unsupportedModel,
          'The model input must be float32 or int8.',
        );
      }
      for (final output in info.outputs) {
        if (output.elementType != ElementTypeMessage.float32) {
          throw FlutterVisionException(
            FlutterVisionException.unsupportedModel,
            'The model outputs must be float32. Export it with float '
            'outputs, for example yolo11n_float16.tflite, yolo11n_float32.tflite '
            'or yolo11n_int8.tflite from Ultralytics.',
          );
        }
      }
      final inputLayout = InputLayout.fromShape(info.input.shape);
      final outputShapes = [for (final o in info.outputs) o.shape];
      final layout = ModelLayout.resolve(
        outputShapes: outputShapes,
        labelCount: names.length,
        format: format,
        task: task,
      );
      return YoloModel._(
        handle: info.handle,
        labels: List.unmodifiable(names),
        layout: layout,
        inputLayout: inputLayout,
        accelerator: Accelerator.values[info.accelerator.index],
        outputShapes: outputShapes,
        postprocessor: Postprocessor(
          layout: layout,
          input: inputLayout,
          labels: names,
        ),
      );
    } catch (_) {
      await VisionPlatform.instance.disposeModel(info.handle);
      rethrow;
    }
  }

  /// Detects objects in an encoded image (JPEG, PNG, WebP, HEIC...).
  ///
  /// The EXIF orientation is applied first, so coordinates match what
  /// `Image.memory` / `Image.file` display.
  Future<YoloResult> detectImage(
    Uint8List encodedImage, {
    DetectionOptions options = const DetectionOptions(),
  }) async {
    _checkNotDisposed();
    final raw = await VisionPlatform.instance.runOnImage(_handle, encodedImage);
    return _postprocess(raw, options);
  }

  /// Detects objects in a camera frame.
  ///
  /// When [dropIfBusy] is true (the default) and the previous frame is still
  /// being processed, returns `null` immediately without sending the frame.
  /// This keeps live detection real time instead of queuing frames.
  Future<YoloResult?> detectFrame(
    CameraFrame frame, {
    DetectionOptions options = const DetectionOptions(),
    bool dropIfBusy = true,
  }) async {
    _checkNotDisposed();
    if (_frameInFlight && dropIfBusy) return null;
    _frameInFlight = true;
    try {
      final raw = await VisionPlatform.instance.runOnFrame(
        _handle,
        FrameMessage(
          width: frame.width,
          height: frame.height,
          format: FrameFormatMessage.values[frame.format.index],
          planes: [
            for (final p in frame.planes)
              PlaneMessage(
                bytes: p.bytes,
                bytesPerRow: p.bytesPerRow,
                bytesPerPixel: p.bytesPerPixel,
              ),
          ],
          rotationDegrees: frame.rotation,
        ),
      );
      if (_disposed) return null;
      return _postprocess(raw, options);
    } finally {
      _frameInFlight = false;
    }
  }

  /// Releases the native model. Pending calls finish first. The model cannot
  /// be used afterwards.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    await VisionPlatform.instance.disposeModel(_handle);
  }

  YoloResult _postprocess(InferenceResultMessage raw, DetectionOptions options) {
    final watch = Stopwatch()..start();
    final outputs = <OutputTensor>[
      for (var i = 0; i < raw.outputs.length; i++)
        OutputTensor.fromBytes(_outputShapes[i], raw.outputs[i]),
    ];
    final detections = _postprocessor.process(
      outputs,
      Letterbox(
        imageWidth: raw.imageWidth,
        imageHeight: raw.imageHeight,
        scale: raw.scale,
        padX: raw.padX,
        padY: raw.padY,
      ),
      options,
    );
    watch.stop();
    return YoloResult(
      detections: detections,
      imageSize: Size(raw.imageWidth.toDouble(), raw.imageHeight.toDouble()),
      preprocessTime: Duration(microseconds: raw.preprocessMicros),
      inferenceTime: Duration(microseconds: raw.inferenceMicros),
      postprocessTime: watch.elapsed,
    );
  }

  void _checkNotDisposed() {
    if (_disposed) {
      throw const FlutterVisionException(
        FlutterVisionException.disposed,
        'The model was disposed.',
      );
    }
  }

  static List<AcceleratorMessage> _preference(Accelerator accelerator) {
    switch (accelerator) {
      case Accelerator.npu:
        return [
          AcceleratorMessage.npu,
          AcceleratorMessage.gpu,
          AcceleratorMessage.cpu,
        ];
      case Accelerator.gpu:
        return [AcceleratorMessage.gpu, AcceleratorMessage.cpu];
      case Accelerator.cpu:
        return [AcceleratorMessage.cpu];
    }
  }

  static Future<String> _readText(String path, ModelSource source) async {
    try {
      return source == ModelSource.asset
          ? await rootBundle.loadString(path)
          : await File(path).readAsString();
    } catch (e) {
      throw FlutterVisionException(
        FlutterVisionException.notFound,
        'Cannot read labels from "$path" ($e). For assets, check that the path '
        'matches the entry in pubspec.yaml.',
      );
    }
  }

  /// One label per non-empty line. A `0: person` style prefix (as in
  /// Ultralytics `data.yaml`) is removed.
  static List<String> parseLabels(String text) {
    final prefix = RegExp(r'^\d+\s*:\s*');
    return [
      for (final line in text.split(RegExp(r'\r?\n')))
        if (line.trim().isNotEmpty) line.trim().replaceFirst(prefix, ''),
    ];
  }
}
