import 'dart:ui';

import 'package:flutter/foundation.dart';

/// What a model predicts.
enum YoloTask {
  /// Bounding boxes only.
  detect,

  /// Bounding boxes plus an outline (polygon) for every object.
  segment,
}

/// Output layout of a YOLO model.
///
/// It is detected automatically from the model's output shape and the number
/// of labels. Set it explicitly only when detection fails.
enum YoloFormat {
  /// YOLOv5 / YOLOv7 style output with an objectness score:
  /// `[1, N, 5 + classes]`.
  anchorBased,

  /// YOLOv8, v9, v11 and v12 style output without objectness:
  /// `[1, 4 + classes, N]` (or transposed).
  anchorFree,

  /// NMS-free models such as YOLOv10 and YOLO26:
  /// `[1, N, 6]` = x1, y1, x2, y2, score, class.
  endToEnd,
}

/// Hardware used to run the model.
enum Accelerator {
  /// Always available. Uses XNNPACK.
  cpu,

  /// OpenCL/OpenGL on Android, Metal on iOS. Falls back to CPU when the model
  /// cannot run on the GPU.
  gpu,

  /// Android NPU (Qualcomm, MediaTek, Google Tensor) when the device has one.
  /// Falls back to GPU, then CPU.
  npu,
}

/// Where the model and labels files live.
enum ModelSource {
  /// A Flutter asset declared in `pubspec.yaml`, for example
  /// `assets/yolo11n.tflite`.
  asset,

  /// An absolute path on the device, for example a downloaded model.
  file,
}

/// Settings for one detection call.
@immutable
class DetectionOptions {
  const DetectionOptions({
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.45,
    this.maxDetections = 300,
    this.classAgnosticNms = false,
    this.classes,
    this.maskThreshold = 0.5,
  })  : assert(confidenceThreshold >= 0 && confidenceThreshold <= 1),
        assert(iouThreshold >= 0 && iouThreshold <= 1),
        assert(maxDetections > 0),
        assert(maskThreshold > 0 && maskThreshold < 1);

  /// Minimum score (0-1) a detection needs to be returned.
  final double confidenceThreshold;

  /// Boxes that overlap more than this (intersection over union) with a
  /// higher scoring box are discarded. Not used by [YoloFormat.endToEnd].
  final double iouThreshold;

  /// Maximum number of detections returned, highest scores first.
  final int maxDetections;

  /// When true, overlapping boxes of different classes also suppress each
  /// other. By default only boxes of the same class do.
  final bool classAgnosticNms;

  /// Only return these class indices. `null` returns every class.
  final Set<int>? classes;

  /// Probability above which a pixel belongs to the object (segmentation).
  final double maskThreshold;
}

/// One detected object.
@immutable
class Detection {
  const Detection({
    required this.box,
    required this.classIndex,
    required this.label,
    required this.confidence,
    this.polygon,
  });

  /// Bounding box in pixels of the upright image (EXIF orientation and frame
  /// rotation already applied). Use [BoxMapper] to draw it on screen.
  final Rect box;

  final int classIndex;
  final String label;

  /// Score between 0 and 1.
  final double confidence;

  /// Outline of the object in pixels of the upright image. Only set for
  /// segmentation models. Empty when the mask is too small to trace.
  final List<Offset>? polygon;

  @override
  String toString() =>
      'Detection($label ${(confidence * 100).toStringAsFixed(1)}% $box)';
}

/// Result of one detection call.
@immutable
class YoloResult {
  const YoloResult({
    required this.detections,
    required this.imageSize,
    required this.preprocessTime,
    required this.inferenceTime,
    required this.postprocessTime,
  });

  /// Detections sorted by confidence, highest first.
  final List<Detection> detections;

  /// Size of the upright image the coordinates refer to.
  final Size imageSize;

  final Duration preprocessTime;
  final Duration inferenceTime;
  final Duration postprocessTime;

  Duration get totalTime => preprocessTime + inferenceTime + postprocessTime;
}
