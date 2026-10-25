import 'dart:typed_data';
import 'dart:ui';

import '../exceptions.dart';
import '../types.dart';
import 'decoder.dart';
import 'mask.dart';
import 'model_layout.dart';

/// Letterbox transform used to build the input tensor:
/// `inputPixel = imagePixel * scale + pad`.
class Letterbox {
  const Letterbox({
    required this.imageWidth,
    required this.imageHeight,
    required this.scale,
    required this.padX,
    required this.padY,
  });

  final int imageWidth;
  final int imageHeight;
  final double scale;
  final double padX;
  final double padY;

  double toImageX(double x) =>
      ((x - padX) / scale).clamp(0.0, imageWidth.toDouble());
  double toImageY(double y) =>
      ((y - padY) / scale).clamp(0.0, imageHeight.toDouble());
}

/// Raw output tensor: shape plus float values.
class OutputTensor {
  const OutputTensor(this.shape, this.data);

  /// Wraps little-endian float32 bytes, copying only when they are not
  /// 4-byte aligned.
  factory OutputTensor.fromBytes(List<int> shape, Uint8List bytes) {
    final length = bytes.lengthInBytes ~/ 4;
    final Float32List data = bytes.offsetInBytes % 4 == 0
        ? bytes.buffer.asFloat32List(bytes.offsetInBytes, length)
        : Uint8List.fromList(bytes).buffer.asFloat32List(0, length);
    var expected = 1;
    for (final d in shape) {
      expected *= d;
    }
    if (expected != length) {
      throw FlutterVisionException(
        FlutterVisionException.inferenceFailed,
        'Output of shape $shape should have $expected values, got $length.',
      );
    }
    return OutputTensor(shape, data);
  }

  final List<int> shape;
  final Float32List data;
}

/// Turns raw model outputs into [Detection]s in image coordinates.
class Postprocessor {
  Postprocessor({
    required this.layout,
    required this.input,
    required this.labels,
  })  : decoder = Decoder(layout,
            inputWidth: input.width, inputHeight: input.height),
        tracer = layout.task == YoloTask.segment
            ? MaskTracer(layout,
                inputWidth: input.width, inputHeight: input.height)
            : null;

  final ModelLayout layout;
  final InputLayout input;
  final List<String> labels;
  final Decoder decoder;
  final MaskTracer? tracer;

  List<Detection> process(
    List<OutputTensor> outputs,
    Letterbox letterbox,
    DetectionOptions options,
  ) {
    final predictions = outputs[layout.predictionsIndex].data;
    final candidates = decoder.decode(predictions, options);
    final kept = layout.format == YoloFormat.endToEnd
        ? candidates.take(options.maxDetections).toList()
        : nonMaxSuppression(
            candidates,
            iouThreshold: options.iouThreshold,
            classAgnostic: options.classAgnosticNms,
            maxDetections: options.maxDetections,
          );

    final protos =
        layout.protosIndex == null ? null : outputs[layout.protosIndex!].data;
    final detections = <Detection>[];
    for (final c in kept) {
      final box = Rect.fromLTRB(
        letterbox.toImageX(c.x1),
        letterbox.toImageY(c.y1),
        letterbox.toImageX(c.x2),
        letterbox.toImageY(c.y2),
      );
      if (box.width <= 0 || box.height <= 0) continue;

      List<Offset>? polygon;
      if (protos != null && tracer != null) {
        final coefficients = decoder.maskCoefficients(predictions, c);
        polygon = [
          for (final p
              in tracer!.outline(protos, coefficients, c, options.maskThreshold))
            Offset(letterbox.toImageX(p.dx), letterbox.toImageY(p.dy)),
        ];
      }
      detections.add(Detection(
        box: box,
        classIndex: c.classIndex,
        label: c.classIndex < labels.length
            ? labels[c.classIndex]
            : 'class_${c.classIndex}',
        confidence: c.score.clamp(0.0, 1.0),
        polygon: polygon,
      ));
    }
    return detections;
  }
}
