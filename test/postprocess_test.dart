import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:flutter_vision/src/postprocess/decoder.dart';
import 'package:flutter_vision/src/postprocess/mask.dart';
import 'package:flutter_vision/src/postprocess/postprocessor.dart';

/// Builds a channels-first anchor-free tensor `[1, 4 + nc + nm, n]`.
Float32List anchorFreeTensor({
  required int n,
  required int nc,
  int nm = 0,
  required List<({int anchor, List<double> box, int cls, double score})>
      boxes,
  List<List<double>>? masks,
}) {
  final c = 4 + nc + nm;
  final data = Float32List(c * n);
  for (var b = 0; b < boxes.length; b++) {
    final box = boxes[b];
    for (var k = 0; k < 4; k++) {
      data[k * n + box.anchor] = box.box[k];
    }
    data[(4 + box.cls) * n + box.anchor] = box.score;
    if (masks != null) {
      for (var k = 0; k < nm; k++) {
        data[(4 + nc + k) * n + box.anchor] = masks[b][k];
      }
    }
  }
  return data;
}

const identity = Letterbox(
    imageWidth: 640, imageHeight: 640, scale: 1, padX: 0, padY: 0);
const input640 = InputLayout(width: 640, height: 640, channelsFirst: false);

void main() {
  group('anchor-free decoding', () {
    final layout = ModelLayout.resolve(outputShapes: [
      [1, 6, 100]
    ], labelCount: 2);

    test('scales normalized boxes, applies NMS per class and the letterbox',
        () {
      final data = anchorFreeTensor(n: 100, nc: 2, boxes: [
        // cx, cy, w, h normalized.
        (anchor: 3, box: [0.5, 0.5, 0.2, 0.2], cls: 0, score: 0.9),
        // Same place, lower score, same class: suppressed.
        (anchor: 7, box: [0.51, 0.5, 0.2, 0.2], cls: 0, score: 0.8),
        // Same place, other class: kept (class-aware NMS).
        (anchor: 9, box: [0.5, 0.5, 0.2, 0.2], cls: 1, score: 0.7),
        // Below threshold.
        (anchor: 11, box: [0.1, 0.1, 0.1, 0.1], cls: 1, score: 0.1),
      ]);
      final post = Postprocessor(
          layout: layout, input: input640, labels: const ['cat', 'dog']);
      // Image 1280x720 letterboxed into 640x640: scale 0.5, padY 140.
      final detections = post.process(
        [OutputTensor(const [1, 6, 100], data)],
        const Letterbox(
            imageWidth: 1280, imageHeight: 720, scale: 0.5, padX: 0, padY: 140),
        const DetectionOptions(),
      );
      expect(detections, hasLength(2));
      expect(detections[0].label, 'cat');
      expect(detections[0].confidence, closeTo(0.9, 1e-6));
      // x: (0.4 * 640) / 0.5 = 512, y: (0.4 * 640 - 140) / 0.5 = 232.
      expect(detections[0].box.left, closeTo(512, 0.01));
      expect(detections[0].box.top, closeTo(232, 0.01));
      expect(detections[0].box.right, closeTo(768, 0.01));
      expect(detections[0].box.bottom, closeTo(488, 0.01));
      expect(detections[1].label, 'dog');
    });

    test('class agnostic NMS and class filter', () {
      final data = anchorFreeTensor(n: 100, nc: 2, boxes: [
        (anchor: 3, box: [0.5, 0.5, 0.2, 0.2], cls: 0, score: 0.9),
        (anchor: 9, box: [0.5, 0.5, 0.2, 0.2], cls: 1, score: 0.7),
      ]);
      final post =
          Postprocessor(layout: layout, input: input640, labels: const []);
      final agnostic = post.process(
        [OutputTensor(const [1, 6, 100], data)],
        identity,
        const DetectionOptions(classAgnosticNms: true),
      );
      expect(agnostic, hasLength(1));
      expect(agnostic.single.label, 'class_0');

      final onlyDogs = post.process(
        [OutputTensor(const [1, 6, 100], data)],
        identity,
        const DetectionOptions(classes: {1}),
      );
      expect(onlyDogs.single.classIndex, 1);
    });

    test('keeps pixel coordinates when the model is not normalized', () {
      final data = anchorFreeTensor(n: 100, nc: 2, boxes: [
        (anchor: 0, box: [320, 320, 100, 50], cls: 1, score: 0.6),
      ]);
      final post =
          Postprocessor(layout: layout, input: input640, labels: const []);
      final d = post
          .process([OutputTensor(const [1, 6, 100], data)], identity,
              const DetectionOptions())
          .single;
      expect(d.box.left, closeTo(270, 1e-3));
      expect(d.box.top, closeTo(295, 1e-3));
      expect(d.box.width, closeTo(100, 1e-3));
    });

    test('maxDetections caps the result', () {
      final data = anchorFreeTensor(n: 100, nc: 2, boxes: [
        for (var i = 0; i < 10; i++)
          (
            anchor: i,
            box: [0.05 + i * 0.09, 0.5, 0.05, 0.05],
            cls: 0,
            score: 0.5 + i * 0.01
          ),
      ]);
      final post =
          Postprocessor(layout: layout, input: input640, labels: const []);
      final detections = post.process(
          [OutputTensor(const [1, 6, 100], data)],
          identity,
          const DetectionOptions(maxDetections: 3));
      expect(detections, hasLength(3));
      expect(detections.first.confidence, closeTo(0.59, 1e-6));
    });
  });

  test('anchor-based score is objectness times class score', () {
    final layout = ModelLayout.resolve(outputShapes: [
      [1, 10, 7]
    ], labelCount: 2);
    expect(layout.format, YoloFormat.anchorBased);
    final data = Float32List(10 * 7);
    void put(int anchor, List<double> values) {
      for (var k = 0; k < values.length; k++) {
        data[anchor * 7 + k] = values[k];
      }
    }

    put(2, [0.5, 0.5, 0.1, 0.1, 0.9, 0.2, 0.8]); // 0.72 class 1
    put(4, [0.2, 0.2, 0.1, 0.1, 0.2, 1.0, 0.0]); // objectness too low
    final post =
        Postprocessor(layout: layout, input: input640, labels: const ['a', 'b']);
    final d = post
        .process([OutputTensor(const [1, 10, 7], data)], identity,
            const DetectionOptions())
        .single;
    expect(d.label, 'b');
    expect(d.confidence, closeTo(0.72, 1e-6));
  });

  test('end-to-end output is read as xyxy without NMS', () {
    final layout = ModelLayout.resolve(outputShapes: [
      [1, 300, 6]
    ], labelCount: 3);
    final data = Float32List(300 * 6);
    data.setAll(0, [0.1, 0.1, 0.3, 0.3, 0.95, 2]);
    data.setAll(6, [0.1, 0.1, 0.3, 0.3, 0.90, 2]); // overlapping, kept
    final post = Postprocessor(
        layout: layout, input: input640, labels: const ['a', 'b', 'c']);
    final detections = post.process(
        [OutputTensor(const [1, 300, 6], data)],
        identity,
        const DetectionOptions());
    expect(detections, hasLength(2));
    expect(detections.first.label, 'c');
    expect(detections.first.box.left, closeTo(64, 1e-3));
    expect(detections.first.box.right, closeTo(192, 1e-3));
  });

  group('contours', () {
    Uint8List grid(List<String> rows) {
      final w = rows.first.length;
      final out = Uint8List(w * rows.length);
      for (var y = 0; y < rows.length; y++) {
        for (var x = 0; x < w; x++) {
          out[y * w + x] = rows[y][x] == '#' ? 1 : 0;
        }
      }
      return out;
    }

    test('traces the outline of the largest region', () {
      final mask = grid([
        '#.......',
        '..####..',
        '..####..',
        '..####..',
        '........',
      ]);
      final contour = traceLargestContour(mask, 8, 5);
      expect(contour.toSet(), {
        for (var x = 2; x <= 5; x++) GridPoint(x, 1),
        for (var x = 2; x <= 5; x++) GridPoint(x, 3),
        const GridPoint(2, 2),
        const GridPoint(5, 2),
      });
      final simplified = simplifyClosedPolygon(contour, 0.5);
      expect(simplified.toSet(), {
        const GridPoint(2, 1),
        const GridPoint(5, 1),
        const GridPoint(5, 3),
        const GridPoint(2, 3),
      });
    });

    test('handles concave shapes and single pixels', () {
      final mask = grid([
        '###.',
        '#...',
        '###.',
      ]);
      final contour = traceLargestContour(mask, 4, 3);
      expect(contour.toSet().length, 7);
      expect(traceLargestContour(grid(['.#.']), 3, 1),
          [const GridPoint(1, 0)]);
      expect(traceLargestContour(grid(['...']), 3, 1), isEmpty);
    });
  });

  test('segmentation outline follows the mask', () {
    // 1 class, 1 mask channel, prototypes 16x16 for a 64x64 input.
    final layout = ModelLayout.resolve(outputShapes: [
      [1, 6, 10],
      [1, 16, 16, 1],
    ], labelCount: 1);
    expect(layout.task, YoloTask.segment);
    final preds = anchorFreeTensor(
      n: 10,
      nc: 1,
      nm: 1,
      boxes: [(anchor: 0, box: [0.5, 0.5, 1.0, 1.0], cls: 0, score: 0.9)],
      masks: [
        [1.0]
      ],
    );
    // Disk of radius 5 cells centred in the grid: positive inside.
    final protos = Float32List(16 * 16);
    for (var y = 0; y < 16; y++) {
      for (var x = 0; x < 16; x++) {
        final d = math.sqrt(math.pow(x + 0.5 - 8, 2) + math.pow(y + 0.5 - 8, 2));
        protos[y * 16 + x] = d < 5 ? 5 : -5;
      }
    }
    const input = InputLayout(width: 64, height: 64, channelsFirst: false);
    final post = Postprocessor(layout: layout, input: input, labels: const ['o']);
    final d = post
        .process(
          [
            OutputTensor(const [1, 6, 10], preds),
            OutputTensor(const [1, 16, 16, 1], protos),
          ],
          const Letterbox(
              imageWidth: 64, imageHeight: 64, scale: 1, padX: 0, padY: 0),
          const DetectionOptions(),
        )
        .single;
    final polygon = d.polygon!;
    expect(polygon.length, greaterThan(6));
    // Disk radius is 5 cells = 20 input pixels around (32, 32).
    for (final p in polygon) {
      final r = math.sqrt(math.pow(p.dx - 32, 2) + math.pow(p.dy - 32, 2));
      expect(r, inInclusiveRange(14, 24));
    }
  });

  test('OutputTensor copies unaligned bytes and checks the size', () {
    final floats = Float32List.fromList([1.5, -2, 3]);
    final padded = Uint8List(13)..setAll(1, floats.buffer.asUint8List());
    final view = Uint8List.sublistView(padded, 1);
    final tensor = OutputTensor.fromBytes(const [1, 3], view);
    expect(tensor.data, [1.5, -2, 3]);
    expect(() => OutputTensor.fromBytes(const [1, 4], view),
        throwsA(isA<FlutterVisionException>()));
  });

  test('NMS keeps the best box of each overlapping group', () {
    final kept = nonMaxSuppression(
      [
        Candidate(0, 0, 10, 10, 0.9, 0, 0),
        Candidate(1, 1, 11, 11, 0.8, 0, 1),
        Candidate(50, 50, 60, 60, 0.7, 0, 2),
      ],
      iouThreshold: 0.45,
      classAgnostic: false,
      maxDetections: 10,
    );
    expect(kept.map((c) => c.anchor), [0, 2]);
  });
}
