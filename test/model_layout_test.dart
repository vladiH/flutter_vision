import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_vision/flutter_vision.dart';

void main() {
  group('InputLayout', () {
    test('reads NHWC', () {
      final input = InputLayout.fromShape([1, 640, 480, 3]);
      expect(input.width, 480);
      expect(input.height, 640);
      expect(input.channelsFirst, isFalse);
    });

    test('reads NCHW', () {
      final input = InputLayout.fromShape([1, 3, 320, 640]);
      expect(input.width, 640);
      expect(input.height, 320);
      expect(input.channelsFirst, isTrue);
    });

    test('rejects non image inputs', () {
      expect(() => InputLayout.fromShape([1, 10]),
          throwsA(isA<FlutterVisionException>()));
      expect(() => InputLayout.fromShape([1, 640, 640, 1]),
          throwsA(isA<FlutterVisionException>()));
    });
  });

  group('ModelLayout.resolve', () {
    test('YOLOv8/v11 detection', () {
      final layout = ModelLayout.resolve(outputShapes: [
        [1, 84, 8400]
      ], labelCount: 80);
      expect(layout.format, YoloFormat.anchorFree);
      expect(layout.task, YoloTask.detect);
      expect(layout.channelsFirst, isTrue);
      expect(layout.numCandidates, 8400);
      expect(layout.numClasses, 80);
    });

    test('YOLOv8 transposed output', () {
      final layout = ModelLayout.resolve(outputShapes: [
        [1, 8400, 84]
      ], labelCount: 80);
      expect(layout.format, YoloFormat.anchorFree);
      expect(layout.channelsFirst, isFalse);
      expect(layout.numCandidates, 8400);
    });

    test('YOLOv5 detection', () {
      final layout = ModelLayout.resolve(outputShapes: [
        [1, 25200, 85]
      ], labelCount: 80);
      expect(layout.format, YoloFormat.anchorBased);
      expect(layout.channelsFirst, isFalse);
      expect(layout.classOffset, 5);
    });

    test('end-to-end (YOLOv10 / YOLO26)', () {
      final layout = ModelLayout.resolve(outputShapes: [
        [1, 300, 6]
      ], labelCount: 80);
      expect(layout.format, YoloFormat.endToEnd);
      expect(layout.numCandidates, 300);
    });

    test('segmentation with outputs in either order', () {
      for (final shapes in [
        [
          [1, 116, 8400],
          [1, 160, 160, 32],
        ],
        [
          [1, 160, 160, 32],
          [1, 116, 8400],
        ],
      ]) {
        final layout =
            ModelLayout.resolve(outputShapes: shapes, labelCount: 80);
        expect(layout.task, YoloTask.segment);
        expect(layout.format, YoloFormat.anchorFree);
        expect(layout.numMasks, 32);
        expect(layout.protoHeight, 160);
        expect(layout.protosChannelsFirst, isFalse);
        expect(layout.maskOffset, 84);
        expect(shapes[layout.predictionsIndex], [1, 116, 8400]);
        expect(shapes[layout.protosIndex!], [1, 160, 160, 32]);
      }
    });

    test('segmentation with channels-first prototypes', () {
      final layout = ModelLayout.resolve(outputShapes: [
        [1, 116, 8400],
        [1, 32, 160, 160],
      ], labelCount: 80);
      expect(layout.protosChannelsFirst, isTrue);
      expect(layout.numMasks, 32);
    });

    test('guesses the layout without labels', () {
      final layout = ModelLayout.resolve(outputShapes: [
        [1, 84, 8400]
      ], labelCount: 0);
      expect(layout.format, YoloFormat.anchorFree);
      expect(layout.numClasses, 80);
    });

    test('reports labels that do not match the model', () {
      expect(
        () => ModelLayout.resolve(outputShapes: [
          [1, 84, 8400]
        ], labelCount: 3),
        throwsA(isA<FlutterVisionException>().having(
            (e) => e.message, 'message', contains('80 classes'))),
      );
    });

    test('reports a task mismatch', () {
      expect(
        () => ModelLayout.resolve(outputShapes: [
          [1, 84, 8400]
        ], labelCount: 80, task: YoloTask.segment),
        throwsA(isA<FlutterVisionException>()),
      );
    });

    test('honours an explicit format', () {
      // 1 class: v5 has 6 channels, like an end-to-end model.
      final layout = ModelLayout.resolve(
        outputShapes: [
          [1, 300, 6]
        ],
        labelCount: 1,
        format: YoloFormat.anchorBased,
      );
      expect(layout.format, YoloFormat.anchorBased);
    });

    test('rejects outputs that are not YOLO', () {
      expect(
        () => ModelLayout.resolve(outputShapes: [
          [1, 1001]
        ], labelCount: 0),
        throwsA(isA<FlutterVisionException>()),
      );
    });
  });

  test('parseLabels skips blank lines and index prefixes', () {
    expect(
      YoloModel.parseLabels('person\r\n\n  bicycle \n0: car\n3 wheeler\n'),
      ['person', 'bicycle', 'car', '3 wheeler'],
    );
  });
}
