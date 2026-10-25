import 'package:flutter/painting.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_vision/flutter_vision.dart';

void main() {
  group('BoxMapper', () {
    const image = Size(1000, 500);

    test('contain letterboxes vertically', () {
      final mapper = BoxMapper(
          imageSize: image, widgetSize: const Size(500, 500));
      expect(mapper.imageArea, const Rect.fromLTWH(0, 125, 500, 250));
      expect(mapper.mapPoint(const Offset(1000, 500)), const Offset(500, 375));
      expect(mapper.mapRect(const Rect.fromLTWH(100, 100, 200, 100)),
          const Rect.fromLTWH(50, 175, 100, 50));
    });

    test('cover crops the sides', () {
      final mapper = BoxMapper(
          imageSize: image,
          widgetSize: const Size(500, 500),
          fit: BoxFit.cover);
      // Scale 1, the image is centred: 250 px cropped on each side.
      expect(mapper.mapPoint(const Offset(250, 0)), Offset.zero);
      expect(mapper.mapPoint(const Offset(750, 500)), const Offset(500, 500));
    });

    test('mirror flips horizontally', () {
      final mapper = BoxMapper(
          imageSize: const Size(100, 100),
          widgetSize: const Size(100, 100),
          mirror: true);
      expect(mapper.mapPoint(const Offset(10, 20)), const Offset(90, 20));
      final rect = mapper.mapRect(const Rect.fromLTRB(10, 10, 30, 30));
      expect(rect, const Rect.fromLTRB(70, 10, 90, 30));
    });

    test('empty sizes do not throw', () {
      final mapper =
          BoxMapper(imageSize: Size.zero, widgetSize: const Size(10, 10));
      expect(mapper.mapPoint(const Offset(5, 5)), Offset.zero);
    });
  });

  group('CameraFrame', () {
    test('rotationFor', () {
      // Typical Android back camera in portrait.
      expect(
          CameraFrame.rotationFor(
              sensorOrientation: 90, deviceOrientation: 0, frontCamera: false),
          90);
      // Landscape left.
      expect(
          CameraFrame.rotationFor(
              sensorOrientation: 90, deviceOrientation: 90, frontCamera: false),
          0);
      // Front camera in portrait.
      expect(
          CameraFrame.rotationFor(
              sensorOrientation: 270, deviceOrientation: 0, frontCamera: true),
          270);
      expect(CameraFrame.degreesFor('landscapeRight'), 270);
    });

    test('upright size swaps for 90 and 270', () {
      final frame = CameraFrame(
        width: 640,
        height: 480,
        format: FrameFormat.nv21,
        planes: const [],
        rotation: 90,
      );
      expect(frame.uprightWidth, 480);
      expect(frame.uprightHeight, 640);
    });
  });
}
