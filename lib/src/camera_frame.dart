import 'package:flutter/foundation.dart';

/// Pixel layout of a [CameraFrame].
enum FrameFormat {
  /// One plane, Y followed by interleaved V/U. Android `ImageFormatGroup.nv21`.
  nv21,

  /// Three planes (Y, U, V) with row and pixel strides.
  /// Android `ImageFormatGroup.yuv420` and iOS `ImageFormatGroup.yuv420`
  /// (two planes: Y and interleaved U/V).
  yuv420,

  /// One plane, 4 bytes per pixel. iOS `ImageFormatGroup.bgra8888`.
  bgra8888,

  /// One plane, 4 bytes per pixel.
  rgba8888,
}

/// One plane of a [CameraFrame].
@immutable
class FramePlane {
  const FramePlane({
    required this.bytes,
    required this.bytesPerRow,
    this.bytesPerPixel = 1,
  });

  final Uint8List bytes;

  /// Row stride. Can be larger than the width because of padding.
  final int bytesPerRow;

  /// Pixel stride: distance in bytes between two pixels of the same row.
  final int bytesPerPixel;
}

/// A camera image, for example a `CameraImage` from the `camera` plugin.
///
/// ```dart
/// final frame = CameraFrame(
///   width: image.width,
///   height: image.height,
///   format: FrameFormat.nv21,
///   planes: [
///     for (final p in image.planes)
///       FramePlane(
///         bytes: p.bytes,
///         bytesPerRow: p.bytesPerRow,
///         bytesPerPixel: p.bytesPerPixel ?? 1,
///       ),
///   ],
///   rotation: CameraFrame.rotationFor(
///     sensorOrientation: controller.description.sensorOrientation,
///     deviceOrientation: controller.value.deviceOrientation,
///     frontCamera: false,
///   ),
/// );
/// ```
@immutable
class CameraFrame {
  const CameraFrame({
    required this.width,
    required this.height,
    required this.format,
    required this.planes,
    this.rotation = 0,
  })  : assert(width > 0 && height > 0),
        assert(rotation == 0 ||
            rotation == 90 ||
            rotation == 180 ||
            rotation == 270);

  /// Size as delivered by the camera, before [rotation].
  final int width;
  final int height;
  final FrameFormat format;
  final List<FramePlane> planes;

  /// Clockwise rotation in degrees (0, 90, 180, 270) that makes the frame
  /// upright. Detections are returned in the rotated (upright) frame.
  final int rotation;

  /// Rotation needed for a frame captured by a camera with
  /// [sensorOrientation] (from `CameraDescription.sensorOrientation`) while
  /// the device is in [deviceOrientation].
  ///
  /// [deviceOrientation] is `DeviceOrientation` from `package:flutter/services.dart`
  /// expressed as degrees: portraitUp 0, landscapeLeft 90, portraitDown 180,
  /// landscapeRight 270. Use [degreesFor] to convert.
  static int rotationFor({
    required int sensorOrientation,
    required int deviceOrientation,
    required bool frontCamera,
  }) {
    final rotation = frontCamera
        ? (sensorOrientation + deviceOrientation) % 360
        : (sensorOrientation - deviceOrientation + 360) % 360;
    return (rotation ~/ 90) * 90;
  }

  /// Converts a `DeviceOrientation` name to degrees for [rotationFor].
  static int degreesFor(String deviceOrientationName) {
    switch (deviceOrientationName) {
      case 'landscapeLeft':
        return 90;
      case 'portraitDown':
        return 180;
      case 'landscapeRight':
        return 270;
      case 'portraitUp':
      default:
        return 0;
    }
  }

  /// Width once [rotation] is applied.
  int get uprightWidth => rotation % 180 == 0 ? width : height;

  /// Height once [rotation] is applied.
  int get uprightHeight => rotation % 180 == 0 ? height : width;
}
