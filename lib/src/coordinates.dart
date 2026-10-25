import 'package:flutter/painting.dart';

/// Converts detection coordinates (pixels of the upright image) into the
/// coordinate space of the widget that displays the image.
///
/// Use the same [fit] as the widget showing the picture or camera preview:
/// `BoxFit.contain` for `Image` with default settings, `BoxFit.cover` for a
/// full-screen camera preview. Set [mirror] for the front camera preview.
///
/// ```dart
/// final mapper = BoxMapper(
///   imageSize: result.imageSize,
///   widgetSize: constraints.biggest,
///   fit: BoxFit.cover,
/// );
/// final rect = mapper.mapRect(detection.box);
/// ```
class BoxMapper {
  factory BoxMapper({
    required Size imageSize,
    required Size widgetSize,
    BoxFit fit = BoxFit.contain,
    Alignment alignment = Alignment.center,
    bool mirror = false,
  }) {
    if (imageSize.isEmpty || widgetSize.isEmpty) {
      return BoxMapper._(Rect.zero, Rect.zero, widgetSize, mirror);
    }
    final fitted = applyBoxFit(fit, imageSize, widgetSize);
    final source = alignment.inscribe(fitted.source, Offset.zero & imageSize);
    final destination =
        alignment.inscribe(fitted.destination, Offset.zero & widgetSize);
    return BoxMapper._(source, destination, widgetSize, mirror);
  }

  BoxMapper._(this._source, this._destination, this.widgetSize, this.mirror);

  final Rect _source;
  final Rect _destination;
  final Size widgetSize;
  final bool mirror;

  /// Area of the widget actually covered by the image.
  Rect get imageArea => _destination;

  Offset mapPoint(Offset point) {
    if (_source.isEmpty) return Offset.zero;
    final sx = _destination.width / _source.width;
    final sy = _destination.height / _source.height;
    var x = _destination.left + (point.dx - _source.left) * sx;
    final y = _destination.top + (point.dy - _source.top) * sy;
    if (mirror) x = widgetSize.width - x;
    return Offset(x, y);
  }

  Rect mapRect(Rect rect) =>
      Rect.fromPoints(mapPoint(rect.topLeft), mapPoint(rect.bottomRight));

  List<Offset> mapPolygon(List<Offset> polygon) =>
      [for (final p in polygon) mapPoint(p)];
}
