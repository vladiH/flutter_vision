import 'package:flutter/material.dart';
import 'package:flutter_vision/flutter_vision.dart';

/// Draws boxes, labels and segmentation outlines over an image or preview.
class DetectionPainter extends CustomPainter {
  DetectionPainter({
    required this.result,
    this.fit = BoxFit.contain,
    this.mirror = false,
  });

  final YoloResult result;
  final BoxFit fit;
  final bool mirror;

  static const List<Color> _palette = [
    Color(0xFFFF3B30),
    Color(0xFF34C759),
    Color(0xFF007AFF),
    Color(0xFFFF9500),
    Color(0xFFAF52DE),
    Color(0xFF00C7BE),
    Color(0xFFFFCC00),
    Color(0xFFFF2D55),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final mapper = BoxMapper(
      imageSize: result.imageSize,
      widgetSize: size,
      fit: fit,
      mirror: mirror,
    );
    canvas.save();
    canvas.clipRect(Offset.zero & size);
    for (final detection in result.detections) {
      final color = _palette[detection.classIndex % _palette.length];
      final polygon = detection.polygon;
      if (polygon != null && polygon.length >= 3) {
        final path = Path()..addPolygon(mapper.mapPolygon(polygon), true);
        canvas.drawPath(path, Paint()..color = color.withValues(alpha: 0.35));
        canvas.drawPath(
          path,
          Paint()
            ..color = color
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.5,
        );
      }
      final rect = mapper.mapRect(detection.box);
      canvas.drawRect(
        rect,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5,
      );
      final label = TextPainter(
        text: TextSpan(
          text:
              ' ${detection.label} ${(detection.confidence * 100).toStringAsFixed(0)}% ',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      final top = rect.top - label.height < 0 ? rect.top : rect.top - label.height;
      canvas.drawRect(
        Rect.fromLTWH(rect.left, top, label.width, label.height),
        Paint()..color = color,
      );
      label.paint(canvas, Offset(rect.left, top));
    }
    canvas.restore();
  }

  @override
  bool shouldRepaint(DetectionPainter oldDelegate) =>
      oldDelegate.result != result ||
      oldDelegate.fit != fit ||
      oldDelegate.mirror != mirror;
}
