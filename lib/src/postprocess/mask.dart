import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui';

import 'decoder.dart';
import 'model_layout.dart';

/// Builds the outline of a segmented object from the mask prototypes.
class MaskTracer {
  MaskTracer(this.layout, {required this.inputWidth, required this.inputHeight});

  final ModelLayout layout;
  final int inputWidth;
  final int inputHeight;

  /// Upsampling applied to the prototype grid before tracing, so outlines are
  /// smoother than the 160x160 prototypes.
  static const double maxUpsample = 4;

  /// Largest side, in samples, of the grid traced for one object.
  static const int maxSamples = 320;

  /// Returns the outline of [candidate] in model input pixels, or an empty
  /// list when the mask is empty.
  List<Offset> outline(
    Float32List protos,
    Float32List coefficients,
    Candidate candidate,
    double threshold,
  ) {
    final mh = layout.protoHeight;
    final mw = layout.protoWidth;
    final nm = layout.numMasks;
    final toProtoX = mw / inputWidth;
    final toProtoY = mh / inputHeight;

    final bx1 = (candidate.x1 * toProtoX).clamp(0.0, mw.toDouble());
    final bx2 = (candidate.x2 * toProtoX).clamp(0.0, mw.toDouble());
    final by1 = (candidate.y1 * toProtoY).clamp(0.0, mh.toDouble());
    final by2 = (candidate.y2 * toProtoY).clamp(0.0, mh.toDouble());
    final boxW = bx2 - bx1;
    final boxH = by2 - by1;
    if (boxW < 0.5 || boxH < 0.5) return const [];

    // Mask logits on the prototype cells covering the box (+1 cell margin).
    final gx0 = math.max(0, bx1.floor() - 1);
    final gy0 = math.max(0, by1.floor() - 1);
    final gx1 = math.min(mw - 1, bx2.ceil());
    final gy1 = math.min(mh - 1, by2.ceil());
    final gw = gx1 - gx0 + 1;
    final gh = gy1 - gy0 + 1;
    final logits = Float32List(gw * gh);
    final planeSize = mh * mw;
    for (var y = 0; y < gh; y++) {
      for (var x = 0; x < gw; x++) {
        final cell = (gy0 + y) * mw + gx0 + x;
        var sum = 0.0;
        if (layout.protosChannelsFirst) {
          for (var k = 0; k < nm; k++) {
            sum += coefficients[k] * protos[k * planeSize + cell];
          }
        } else {
          final base = cell * nm;
          for (var k = 0; k < nm; k++) {
            sum += coefficients[k] * protos[base + k];
          }
        }
        logits[y * gw + x] = sum;
      }
    }

    // Sample the box at a finer resolution with bilinear interpolation.
    var factor = math.min(maxUpsample, inputWidth / mw);
    factor = math.max(1.0, factor);
    factor = math.min(factor, maxSamples / math.max(boxW, boxH));
    final w = math.max(1, (boxW * factor).ceil());
    final h = math.max(1, (boxH * factor).ceil());
    final logitThreshold = math.log(threshold / (1 - threshold));
    final mask = Uint8List(w * h);
    for (var y = 0; y < h; y++) {
      final fy = (by1 + (y + 0.5) / factor - 0.5 - gy0).clamp(0.0, gh - 1.0);
      final y0 = fy.floor();
      final y1 = math.min(y0 + 1, gh - 1);
      final wy = fy - y0;
      for (var x = 0; x < w; x++) {
        final fx = (bx1 + (x + 0.5) / factor - 0.5 - gx0).clamp(0.0, gw - 1.0);
        final x0 = fx.floor();
        final x1 = math.min(x0 + 1, gw - 1);
        final wx = fx - x0;
        final top = logits[y0 * gw + x0] * (1 - wx) + logits[y0 * gw + x1] * wx;
        final bottom =
            logits[y1 * gw + x0] * (1 - wx) + logits[y1 * gw + x1] * wx;
        if (top * (1 - wy) + bottom * wy > logitThreshold) mask[y * w + x] = 1;
      }
    }

    final contour = traceLargestContour(mask, w, h);
    if (contour.length < 3) return const [];
    final simplified = simplifyClosedPolygon(contour, 0.75);
    return [
      for (final p in simplified)
        Offset(
          (bx1 + (p.x + 0.5) / factor) / toProtoX,
          (by1 + (p.y + 0.5) / factor) / toProtoY,
        ),
    ];
  }
}

/// Grid point used while tracing.
class GridPoint {
  const GridPoint(this.x, this.y);
  final int x;
  final int y;

  @override
  bool operator ==(Object other) =>
      other is GridPoint && other.x == x && other.y == y;

  @override
  int get hashCode => Object.hash(x, y);

  @override
  String toString() => '($x, $y)';
}

// Clockwise 8-neighbourhood (y grows downwards), starting at West.
const List<int> _dx = [-1, -1, 0, 1, 1, 1, 0, -1];
const List<int> _dy = [0, -1, -1, -1, 0, 1, 1, 1];

int _direction(int dx, int dy) {
  for (var d = 0; d < 8; d++) {
    if (_dx[d] == dx && _dy[d] == dy) return d;
  }
  return 0;
}

/// Outer boundary of the largest 8-connected region of ones in [mask],
/// in clockwise order. Uses Moore-neighbour tracing.
List<GridPoint> traceLargestContour(Uint8List mask, int width, int height) {
  final labels = Int32List(width * height);
  final stack = <int>[];
  var label = 0;
  var bestLabel = 0;
  var bestSize = 0;
  var bestStart = -1;
  for (var start = 0; start < mask.length; start++) {
    if (mask[start] == 0 || labels[start] != 0) continue;
    label++;
    var size = 0;
    labels[start] = label;
    stack.add(start);
    while (stack.isNotEmpty) {
      final index = stack.removeLast();
      size++;
      final x = index % width;
      final y = index ~/ width;
      for (var d = 0; d < 8; d++) {
        final nx = x + _dx[d];
        final ny = y + _dy[d];
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        final n = ny * width + nx;
        if (mask[n] == 1 && labels[n] == 0) {
          labels[n] = label;
          stack.add(n);
        }
      }
    }
    if (size > bestSize) {
      bestSize = size;
      bestLabel = label;
      bestStart = start;
    }
  }
  if (bestStart < 0) return const [];

  bool inside(int x, int y) =>
      x >= 0 &&
      y >= 0 &&
      x < width &&
      y < height &&
      labels[y * width + x] == bestLabel;

  final sx = bestStart % width;
  final sy = bestStart ~/ width;
  final contour = <GridPoint>[];
  var cx = sx;
  var cy = sy;
  // The start is the first pixel in raster order, so its West neighbour is
  // outside the region: begin the search from there.
  var back = 0;
  int? secondX;
  int? secondY;
  final maxSteps = 4 * width * height + 8;
  for (var step = 0; step < maxSteps; step++) {
    contour.add(GridPoint(cx, cy));
    var found = false;
    var nx = 0;
    var ny = 0;
    var newBack = 0;
    for (var k = 1; k <= 8; k++) {
      final d = (back + k) % 8;
      final tx = cx + _dx[d];
      final ty = cy + _dy[d];
      if (inside(tx, ty)) {
        final prev = (d + 7) % 8;
        newBack = _direction(cx + _dx[prev] - tx, cy + _dy[prev] - ty);
        nx = tx;
        ny = ty;
        found = true;
        break;
      }
    }
    if (!found) break; // Single isolated pixel.
    if (step == 0) {
      secondX = nx;
      secondY = ny;
    } else if (cx == sx && cy == sy && nx == secondX && ny == secondY) {
      contour.removeLast();
      break;
    }
    cx = nx;
    cy = ny;
    back = newBack;
  }
  return contour;
}

/// Douglas-Peucker simplification of a closed polygon.
List<GridPoint> simplifyClosedPolygon(List<GridPoint> points, double epsilon) {
  if (points.length < 4) return points;
  final first = points.first;
  var far = 0;
  var farDistance = -1.0;
  for (var i = 1; i < points.length; i++) {
    final dx = (points[i].x - first.x).toDouble();
    final dy = (points[i].y - first.y).toDouble();
    final d = dx * dx + dy * dy;
    if (d > farDistance) {
      farDistance = d;
      far = i;
    }
  }
  final a = _simplifyOpen(points.sublist(0, far + 1), epsilon);
  final b = _simplifyOpen([...points.sublist(far), first], epsilon);
  return [...a.sublist(0, a.length - 1), ...b.sublist(0, b.length - 1)];
}

List<GridPoint> _simplifyOpen(List<GridPoint> points, double epsilon) {
  if (points.length < 3) return points;
  final keep = List<bool>.filled(points.length, false);
  keep[0] = true;
  keep[points.length - 1] = true;
  final ranges = <(int, int)>[(0, points.length - 1)];
  while (ranges.isNotEmpty) {
    final (start, end) = ranges.removeLast();
    if (end - start < 2) continue;
    final a = points[start];
    final b = points[end];
    final abx = (b.x - a.x).toDouble();
    final aby = (b.y - a.y).toDouble();
    final length = math.sqrt(abx * abx + aby * aby);
    var maxDistance = -1.0;
    var index = start;
    for (var i = start + 1; i < end; i++) {
      final p = points[i];
      final double distance;
      if (length == 0) {
        final dx = (p.x - a.x).toDouble();
        final dy = (p.y - a.y).toDouble();
        distance = math.sqrt(dx * dx + dy * dy);
      } else {
        distance = ((p.x - a.x) * aby - (p.y - a.y) * abx).abs() / length;
      }
      if (distance > maxDistance) {
        maxDistance = distance;
        index = i;
      }
    }
    if (maxDistance > epsilon) {
      keep[index] = true;
      ranges
        ..add((start, index))
        ..add((index, end));
    }
  }
  return [
    for (var i = 0; i < points.length; i++)
      if (keep[i]) points[i],
  ];
}
