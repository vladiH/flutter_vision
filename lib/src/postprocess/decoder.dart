import 'dart:math' as math;
import 'dart:typed_data';

import '../types.dart';
import 'model_layout.dart';

/// A box proposed by the model, in model input pixels.
class Candidate {
  Candidate(this.x1, this.y1, this.x2, this.y2, this.score, this.classIndex,
      this.anchor);

  double x1;
  double y1;
  double x2;
  double y2;
  final double score;
  final int classIndex;

  /// Position of the candidate in the predictions tensor, used to read its
  /// mask coefficients.
  final int anchor;

  double get area => math.max(0, x2 - x1) * math.max(0, y2 - y1);
}

/// Turns the raw predictions tensor into scored candidates.
class Decoder {
  Decoder(this.layout, {required this.inputWidth, required this.inputHeight});

  final ModelLayout layout;
  final int inputWidth;
  final int inputHeight;

  /// Whether box coordinates are normalized (0-1) instead of pixels.
  /// Ultralytics TFLite exports are normalized; other converters are not.
  /// Detected on the first call that produces candidates.
  bool? normalized;

  /// Upper bound of candidates kept before NMS, highest scores first.
  static const int maxCandidates = 3000;

  List<Candidate> decode(Float32List data, DetectionOptions options) {
    final candidates = switch (layout.format) {
      YoloFormat.anchorFree => _anchorFree(data, options),
      YoloFormat.anchorBased => _anchorBased(data, options),
      YoloFormat.endToEnd => _endToEnd(data, options),
    };
    if (candidates.isEmpty) return candidates;

    normalized ??= _looksNormalized(candidates);
    if (normalized!) {
      for (final c in candidates) {
        c.x1 *= inputWidth;
        c.x2 *= inputWidth;
        c.y1 *= inputHeight;
        c.y2 *= inputHeight;
      }
    }
    candidates.sort((a, b) => b.score.compareTo(a.score));
    if (candidates.length > maxCandidates) {
      return candidates.sublist(0, maxCandidates);
    }
    return candidates;
  }

  static bool _looksNormalized(List<Candidate> candidates) {
    var maxValue = 0.0;
    for (final c in candidates) {
      maxValue = math.max(maxValue, math.max(c.x2.abs(), c.y2.abs()));
    }
    return maxValue <= 2.0;
  }

  bool _keepClass(int cls, DetectionOptions options) =>
      options.classes == null || options.classes!.contains(cls);

  /// YOLOv8/v9/v11/v12: box (cx, cy, w, h) + class scores [+ masks].
  List<Candidate> _anchorFree(Float32List data, DetectionOptions options) {
    final n = layout.numCandidates;
    final c = layout.numChannels;
    final nc = layout.numClasses;
    final threshold = options.confidenceThreshold;
    final result = <Candidate>[];

    if (layout.channelsFirst) {
      // data[channel * n + anchor]: walk each class row contiguously.
      final best = Float32List(n)..fillRange(0, n, -double.infinity);
      final bestClass = Int32List(n);
      for (var k = 0; k < nc; k++) {
        final base = (4 + k) * n;
        for (var i = 0; i < n; i++) {
          final v = data[base + i];
          if (v > best[i]) {
            best[i] = v;
            bestClass[i] = k;
          }
        }
      }
      for (var i = 0; i < n; i++) {
        final score = best[i];
        if (score < threshold || !_keepClass(bestClass[i], options)) continue;
        final cx = data[i];
        final cy = data[n + i];
        final w = data[2 * n + i];
        final h = data[3 * n + i];
        result.add(Candidate(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
            score, bestClass[i], i));
      }
    } else {
      for (var i = 0; i < n; i++) {
        final base = i * c;
        var score = -double.infinity;
        var cls = 0;
        for (var k = 0; k < nc; k++) {
          final v = data[base + 4 + k];
          if (v > score) {
            score = v;
            cls = k;
          }
        }
        if (score < threshold || !_keepClass(cls, options)) continue;
        final cx = data[base];
        final cy = data[base + 1];
        final w = data[base + 2];
        final h = data[base + 3];
        result.add(Candidate(
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, cls, i));
      }
    }
    return result;
  }

  /// YOLOv5/v7: box (cx, cy, w, h) + objectness + class scores [+ masks].
  /// Final score = objectness * class score.
  List<Candidate> _anchorBased(Float32List data, DetectionOptions options) {
    final n = layout.numCandidates;
    final c = layout.numChannels;
    final nc = layout.numClasses;
    final threshold = options.confidenceThreshold;
    final first = layout.channelsFirst;
    double at(int anchor, int channel) =>
        first ? data[channel * n + anchor] : data[anchor * c + channel];

    final result = <Candidate>[];
    for (var i = 0; i < n; i++) {
      final objectness = at(i, 4);
      if (objectness < threshold) continue;
      var best = -double.infinity;
      var cls = 0;
      for (var k = 0; k < nc; k++) {
        final v = at(i, 5 + k);
        if (v > best) {
          best = v;
          cls = k;
        }
      }
      final score = objectness * best;
      if (score < threshold || !_keepClass(cls, options)) continue;
      final cx = at(i, 0);
      final cy = at(i, 1);
      final w = at(i, 2);
      final h = at(i, 3);
      result.add(Candidate(
          cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, cls, i));
    }
    return result;
  }

  /// YOLOv10/YOLO26: x1, y1, x2, y2, score, class [+ masks]. Already
  /// filtered by the model, no NMS needed.
  List<Candidate> _endToEnd(Float32List data, DetectionOptions options) {
    final n = layout.numCandidates;
    final c = layout.numChannels;
    final first = layout.channelsFirst;
    double at(int anchor, int channel) =>
        first ? data[channel * n + anchor] : data[anchor * c + channel];

    final result = <Candidate>[];
    for (var i = 0; i < n; i++) {
      final score = at(i, 4);
      if (score < options.confidenceThreshold) continue;
      final cls = at(i, 5).round();
      if (cls < 0 || !_keepClass(cls, options)) continue;
      result.add(
          Candidate(at(i, 0), at(i, 1), at(i, 2), at(i, 3), score, cls, i));
    }
    return result;
  }

  /// Mask coefficients of [candidate].
  Float32List maskCoefficients(Float32List data, Candidate candidate) {
    final nm = layout.numMasks;
    final out = Float32List(nm);
    final offset = layout.maskOffset;
    final n = layout.numCandidates;
    final c = layout.numChannels;
    for (var k = 0; k < nm; k++) {
      out[k] = layout.channelsFirst
          ? data[(offset + k) * n + candidate.anchor]
          : data[candidate.anchor * c + offset + k];
    }
    return out;
  }
}

/// Non-maximum suppression over candidates sorted by descending score.
List<Candidate> nonMaxSuppression(
  List<Candidate> sorted, {
  required double iouThreshold,
  required bool classAgnostic,
  required int maxDetections,
}) {
  final kept = <Candidate>[];
  final suppressed = List<bool>.filled(sorted.length, false);
  for (var i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue;
    final a = sorted[i];
    kept.add(a);
    if (kept.length >= maxDetections) break;
    final areaA = a.area;
    for (var j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue;
      final b = sorted[j];
      if (!classAgnostic && a.classIndex != b.classIndex) continue;
      final w = math.min(a.x2, b.x2) - math.max(a.x1, b.x1);
      if (w <= 0) continue;
      final h = math.min(a.y2, b.y2) - math.max(a.y1, b.y1);
      if (h <= 0) continue;
      final inter = w * h;
      final union = areaA + b.area - inter;
      if (union > 0 && inter / union > iouThreshold) suppressed[j] = true;
    }
  }
  return kept;
}
