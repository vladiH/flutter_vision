import 'dart:typed_data';

class OcrResult {
  final double confidence;
  final Box box;
  final String? text;
  final Uint8List image;
  final String tag;
  OcrResult({
    required this.confidence,
    required this.box,
    this.text,
    required this.image,
    required this.tag,
  });

  Map<String, dynamic> toOcrJson() {
    return {
      "confidence": confidence,
      "box": box.toJson(),
      "text": text,
      "image": image,
      "tag": tag,
    };
  }

  Map<String, dynamic> toYoloJson() {
    return {
      "confidence": confidence,
      "box": box.toJson(),
      "image": image,
      "tag": tag,
    };
  }
}

class Box {
  final double x1, y1, x2, y2;
  Box({required this.x1, required this.y1, required this.x2, required this.y2});
  Map<String, dynamic> toJson() {
    return {
      "x1": x1,
      "y1": y1,
      "x2": x2,
      "y2": y2,
    };
  }
}
