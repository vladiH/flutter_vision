import 'dart:typed_data';
import '../../flutter_vision.dart';
import '../utils/response_handler.dart';
import '../utils/result.dart';
import 'base.dart';

class AndroidFlutterVision extends BaseFlutterVision implements FlutterVision {
  @override
  Future<ResponseHandler> ocrOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    required List<int> classIsText,
    double? iouThreshold,
    double? confThreshold,
  }) async {
    List<Map<String, dynamic>> results = [];
    try {
      if (isLoadedOcrModel) {
        results = await _scanOnAndroid(
            bytesList: bytesList,
            imageHeight: imageHeight,
            imageWidth: imageWidth,
            iouThreshold: iouThreshold ?? 0.4,
            confThreshold: confThreshold ?? 0.5,
            classIsText: classIsText);
      }
      return Success(message: 'ok', data: results);
    } catch (e) {
      return Error(message: e.toString());
    }
  }

  Future<List<Map<String, dynamic>>> _scanOnAndroid({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    required double iouThreshold,
    required double confThreshold,
    required List<int> classIsText,
  }) async {
    try {
      final x = await channel.invokeMethod(
        'ocrOnFrame',
        {
          "bytesList": bytesList,
          "image_height": imageHeight,
          "image_width": imageWidth,
          "iou_threshold": iouThreshold,
          "conf_threshold": confThreshold,
          "class_is_text": classIsText
        },
      );
      final List<Map<String, dynamic>> result = (x as List<dynamic>).map((e) {
        final result = Map<String, dynamic>.from(e);
        final List<double> yolo = result["yolo"] as List<double>;
        return OcrResult(
          confidence: yolo[4],
          box: Box(x1: yolo[0], x2: yolo[2], y1: yolo[1], y2: yolo[3]),
          image: result["image"] as Uint8List,
          text: result["prediction"] as String,
          tag: result["tag"] as String,
        ).toJson();
      }).toList();
      return result;
    } catch (e) {
      //print(e);
      rethrow;
    }
  }
}
