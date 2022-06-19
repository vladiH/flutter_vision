import 'dart:typed_data';
import '../../flutter_vision.dart';
import '../utils/response_handler.dart';
import '../utils/result.dart';
import 'base.dart';

class AndroidFlutterVision extends BaseFlutterVision implements FlutterVision {
  @override
  Future<ResponseHandler> loadOcrModel(
      {required String modelPath,
      required String labels,
      int? numThreads,
      bool? useGpu,
      String? language,
      Map<String, String>? args}) async {
    try {
      final String testData = await loadTessData();
      final result = await channel.invokeMethod<String?>('loadOcrModel', {
        'model_path': modelPath,
        'is_asset': true,
        'num_threads': numThreads ?? 1,
        'use_gpu': useGpu ?? false,
        'label_path': labels,
        'image_mean': 0.0,
        'image_std': 255.0,
        'rotation': 90,
        'tess_data': testData,
        'arg': args,
        'language': language ?? 'eng'
      });
      if (result == null) {
        return Error(message: 'Unknown error');
      }
      switch (result.toString()) {
        case 'ok':
          return Success(message: result.toString());
        case 'error':
          return Error(message: result.toString());
        default:
          return Error(message: 'Not a valid status');
      }
    } catch (e, st) {
      return Error(message: 'Unknown error', stackTrace: st);
    }
  }

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
      results = await _ocrOnFrame(
          bytesList: bytesList,
          imageHeight: imageHeight,
          imageWidth: imageWidth,
          iouThreshold: iouThreshold ?? 0.4,
          confThreshold: confThreshold ?? 0.5,
          classIsText: classIsText);
      return Success(message: 'ok', data: results);
    } catch (e) {
      return Error(message: e.toString());
    }
  }

  Future<List<Map<String, dynamic>>> _ocrOnFrame({
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
        ).toOcrJson();
      }).toList();
      return result;
    } catch (e) {
      //print(e);
      rethrow;
    }
  }

  @override
  Future<ResponseHandler> loadYoloModel(
      {required String modelPath,
      required String labels,
      int? numThreads,
      bool? useGpu}) async {
    try {
      final result = await channel.invokeMethod<String?>('loadYoloModel', {
        'model_path': modelPath,
        'is_asset': true,
        'num_threads': numThreads ?? 1,
        'use_gpu': useGpu ?? false,
        'label_path': labels,
        'image_mean': 0.0,
        'image_std': 255.0,
        'rotation': 90,
      });
      if (result == null) {
        return Error(message: 'Unknown error');
      }
      switch (result.toString()) {
        case 'ok':
          return Success(message: result.toString());
        case 'error':
          return Error(message: result.toString());
        default:
          return Error(message: 'Not a valid status');
      }
    } catch (e, st) {
      return Error(message: 'Unknown error', stackTrace: st);
    }
  }

  @override
  Future<ResponseHandler> yoloOnFrame(
      {required List<Uint8List> bytesList,
      required int imageHeight,
      required int imageWidth,
      double? iouThreshold,
      double? confThreshold}) async {
    List<Map<String, dynamic>> results = [];
    try {
      results = await _yoloOnFrame(
        bytesList: bytesList,
        imageHeight: imageHeight,
        imageWidth: imageWidth,
        iouThreshold: iouThreshold ?? 0.4,
        confThreshold: confThreshold ?? 0.5,
      );
      return Success(message: 'ok', data: results);
    } catch (e) {
      return Error(message: e.toString());
    }
  }

  Future<List<Map<String, dynamic>>> _yoloOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    required double iouThreshold,
    required double confThreshold,
  }) async {
    try {
      final x = await channel.invokeMethod(
        'yoloOnFrame',
        {
          "bytesList": bytesList,
          "image_height": imageHeight,
          "image_width": imageWidth,
          "iou_threshold": iouThreshold,
          "conf_threshold": confThreshold,
        },
      );
      final List<Map<String, dynamic>> result = (x as List<dynamic>).map((e) {
        final result = Map<String, dynamic>.from(e);
        final List<double> yolo = result["yolo"] as List<double>;
        return OcrResult(
          confidence: yolo[4],
          box: Box(x1: yolo[0], x2: yolo[2], y1: yolo[1], y2: yolo[3]),
          image: result["image"] as Uint8List,
          tag: result["tag"] as String,
        ).toYoloJson();
      }).toList();
      return result;
    } catch (e) {
      //print(e);
      rethrow;
    }
  }
}
