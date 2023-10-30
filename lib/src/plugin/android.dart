import 'dart:typed_data';

import '../../flutter_vision.dart';
import 'base.dart';

class AndroidFlutterVision extends BaseFlutterVision implements FlutterVision {
  // @override
  // Future<void> loadOcrModel(
  //     {required String modelPath,
  //     required String labels,
  //     int? numThreads,
  //     bool? useGpu,
  //     String? language,
  //     Map<String, String>? args}) async {
  //   try {
  //     final String testData = await loadTessData();
  //     await channel.invokeMethod<String>('loadOcrModel', {
  //       'model_path': modelPath,
  //       'is_asset': true,
  //       'num_threads': numThreads ?? 1,
  //       'use_gpu': useGpu ?? false,
  //       'label_path': labels,
  //       'image_mean': 0.0,
  //       'image_std': 255.0,
  //       'rotation': 90,
  //       'tess_data': testData,
  //       'arg': args,
  //       'language': language ?? 'eng'
  //     });
  //   } catch (e) {
  //     rethrow;
  //   }
  // }

  // @override
  // Future<List<Map<String, dynamic>>> ocrOnFrame({
  //   required List<Uint8List> bytesList,
  //   required int imageHeight,
  //   required int imageWidth,
  //   required List<int> classIsText,
  //   double? iouThreshold,
  //   double? confThreshold,
  // }) async {
  //   try {
  //     return await _ocrOnFrame(
  //         bytesList: bytesList,
  //         imageHeight: imageHeight,
  //         imageWidth: imageWidth,
  //         iouThreshold: iouThreshold ?? 0.4,
  //         confThreshold: confThreshold ?? 0.5,
  //         classIsText: classIsText);
  //   } catch (e) {
  //     rethrow;
  //   }
  // }

  // Future<List<Map<String, dynamic>>> _ocrOnFrame({
  //   required List<Uint8List> bytesList,
  //   required int imageHeight,
  //   required int imageWidth,
  //   required double iouThreshold,
  //   required double confThreshold,
  //   required List<int> classIsText,
  // }) async {
  //   try {
  //     final x = await channel.invokeMethod<List<Map<String, dynamic>>>(
  //       'ocrOnFrame',
  //       {
  //         "bytesList": bytesList,
  //         "image_height": imageHeight,
  //         "image_width": imageWidth,
  //         "iou_threshold": iouThreshold,
  //         "conf_threshold": confThreshold,
  //         "class_is_text": classIsText
  //       },
  //     );
  //     return x ?? [];
  //   } catch (e) {
  //     rethrow;
  //   }
  // }

  @override
  Future<void> loadYoloModel(
      {required String modelPath,
      required String labels,
      required String modelVersion,
      bool? quantization,
      int? numThreads,
      bool? useGpu}) async {
    try {
      await channel.invokeMethod<String?>('loadYoloModel', {
        'model_path': modelPath,
        'is_asset': true,
        'quantization': quantization ?? false,
        'num_threads': numThreads ?? 1,
        'use_gpu': useGpu ?? false,
        'label_path': labels,
        'rotation': 90,
        'model_version': modelVersion
      });
    } catch (e) {
      rethrow;
    }
  }

  @override
  Future<List<Map<String, dynamic>>> yoloOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    double? iouThreshold,
    double? confThreshold,
    double? classThreshold,
  }) async {
    try {
      return (await _yoloOnFrame(
          bytesList: bytesList,
          imageHeight: imageHeight,
          imageWidth: imageWidth,
          iouThreshold: iouThreshold ?? 0.4,
          confThreshold: confThreshold ?? 0.5,
          classThreshold: classThreshold ?? 0.5));
    } catch (e) {
      rethrow;
    }
  }

  Future<List<Map<String, dynamic>>> _yoloOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    required double iouThreshold,
    required double confThreshold,
    required double classThreshold,
  }) async {
    try {
      final x = await channel.invokeMethod<List<dynamic>>(
        'yoloOnFrame',
        {
          "bytesList": bytesList,
          "image_height": imageHeight,
          "image_width": imageWidth,
          "iou_threshold": iouThreshold,
          "conf_threshold": confThreshold,
          "class_threshold": classThreshold
        },
      );
      return x?.isNotEmpty ?? false
          ? x!.map((e) => Map<String, dynamic>.from(e)).toList()
          : [];
    } catch (e) {
      // print(e);
      rethrow;
    }
  }

  @override
  Future<List<Map<String, dynamic>>> yoloOnImage({
    required Uint8List bytesList,
    required int imageHeight,
    required int imageWidth,
    double? iouThreshold,
    double? confThreshold,
    double? classThreshold,
  }) async {
    try {
      return await _yoloOnImage(
          bytesList: bytesList,
          imageHeight: imageHeight,
          imageWidth: imageWidth,
          iouThreshold: iouThreshold ?? 0.4,
          confThreshold: confThreshold ?? 0.5,
          classThreshold: classThreshold ?? 0.5);
    } catch (e) {
      rethrow;
    }
  }

  Future<List<Map<String, dynamic>>> _yoloOnImage({
    required Uint8List bytesList,
    required int imageHeight,
    required int imageWidth,
    required double iouThreshold,
    required double confThreshold,
    required double classThreshold,
  }) async {
    try {
      final x = await channel.invokeMethod<List<dynamic>>(
        'yoloOnImage',
        {
          "bytesList": bytesList,
          "image_height": imageHeight,
          "image_width": imageWidth,
          "iou_threshold": iouThreshold,
          "conf_threshold": confThreshold,
          "class_threshold": classThreshold
        },
      );
      return x?.isNotEmpty ?? false
          ? x!.map((e) => Map<String, dynamic>.from(e)).toList()
          : [];
    } catch (e) {
      rethrow;
    }
  }

  @override
  Future<void> loadTesseractModel(
      {String? language, Map<String, String>? args}) async {
    try {
      final String testData = await loadTessData();
      await channel.invokeMethod<String?>('loadTesseractModel',
          {'tess_data': testData, 'arg': args, 'language': language ?? 'eng'});
    } catch (e) {
      rethrow;
    }
  }

  @override
  Future<List<Map<String, dynamic>>> tesseractOnImage({
    required Uint8List bytesList,
  }) async {
    try {
      return await _tesseractOnImage(bytesList: bytesList);
    } catch (e) {
      rethrow;
    }
  }

  Future<List<Map<String, dynamic>>> _tesseractOnImage(
      {required Uint8List bytesList}) async {
    try {
      final x = await channel.invokeMethod<dynamic>(
        'tesseractOnImage',
        {
          "bytesList": bytesList,
        },
      );
      return x == null ? [] : [Map<String, dynamic>.from(x)];
    } catch (e) {
      rethrow;
    }
  }
}
