import 'dart:typed_data';
import 'package:flutter_vision/src/flutter_vision_base.dart';
import 'package:flutter_vision/src/plugin/base_flutter_vision.dart';

class AndroidFlutterVision extends BaseFlutterVision implements FlutterVision {
  @override
  Future<void> loadYoloModel({
    required String modelPath,
    required String labels,
    required String modelVersion,
    bool? quantization,
    int? numThreads,
    bool? useGpu,
    bool? isAsset,
    int? rotation,
  }) async {
    try {
      await channel.invokeMethod('loadYoloModel', {
        'modelPath': modelPath,
        'labels': labels,
        'modelVersion': modelVersion,
        'quantization': quantization ?? false,
        'numThreads': numThreads ?? 4,
        'useGpu': useGpu ?? false,
        'isAsset': isAsset ?? true,
        'rotation': rotation ?? 90,
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
      final List<dynamic> result = await channel.invokeMethod('yoloOnFrame', {
        'bytesList': bytesList,
        'imageHeight': imageHeight,
        'imageWidth': imageWidth,
        'iouThreshold': iouThreshold ?? 0.4,
        'confThreshold': confThreshold ?? 0.5,
        'classThreshold': classThreshold ?? 0.5,
      });

      return result.map((e) => Map<String, dynamic>.from(e)).toList();
    } catch (e) {
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
      final List<dynamic> result = await channel.invokeMethod('yoloOnImage', {
        'bytesList': bytesList,
        'imageHeight': imageHeight,
        'imageWidth': imageWidth,
        'iouThreshold': iouThreshold ?? 0.4,
        'confThreshold': confThreshold ?? 0.5,
        'classThreshold': classThreshold ?? 0.5,
      });

      return result.map((e) => Map<String, dynamic>.from(e)).toList();
    } catch (e) {
      rethrow;
    }
  }

  // @override
  // Future<void> loadTesseractModel({
  //   String? language,
  //   Map<String, String>? args,
  // }) async {
  //   try {
  //     final String tessDataPath = await loadTessData();

  //     await channel.invokeMethod('loadTesseractModel', {
  //       'language': language ?? 'eng',
  //       'tessDataPath': tessDataPath,
  //       'args': args ?? {},
  //     });
  //   } catch (e) {
  //     rethrow;
  //   }
  // }

  // @override
  // Future<List<Map<String, dynamic>>> tesseractOnImage({
  //   required Uint8List bytesList,
  // }) async {
  //   try {
  //     final List<dynamic> result = await channel.invokeMethod(
  //       'tesseractOnImage',
  //       {'bytesList': bytesList},
  //     );

  //     return result.map((e) => Map<String, dynamic>.from(e)).toList();
  //   } catch (e) {
  //     rethrow;
  //   }
  // }
}
