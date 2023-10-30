import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

abstract class BaseFlutterVision {
  // ignore: constant_identifier_names
  static const String TESS_DATA_CONFIG = 'assets/tessdata_config.json';
  // ignore: constant_identifier_names
  static const String TESS_DATA_PATH = 'assets/tessdata';
  static const MethodChannel _channel = MethodChannel('flutter_vision');
  MethodChannel get channel => _channel;

  Future<String> loadTessData() async {
    try {
      final Directory appDirectory = await getApplicationDocumentsDirectory();
      final String tessdataDirectory = join(appDirectory.path, 'tessdata');

      if (!await Directory(tessdataDirectory).exists()) {
        await Directory(tessdataDirectory).create();
      }
      await copyTessDataToAppDocumentsDirectory(tessdataDirectory);
      return appDirectory.path;
    } catch (e) {
      rethrow;
    }
  }

  Future copyTessDataToAppDocumentsDirectory(String tessdataDirectory) async {
    final String config = await rootBundle.loadString(TESS_DATA_CONFIG);
    Map<String, dynamic> files = jsonDecode(config);
    for (var file in files["files"]) {
      if (!await File('$tessdataDirectory/$file').exists()) {
        final ByteData data = await rootBundle.load('$TESS_DATA_PATH/$file');
        final Uint8List bytes = data.buffer.asUint8List(
          data.offsetInBytes,
          data.lengthInBytes,
        );
        await File('$tessdataDirectory/$file').writeAsBytes(bytes);
      }
    }
  }

  // Future<void> loadOcrModel(
  //     {required String modelPath,
  //     required String labels,
  //     int? numThreads,
  //     bool? useGpu,
  //     String? language,
  //     Map<String, String>? args});

  // Future<List<Map<String, dynamic>>> ocrOnFrame({
  //   required List<Uint8List> bytesList,
  //   required int imageHeight,
  //   required int imageWidth,
  //   required List<int> classIsText,
  //   double? iouThreshold,
  //   double? confThreshold,
  // });

  // Future<void> closeOcrModel() async {
  //   await channel.invokeMethod('closeOcrModel');
  // }

  Future<void> loadYoloModel({
    required String modelPath,
    required String labels,
    required String modelVersion,
    bool? quantization,
    int? numThreads,
    bool? useGpu,
  });

  Future<List<Map<String, dynamic>>> yoloOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    double? iouThreshold,
    double? confThreshold,
    double? classThreshold,
  });

  Future<List<Map<String, dynamic>>> yoloOnImage({
    required Uint8List bytesList,
    required int imageHeight,
    required int imageWidth,
    double? iouThreshold,
    double? confThreshold,
    double? classThreshold,
  });

  Future<void> closeYoloModel() async {
    await channel.invokeMethod('closeYoloModel');
  }

  Future<void> loadTesseractModel(
      {String? language, Map<String, String>? args});

  Future<List<Map<String, dynamic>>> tesseractOnImage(
      {required Uint8List bytesList});

  Future<void> closeTesseractModel() async {
    await channel.invokeMethod('closeTesseractModel');
  }
}
