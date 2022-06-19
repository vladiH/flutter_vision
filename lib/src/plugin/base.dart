import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:path/path.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import '../utils/response_handler.dart';

abstract class BaseFlutterVision {
  // ignore: constant_identifier_names
  static const String TESS_DATA_CONFIG = 'assets/tessdata_config.json';
  // ignore: constant_identifier_names
  static const String TESS_DATA_PATH = 'assets/tessdata';
  static const MethodChannel _channel = MethodChannel('flutter_vision');
  MethodChannel get channel => _channel;

  Future<ResponseHandler> loadOcrModel(
      {required String modelPath,
      required String labels,
      int? numThreads,
      bool? useGpu,
      String? language,
      Map<String, String>? args});

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

  Future<ResponseHandler> ocrOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    required List<int> classIsText,
    double? iouThreshold,
    double? confThreshold,
  });

  Future<void> closeOcrModel() async {
    await channel.invokeMethod('closeOcrModel');
  }

  Future<ResponseHandler> loadYoloModel({
    required String modelPath,
    required String labels,
    int? numThreads,
    bool? useGpu,
  });

  Future<ResponseHandler> yoloOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    double? iouThreshold,
    double? confThreshold,
  });

  Future<void> closeYoloModel() async {
    await channel.invokeMethod('closeYoloModel');
  }
}
