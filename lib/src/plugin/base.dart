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
  bool _isLoadedOcrModel = false;

  Future<ResponseHandler> loadOcrModel(
      {required String modelPath,
      required String labels,
      int? numThreads,
      bool? useGpu,
      String? language,
      Map<String, String>? args}) async {
    try {
      if (isLoadedOcrModel) {
        return Success(message: 'Model is already loaded');
      }
      final String testData = await _loadTessData();
      final result = await _channel.invokeMethod<String?>('loadOcrModel', {
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
          _isLoadedOcrModel = true;
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

  static Future<String> _loadTessData() async {
    try {
      final Directory appDirectory = await getApplicationDocumentsDirectory();
      final String tessdataDirectory = join(appDirectory.path, 'tessdata');

      if (!await Directory(tessdataDirectory).exists()) {
        await Directory(tessdataDirectory).create();
      }
      await _copyTessDataToAppDocumentsDirectory(tessdataDirectory);
      return appDirectory.path;
    } catch (e) {
      rethrow;
    }
  }

  static Future _copyTessDataToAppDocumentsDirectory(
      String tessdataDirectory) async {
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

  MethodChannel get channel => _channel;
  bool get isLoadedOcrModel => _isLoadedOcrModel;
  Future<void> closeOcrModel() async {
    if (_isLoadedOcrModel) {
      await channel.invokeMethod('closeOcrModel');
      _isLoadedOcrModel = false;
    }
  }
}
