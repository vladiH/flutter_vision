import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_vision/src/plugin/android.dart';
import 'package:flutter_vision/src/utils/response_handler.dart';

abstract class FlutterVision {
  factory FlutterVision() {
    switch (Platform.operatingSystem) {
      case 'android':
        return AndroidFlutterVision();
      case 'ios':
        throw UnimplementedError('iOS is not supported for now');
      //return IosDniScanner();
      default:
        throw UnsupportedError('Unsupported platform');
    }
  }

  ///loadOcrModel: loads both YOLOv5 and Tesseract4 model from the assets folder and
  ///return a ResponseHandler object.
  ///
  ///if the load is successful, it returns a ResponseHandler as a success object,
  ///otherwise it returns a ResponseHandler as an error object
  ///```json:{
  /// "type": "success" or "error",
  /// "message": "ok",
  /// "data": {}```
  ///
  /// args: [modelPath] - path to the model file
  /// ,[labelsPath] - path to the labels file
  /// ,[numThreads] - number of threads to use for inference
  /// ,[useGPU] - use GPU for inference
  /// ,[language] - language for tesseract4(en,spa,de,fr,it,nl,ru,pt,tr,zh)
  /// ,[tesseract4Config] - tesseract4 config
  Future<ResponseHandler> loadOcrModel(
      {required String modelPath,
      required String labels,
      int? numThreads,
      bool? useGpu,
      String? language,
      Map<String, String>? args});

  ///scanOnFrame accept a byte List as input and
  ///return a ResponseHandler object.
  ///
  ///if scanOnFrame run without error, it returns a ResponseHandler as a success object,
  ///otherwise it returns a ResponseHandler as an error object.
  ///
  ///```json:{
  ///  "type": 'success',
  ///  "message": "ok",
  ///  "data": List<Map<String, dynamic>>
  /// }```
  ///where map is mapped as follows:
  ///
  ///```Map<String, dynamic>:{
  ///    "confidence": double,
  ///    "box": {x1:double, y1:double, x2:double, y2:double},
  ///    "text": String,
  ///    "image": Uint8List,
  ///    "tag": String
  /// }```
  ///
  ///args: [bytesList] - image as byte list
  ///, [imageHeight] - image height
  ///, [imageWidth] - image width
  ///, [classIsText] - list of classes to be detected as text
  ///, [iouThreshold] - intersection over union threshold
  ///, [confThreshold] - confidence threshold
  Future<ResponseHandler> ocrOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    required List<int> classIsText,
    double? iouThreshold,
    double? confThreshold,
  });

  /// dispose OCRModel, clean and save resources
  Future<void> closeOcrModel();

  ///loadYoloModel: load YOLOv5 model from the assets folder and
  ///return a ResponseHandler object.
  ///
  ///if the load is successful, it returns a ResponseHandler as a success object,
  ///otherwise it returns a ResponseHandler as an error object
  ///```json:{
  /// "type": "success" or "error",
  /// "message": "ok",
  /// "data": {}```
  ///
  /// args: [modelPath] - path to the model file
  /// ,[labelsPath] - path to the labels file
  /// ,[numThreads] - number of threads to use for inference
  /// ,[useGPU] - use GPU for inference
  Future<ResponseHandler> loadYoloModel(
      {required String modelPath,
      required String labels,
      int? numThreads,
      bool? useGpu});

  ///yoloOnFrame accept a byte List as input and
  ///return a ResponseHandler object.
  ///
  ///if yoloOnFrame run without error, it returns a ResponseHandler as a success object,
  ///otherwise it returns a ResponseHandler as an error object.
  ///
  ///```json:{
  ///  "type": 'success',
  ///  "message": "ok",
  ///  "data": List<Map<String, dynamic>>
  /// }```
  ///where map is mapped as follows:
  ///
  ///```Map<String, dynamic>:{
  ///    "confidence": double,
  ///    "box": {x1:double, y1:double, x2:double, y2:double},
  ///    "image": Uint8List,
  ///    "tag": String
  /// }```
  ///
  ///args: [bytesList] - image as byte list
  ///, [imageHeight] - image height
  ///, [imageWidth] - image width
  ///, [iouThreshold] - intersection over union threshold
  ///, [confThreshold] - confidence threshold
  Future<ResponseHandler> yoloOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    double? iouThreshold,
    double? confThreshold,
  });

  /// dispose OCRModel, clean and save resources
  Future<void> closeYoloModel();
}

///loadOcrModel: load custom YOLOv5 model from the assets folder,
///return a ErrorHandler object.
///
///If loads is successful, returns a Success object,
///if there is an error, returns an Error object
/*Future<ErrorHandler> loadYoloModel(
      {required String modelPath,
      required String labels,
      int numThreads,
      bool isAsset,
      bool useGpu});*/
