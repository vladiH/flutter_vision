import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_vision/src/plugin/android.dart';

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

  // ///loadOcrModel: loads both YOLOv5 and Tesseract4 model from the assets folder and
  // ///return a ResponseHandler object.
  // ///
  // ///if the load is successful, it returns a ResponseHandler as a success object,
  // ///otherwise it returns a ResponseHandler as an error object
  // ///```json:{
  // /// "type": "success" or "error",
  // /// "message": "ok",
  // /// "data": {}```
  // ///
  // /// args: [modelPath] - path to the model file
  // /// ,[labelsPath] - path to the labels file
  // /// ,[numThreads] - number of threads to use for inference
  // /// ,[useGPU] - use GPU for inference
  // /// ,[language] - language for tesseract4(en,spa,de,fr,it,nl,ru,pt,tr,zh)
  // /// ,[tesseract4Config] - tesseract4 config
  // Future<void> loadOcrModel(
  //     {required String modelPath,
  //     required String labels,
  //     int? numThreads,
  //     bool? useGpu,
  //     String? language,
  //     Map<String, String>? args});

  // ///scanOnFrame accept a byte List as input and
  // ///return a ResponseHandler object.
  // ///
  // ///if scanOnFrame run without error, it returns a ResponseHandler as a success object,
  // ///otherwise it returns a ResponseHandler as an error object.
  // ///
  // ///```json:{
  // ///  "type": 'success',
  // ///  "message": "ok",
  // ///  "data": List<Map<String, dynamic>>
  // /// }```
  // ///where map is mapped as follows:
  // ///
  // ///```Map<String, dynamic>:{
  // ///    "confidence": double,
  // ///    "box": {x1:double, y1:double, x2:double, y2:double},
  // ///    "text": String,
  // ///    "image": Uint8List,
  // ///    "tag": String
  // /// }```
  // ///
  // ///args: [bytesList] - image as byte list
  // ///, [imageHeight] - image height
  // ///, [imageWidth] - image width
  // ///, [classIsText] - list of classes to be detected as text
  // ///, [iouThreshold] - intersection over union threshold
  // ///, [confThreshold] - confidence threshold
  // Future<List<Map<String, dynamic>>> ocrOnFrame({
  //   required List<Uint8List> bytesList,
  //   required int imageHeight,
  //   required int imageWidth,
  //   required List<int> classIsText,
  //   double? iouThreshold,
  //   double? confThreshold,
  // });

  // /// dispose OCRModel, clean and save resources
  // Future<void> closeOcrModel();

  ///loadYoloModel: load YOLOv5 model from the assets folder
  ///
  /// args: [modelPath] - path to the model file
  /// ,[labelsPath] - path to the labels file
  /// ,[modelVersion] - yolov5, yolov8
  /// ,[quantization] - When set to true, quantized models are used, which can result in faster execution, reduced memory usage, and slightly lower accuracy.
  /// ,[numThreads] - number of threads to use for inference
  /// ,[useGPU] - use GPU for inference
  Future<void> loadYoloModel(
      {required String modelPath,
      required String labels,
      required String modelVersion,
      bool? quantization,
      int? numThreads,
      bool? useGpu});

  ///yoloOnFrame accept a byte List as input and
  ///return a List<Map<String, dynamic>>.
  ///
  ///where map is mapped as follow:
  ///
  ///```Map<String, dynamic>:{
  ///    "box": [x1:left, y1:top, x2:right, y2:bottom, class_confidence]
  ///    "tag": String: detected class
  /// }```
  ///
  ///args: [bytesList] - image as byte list
  ///, [imageHeight] - image height
  ///, [imageWidth] - image width
  ///, [iouThreshold] - intersection over union threshold, default 0.4
  ///, [confThreshold] - model confidence threshold, default 0.5, only for [yolov5]
  ///, [classThreshold] - class confidence threshold, default 0.5
  Future<List<Map<String, dynamic>>> yoloOnFrame({
    required List<Uint8List> bytesList,
    required int imageHeight,
    required int imageWidth,
    double? iouThreshold,
    double? confThreshold,
    double? classThreshold,
  });

  ///yoloOnImage accept a Uint8List as input and
  ///return a List<Map<String, dynamic>>.
  ///
  ///where map is mapped as follows:
  ///
  ///```Map<String, dynamic>:{
  ///    "box": [x1:left, y1:top, x2:right, y2:bottom, class_confidence]
  ///    "tag": String: detected class
  /// }```
  ///
  ///args: [bytesList] - image bytes
  ///, [imageHeight] - image height
  ///, [imageWidth] - image width
  ///, [iouThreshold] - intersection over union threshold, default 0.4
  ///, [confThreshold] - model confidence threshold, default 0.5, only for [yolov5]
  ///, [classThreshold] - class confidence threshold, default 0.5
  Future<List<Map<String, dynamic>>> yoloOnImage({
    required Uint8List bytesList,
    required int imageHeight,
    required int imageWidth,
    double? iouThreshold,
    double? confThreshold,
    double? classThreshold,
  });

  /// dispose OCRModel, clean and save resources
  Future<void> closeYoloModel();

  ///loadTesseractModel: load Tesseract5 model from the assets folder.
  ///
  /// ,[language] - language for tesseract4(en,spa,de,fr,it,nl,ru,pt,tr,zh)
  /// ,[tesseract4Config] - tesseract4 config
  Future<void> loadTesseractModel(
      {String? language, Map<String, String>? args});

  ///tesseractOnImage accept a byte as input and
  ///return a List<Map<String, dynamic>.
  ///
  ///where map is mapped as follows:
  ///
  ///```Map<String, dynamic>:{
  ///    "text": String
  ///    "word_conf": List:int
  ///    "mean_conf": int
  /// }```
  ///
  ///args: [bytesList] - image as byte
  Future<List<Map<String, dynamic>>> tesseractOnImage(
      {required Uint8List bytesList});

  /// dispose Tesseract model, clean and save resources
  Future<void> closeTesseractModel();
}
