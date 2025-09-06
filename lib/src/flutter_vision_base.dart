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
      default:
        throw UnsupportedError('Unsupported platform');
    }
  }

  ///loadYoloModel: load YOLOv5 model from the assets folder
  ///
  /// args: [modelPath] - path to the model file
  /// ,[labelsPath] - path to the labels file
  /// ,[modelVersion] - yolov5, yolov8
  /// ,[quantization] - When set to true, quantized models are used, which can result in faster execution, reduced memory usage, and slightly lower accuracy.
  /// ,[numThreads] - number of threads to use for inference
  /// ,[isAsset] - set to true if the model is in the assets folder
  /// ,[useGPU] - use GPU for inference
  /// ,[rotation] - rotation to apply to the input image by defaut android use 90 degrees
  Future<void> loadYoloModel({
    required String modelPath,
    required String labels,
    required String modelVersion,
    bool? quantization,
    int? numThreads,
    bool? isAsset,
    bool? useGpu,
    int? rotation,
  });

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
}
