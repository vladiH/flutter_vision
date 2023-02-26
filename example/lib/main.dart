import 'dart:typed_data';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'dart:async';
import 'package:flutter_vision/flutter_vision.dart';

enum Models { yolov5, ocr }

late List<CameraDescription> cameras;
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  DartPluginRegistrant.ensureInitialized();
  cameras = await availableCameras();
  runApp(
    const MaterialApp(
      home: MyApp(model: Models.yolov5),
    ),
  );
}

class MyApp extends StatefulWidget {
  const MyApp({required this.model, Key? key}) : super(key: key);
  final Models model;

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late CameraController controller;
  late FlutterVision vision;
  late List<Map<String, dynamic>> yoloResults;
  CameraImage? cameraImage;
  bool isLoaded = false;
  bool isDetecting = false;
  @override
  void initState() {
    super.initState();
    controller = CameraController(cameras[0], ResolutionPreset.low);
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      vision = FlutterVision();
      switch (widget.model) {
        case Models.ocr:
          loadOcrModel().then((value) {
            setState(() {
              isLoaded = true;
              isDetecting = false;
              yoloResults = [];
            });
          });
          break;
        case Models.yolov5:
          loadYoloModel().then((value) {
            setState(() {
              isLoaded = true;
              isDetecting = false;
              yoloResults = [];
            });
          });
          break;
        default:
          break;
      }
    });
  }

  @override
  void dispose() async {
    controller.dispose();
    await vision.closeOcrModel();
    await vision.closeYoloModel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final Size size = MediaQuery.of(context).size;
    if (!isLoaded) {
      return const Scaffold(
        body: Center(
          child: Text("Model not loaded, waiting for it"),
        ),
      );
    }
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          AspectRatio(
            aspectRatio: controller.value.aspectRatio,
            child: CameraPreview(
              controller,
            ),
          ),
          ColorFiltered(
            colorFilter: ColorFilter.mode(
                Colors.black.withOpacity(0.7), BlendMode.srcOut),
            child: Stack(
              children: [
                Container(
                  decoration: const BoxDecoration(
                      color: Colors.black,
                      backgroundBlendMode: BlendMode.dstOut),
                ),
                Align(
                  alignment: Alignment.topCenter,
                  child: Container(
                    margin: EdgeInsets.only(top: size.height * 0.2),
                    height: size.height * 0.30,
                    width: size.width * 0.9,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(10),
                    ),
                  ),
                ),
              ],
            ),
          ),
          ...displayBoxesAroundRecognizedObjects(size),
          Positioned(
            bottom: 75,
            width: MediaQuery.of(context).size.width,
            child: Container(
              height: 80,
              width: 80,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                    width: 5, color: Colors.white, style: BorderStyle.solid),
              ),
              child: isDetecting
                  ? IconButton(
                      onPressed: () async {
                        stopImageStream();
                      },
                      icon: const Icon(
                        Icons.stop,
                        color: Colors.red,
                      ),
                      iconSize: 50,
                    )
                  : IconButton(
                      onPressed: () async {
                        await startImageStream();
                      },
                      icon: const Icon(
                        Icons.play_arrow,
                        color: Colors.white,
                      ),
                      iconSize: 50,
                    ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          await stopImageStream();
        },
        child: const Icon(Icons.restart_alt_rounded),
      ),
    );
  }

  Future<void> loadOcrModel() async {
    await vision.loadOcrModel(
        labels: 'assets/labels.txt',
        modelPath: 'assets/best-fp16.tflite',
        args: {
          'psm': '11',
          'oem': '1',
          'preserve_interword_spaces': '1',
        },
        language: 'spa',
        numThreads: 1,
        useGpu: false);
    setState(() {
      isLoaded = true;
    });
  }

  Future<void> loadYoloModel() async {
    await vision.loadYoloModel(
        labels: 'assets/labelss.txt',
        modelPath: 'assets/yolov5s.tflite',
        numThreads: 1,
        useGpu: false);
    setState(() {
      isLoaded = true;
    });
  }

  Future<void> startImageStream() async {
    if (!controller.value.isInitialized) {
      return;
    }
    setState(() {
      isDetecting = true;
    });
    await controller.startImageStream((image) async {
      // if (isDetecting) {
      //   return;
      // }
      cameraImage = image;
      switch (widget.model) {
        case Models.ocr:
          ocrOnFrame(image);
          break;
        case Models.yolov5:
          yoloOnFrame(image);
          break;
        default:
          break;
      }
    });
  }

  Future<void> stopImageStream() async {
    if (!controller.value.isInitialized) {
      return;
    }
    if (controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
    setState(() {
      yoloResults.clear();
    });
  }

  ocrOnFrame(CameraImage cameraImage) async {
    final result = await vision.ocrOnFrame(
        bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        classIsText: [0],
        iouThreshold: 0.6,
        confThreshold: 0.8);
    setState(() {
      yoloResults = result;
    });
  }

  yoloOnFrame(CameraImage cameraImage) async {
    print("88888888888888888888");
    print(cameraImage.height);
    print(cameraImage.width);
    final result = await vision.yoloOnFrame(
        bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        iouThreshold: 0.6,
        confThreshold: 0.8);
    if (result.isNotEmpty) {
      setState(() {
        yoloResults = result;
      });
    }
  }

  List<Widget> displayBoxesAroundRecognizedObjects(Size screen) {
    if (yoloResults.isEmpty) return [];

    double factorX = screen.width / (cameraImage?.height ?? 1);
    double factorY = screen.height / (cameraImage?.width ?? 1);

    Color colorPick = Color.fromARGB(255, 50, 233, 30);

    return yoloResults.map((result) {
      print(result["box"][0] * factorX);
      print(result["box"][1] * factorY);
      print(result["box"][2] * factorX);
      print(result["box"][3] * factorY);
      return Positioned(
        left: result["box"][0] * factorX,
        top: result["box"][1] * factorY,
        right: result["box"][2] * factorX,
        bottom: result["box"][3] * factorY,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: const BorderRadius.all(Radius.circular(10.0)),
            border: Border.all(color: Colors.pink, width: 2.0),
          ),
          child: Text(
            "${result['tag']} ${(result['box'][4] * 100).toStringAsFixed(0)}%",
            style: TextStyle(
              background: Paint()..color = colorPick,
              color: Colors.white,
              fontSize: 18.0,
            ),
          ),
        ),
      );
    }).toList();
  }
}
