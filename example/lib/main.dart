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
  late List<Map<String, dynamic>> modelResults;
  bool isLoaded = false;
  bool isDetecting = false;
  @override
  void initState() {
    super.initState();
    controller = CameraController(cameras[0], ResolutionPreset.max);
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
              modelResults = [];
            });
          });
          break;
        case Models.yolov5:
          loadYoloModel().then((value) {
            setState(() {
              isLoaded = true;
              isDetecting = false;
              modelResults = [];
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
      body: modelResults.isEmpty
          ? Stack(
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
                Positioned(
                  bottom: 75,
                  width: MediaQuery.of(context).size.width,
                  child: Container(
                    height: 80,
                    width: 80,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                          width: 5,
                          color: Colors.white,
                          style: BorderStyle.solid),
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
            )
          : ListView.builder(
              shrinkWrap: true,
              itemCount: modelResults.length,
              itemBuilder: (context, index) {
                Map<String, dynamic> result = modelResults[index];
                return Card(
                  child: Column(
                    children: [
                      Text(
                        result['tag'].toString().toUpperCase(),
                        style: const TextStyle(
                            fontSize: 14, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 6),
                      Image.memory(result['image'] as Uint8List),
                      const SizedBox(height: 6),
                      if (widget.model == Models.ocr)
                        Visibility(
                          visible: (result['text'] as String) != "None",
                          child: Column(
                            children: [
                              const Text(
                                "Image as text:",
                                style: TextStyle(
                                    fontSize: 12, fontWeight: FontWeight.bold),
                              ),
                              Text((result['text'] as String))
                            ],
                          ),
                        ),
                    ],
                  ),
                );
              },
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
    final responseHandler = await vision.loadOcrModel(
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
    if (responseHandler.type != 'success') {
      setState(() {
        isLoaded = true;
      });
    }
  }

  Future<void> loadYoloModel() async {
    final responseHandler = await vision.loadYoloModel(
        labels: 'assets/labels.txt',
        modelPath: 'assets/best-fp16.tflite',
        numThreads: 1,
        useGpu: false);
    if (responseHandler.type != 'success') {
      setState(() {
        isLoaded = true;
      });
    }
  }

  Future<void> startImageStream() async {
    if (!controller.value.isInitialized) {
      print('controller not initialized');
      return;
    }
    await controller.startImageStream((image) async {
      if (isDetecting) {
        return;
      }
      setState(() {
        isDetecting = true;
      });
      switch (widget.model) {
        case Models.ocr:
          await ocrOnFrame(image);
          break;
        case Models.yolov5:
          await yoloOnFrame(image);
          break;
        default:
          break;
      }
    });
  }

  Future<void> stopImageStream() async {
    if (!controller.value.isInitialized) {
      print('controller not initialized');
      return;
    }
    if (controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
    setState(() {
      isDetecting = false;
      modelResults.clear();
    });
  }

  Future<void> ocrOnFrame(CameraImage cameraImage) async {
    final result = await vision.ocrOnFrame(
        bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        classIsText: [0],
        iouThreshold: 0.6,
        confThreshold: 0.6);
    setState(() {
      modelResults = result.data as List<Map<String, dynamic>>;
    });
  }

  Future<void> yoloOnFrame(CameraImage cameraImage) async {
    final result = await vision.yoloOnFrame(
        bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        iouThreshold: 0.6,
        confThreshold: 0.6);
    setState(() {
      modelResults = result.data as List<Map<String, dynamic>>;
    });
  }
}
