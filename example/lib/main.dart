import 'dart:io';
import 'dart:typed_data';
import 'dart:ui';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'dart:async';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:image_picker/image_picker.dart';

enum Models { yolo, ocr }

late List<CameraDescription> cameras;
main() async {
  WidgetsFlutterBinding.ensureInitialized();
  DartPluginRegistrant.ensureInitialized();
  cameras = await availableCameras();
  runApp(
    const MaterialApp(
      home: MyApp(model: Models.yolo),
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
  File? imageFile;
  bool isLoaded = false;
  bool isDetecting = false;

  @override
  void initState() {
    super.initState();
    vision = FlutterVision();
    switch (widget.model) {
      case Models.ocr:
        // controller = CameraController(cameras[0], ResolutionPreset.low);
        // controller.initialize();
        loadOcrModel().then((value) {
          setState(() {
            isLoaded = true;
            isDetecting = false;
            yoloResults = [];
          });
        });
        break;
      case Models.yolo:
        // controller = CameraController(cameras[0], ResolutionPreset.low);
        // controller.initialize();
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
          imageFile != null
              ? Image.file(imageFile!)
              : TextButton(
                  onPressed: pickImage, child: const Text("Pick image")),
          // AspectRatio(
          //   aspectRatio: controller.value.aspectRatio,
          //   child: CameraPreview(
          //     controller,
          //   ),
          // ),
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
                        stopDetection();
                      },
                      icon: const Icon(
                        Icons.stop,
                        color: Colors.red,
                      ),
                      iconSize: 50,
                    )
                  : IconButton(
                      onPressed: () async {
                        await startDetection();
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
          await yoloOnImage();
        },
        child: const Icon(Icons.photo),
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
        modelVersion: "yolov5",
        numThreads: 1,
        useGpu: false);
    setState(() {
      isLoaded = true;
    });
  }

  Future<void> startDetection() async {
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
        case Models.yolo:
          yoloOnFrame(image);
          break;
        default:
          break;
      }
    });
  }

  Future<void> stopDetection() async {
    setState(() {
      yoloResults.clear();
      isDetecting = false;
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
    final result = await vision.yoloOnFrame(
        bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        iouThreshold: 0.4,
        confThreshold: 0.8,
        classThreshold: 0.5);
    if (result.isNotEmpty) {
      setState(() {
        yoloResults = result;
      });
    }
  }

  Future<void> pickImage() async {
    final ImagePicker picker = ImagePicker();
    // Capture a photo
    final XFile? photo = await picker.pickImage(source: ImageSource.gallery);
    if (photo != null) {
      setState(() {
        imageFile = File(photo.path);
      });
    }
  }

  yoloOnImage() async {
    Uint8List byte = await imageFile!.readAsBytes();
    final image = await decodeImageFromList(byte);
    final result = await vision.yoloOnImage(
        bytesList: byte,
        imageHeight: image.height,
        imageWidth: image.width,
        iouThreshold: 0.8,
        confThreshold: 0.8,
        classThreshold: 0.7);
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

    Color colorPick = const Color.fromARGB(255, 50, 233, 30);

    return yoloResults.map((result) {
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
