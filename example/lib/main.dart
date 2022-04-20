import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'dart:async';
import 'package:flutter_vision/flutter_vision.dart';

late List<CameraDescription> cameras;
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(
    const MaterialApp(
      home: MyApp(),
    ),
  );
}

class MyApp extends StatefulWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late CameraController controller;
  late FlutterVision vision;
  late List<Map<String, dynamic>> ocrResults;
  bool isLoaded = false;
  bool isDetecting = false;
  bool isPredicted = false;
  @override
  void initState() {
    super.initState();
    controller = CameraController(cameras[0], ResolutionPreset.max);
    controller.initialize().then((_) {
      if (!mounted) {
        return;
      }
      vision = FlutterVision();
      loadModel().then((value) {
        setState(() {
          isLoaded = true;
          isDetecting = false;
          isPredicted = false;
          ocrResults = [];
        });
      });
    });
  }

  @override
  void dispose() async {
    controller.dispose();
    await vision.closeOcrModel();
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
      body: !isPredicted
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
                              if (controller.value.isStreamingImages) {
                                await controller.stopImageStream();
                              }
                              setState(() {
                                isDetecting = false;
                              });
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
              itemCount: ocrResults.length,
              itemBuilder: (context, index) {
                Map<String, dynamic> result = ocrResults[index];
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
        onPressed: () {
          setState(() {
            isDetecting = false;
            isPredicted = false;
          });
        },
        child: const Icon(Icons.restart_alt_rounded),
      ),
    );
  }

  Future<void> loadModel() async {
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

  Future<void> startImageStream() async {
    if (controller.value.isInitialized) {
      if (controller.value.isStreamingImages) {
        await controller.stopImageStream();
      }
      await controller.startImageStream((image) async {
        if (!isDetecting) {
          setState(() {
            isDetecting = true;
          });
          await ocrOnFrame(image);
        }
      });
    } else {
      setState(() {
        isDetecting = false;
      });
    }
  }

  Future<void> ocrOnFrame(CameraImage cameraImage) async {
    await controller.stopImageStream();
    final result = await vision.ocrOnFrame(
        bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        classIsText: [0],
        iouThreshold: 0.6,
        confThreshold: 0.6);
    setState(() {
      isPredicted = true;
      ocrResults = result.data as List<Map<String, dynamic>>;
    });
  }
}
