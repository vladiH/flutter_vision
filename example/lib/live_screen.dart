import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_vision/flutter_vision.dart';

import 'detection_painter.dart';
import 'model_catalog.dart';

/// Runs the model on the camera stream.
class LiveScreen extends StatefulWidget {
  const LiveScreen({super.key, required this.model, required this.accelerator});

  final ModelOption model;
  final Accelerator accelerator;

  @override
  State<LiveScreen> createState() => _LiveScreenState();
}

class _LiveScreenState extends State<LiveScreen> with WidgetsBindingObserver {
  List<CameraDescription> _cameras = const [];
  int _cameraIndex = 0;
  CameraController? _controller;
  YoloModel? _yolo;
  YoloResult? _result;
  String? _error;

  int _frames = 0;
  final Stopwatch _fpsWatch = Stopwatch();
  double _fps = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _init();
  }

  Future<void> _init() async {
    try {
      final yolo = await loadModel(widget.model, widget.accelerator);
      if (!mounted) {
        await yolo.dispose();
        return;
      }
      _yolo = yolo;
      _cameras = await availableCameras();
      if (_cameras.isEmpty) throw StateError('No camera available');
      await _startCamera();
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  CameraDescription get _camera => _cameras[_cameraIndex];

  Future<void> _startCamera() async {
    final controller = CameraController(
      _camera,
      ResolutionPreset.medium,
      enableAudio: false,
      // NV21 on Android, BGRA on iOS: both are read without extra copies.
      imageFormatGroup:
          Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
    );
    _controller = controller;
    await controller.initialize();
    if (!mounted) return;
    _fpsWatch
      ..reset()
      ..start();
    _frames = 0;
    await controller.startImageStream(_onFrame);
    setState(() {});
  }

  Future<void> _stopCamera() async {
    final controller = _controller;
    _controller = null;
    if (controller == null) return;
    if (controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
    await controller.dispose();
  }

  void _onFrame(CameraImage image) {
    final yolo = _yolo;
    final controller = _controller;
    // Skip frames while the previous one is still being processed.
    if (yolo == null || controller == null || yolo.isBusy) return;
    final frame = _toFrame(image, controller);
    if (frame == null) return;
    yolo.detectFrame(frame).then((result) {
      if (result == null || !mounted) return;
      _frames++;
      final seconds = _fpsWatch.elapsedMilliseconds / 1000;
      if (seconds >= 1) {
        _fps = _frames / seconds;
        _frames = 0;
        _fpsWatch
          ..reset()
          ..start();
      }
      setState(() => _result = result);
    }).catchError((Object e) {
      if (mounted) setState(() => _error = '$e');
    });
  }

  CameraFrame? _toFrame(CameraImage image, CameraController controller) {
    final FrameFormat format;
    switch (image.format.group) {
      case ImageFormatGroup.nv21:
        format = FrameFormat.nv21;
      case ImageFormatGroup.yuv420:
        format = FrameFormat.yuv420;
      case ImageFormatGroup.bgra8888:
        format = FrameFormat.bgra8888;
      default:
        return null;
    }
    final front = controller.description.lensDirection == CameraLensDirection.front;
    // iOS already delivers upright (and mirrored front) frames. Android
    // delivers them in the sensor orientation.
    final rotation = Platform.isAndroid
        ? CameraFrame.rotationFor(
            sensorOrientation: controller.description.sensorOrientation,
            deviceOrientation:
                CameraFrame.degreesFor(controller.value.deviceOrientation.name),
            frontCamera: front,
          )
        : 0;
    return CameraFrame(
      width: image.width,
      height: image.height,
      format: format,
      planes: [
        for (final plane in image.planes)
          FramePlane(
            bytes: plane.bytes,
            bytesPerRow: plane.bytesPerRow,
            bytesPerPixel: plane.bytesPerPixel ?? 1,
          ),
      ],
      rotation: rotation,
    );
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2) return;
    await _stopCamera();
    setState(() {
      _cameraIndex = (_cameraIndex + 1) % _cameras.length;
      _result = null;
    });
    await _startCamera();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_yolo == null || _cameras.isEmpty) return;
    if (state == AppLifecycleState.inactive) {
      _stopCamera();
    } else if (state == AppLifecycleState.resumed && _controller == null) {
      _startCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopCamera().whenComplete(() => _yolo?.dispose());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    final result = _result;
    final ready = controller != null && controller.value.isInitialized;
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(title: Text(widget.model.name)),
      floatingActionButton: _cameras.length > 1
          ? FloatingActionButton(
              onPressed: _switchCamera,
              child: const Icon(Icons.cameraswitch),
            )
          : null,
      body: _error != null
          ? Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(_error!, style: const TextStyle(color: Colors.white)),
              ),
            )
          : !ready
              ? const Center(child: CircularProgressIndicator())
              : Stack(
                  fit: StackFit.expand,
                  children: [
                    _Preview(controller: controller),
                    if (result != null)
                      CustomPaint(
                        painter: DetectionPainter(
                          result: result,
                          fit: BoxFit.cover,
                          mirror: Platform.isAndroid &&
                              controller.description.lensDirection ==
                                  CameraLensDirection.front,
                        ),
                      ),
                    if (result != null)
                      Positioned(
                        left: 12,
                        top: 12,
                        child: DecoratedBox(
                          decoration: BoxDecoration(
                            color: Colors.black54,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Padding(
                            padding: const EdgeInsets.all(8),
                            child: Text(
                              '${_fps.toStringAsFixed(1)} FPS · '
                              '${_yolo!.accelerator.name.toUpperCase()}\n'
                              'inference ${result.inferenceTime.inMilliseconds} ms · '
                              'objects ${result.detections.length}',
                              style: const TextStyle(color: Colors.white),
                            ),
                          ),
                        ),
                      ),
                  ],
                ),
    );
  }
}

/// Camera preview filling the screen (cropped, like BoxFit.cover).
class _Preview extends StatelessWidget {
  const _Preview({required this.controller});

  final CameraController controller;

  @override
  Widget build(BuildContext context) {
    final previewSize = controller.value.previewSize;
    if (previewSize == null) return CameraPreview(controller);
    final portrait =
        MediaQuery.orientationOf(context) == Orientation.portrait;
    final width = portrait ? previewSize.height : previewSize.width;
    final height = portrait ? previewSize.width : previewSize.height;
    return ClipRect(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: width,
          height: height,
          child: CameraPreview(controller),
        ),
      ),
    );
  }
}
