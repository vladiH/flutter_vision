import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:image_picker/image_picker.dart';

import 'detection_painter.dart';
import 'model_catalog.dart';

/// Picks a photo and runs the model on it.
class ImageScreen extends StatefulWidget {
  const ImageScreen({super.key, required this.model, required this.accelerator});

  final ModelOption model;
  final Accelerator accelerator;

  @override
  State<ImageScreen> createState() => _ImageScreenState();
}

class _ImageScreenState extends State<ImageScreen> {
  YoloModel? _yolo;
  Uint8List? _bytes;
  YoloResult? _result;
  String? _error;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    loadModel(widget.model, widget.accelerator).then((model) {
      if (!mounted) {
        model.dispose();
        return;
      }
      setState(() => _yolo = model);
    }, onError: (Object e) {
      if (mounted) setState(() => _error = '$e');
    });
  }

  @override
  void dispose() {
    _yolo?.dispose();
    super.dispose();
  }

  Future<void> _pickAndDetect() async {
    final file = await ImagePicker().pickImage(source: ImageSource.gallery);
    if (file == null) return;
    final bytes = await file.readAsBytes();
    setState(() {
      _bytes = bytes;
      _result = null;
      _busy = true;
      _error = null;
    });
    try {
      final result = await _yolo!.detectImage(bytes);
      if (mounted) setState(() => _result = result);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final yolo = _yolo;
    final result = _result;
    return Scaffold(
      appBar: AppBar(title: Text(widget.model.name)),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: yolo == null || _busy ? null : _pickAndDetect,
        icon: const Icon(Icons.photo_library),
        label: const Text('Pick photo'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Center(
              child: _bytes == null
                  ? Text(yolo == null
                      ? (_error ?? 'Loading model...')
                      : 'Pick a photo to detect objects')
                  : Stack(
                      fit: StackFit.expand,
                      children: [
                        // Image.memory applies the EXIF orientation, like the
                        // plugin, so the boxes line up.
                        Image.memory(_bytes!, fit: BoxFit.contain),
                        if (result != null)
                          CustomPaint(
                            painter: DetectionPainter(
                                result: result, fit: BoxFit.contain),
                          ),
                        if (_busy)
                          const Center(child: CircularProgressIndicator()),
                      ],
                    ),
            ),
          ),
          if (_error != null && yolo != null)
            Padding(
              padding: const EdgeInsets.all(8),
              child: Text(_error!,
                  style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
          if (result != null && yolo != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 88),
              child: Text(
                '${result.detections.length} objects · '
                '${yolo.accelerator.name.toUpperCase()} · '
                'pre ${result.preprocessTime.inMilliseconds} ms · '
                'inference ${result.inferenceTime.inMilliseconds} ms · '
                'post ${result.postprocessTime.inMilliseconds} ms',
                textAlign: TextAlign.center,
              ),
            ),
        ],
      ),
    );
  }
}
