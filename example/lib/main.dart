import 'package:flutter/material.dart';
import 'package:flutter_vision/flutter_vision.dart';

import 'image_screen.dart';
import 'live_screen.dart';
import 'model_catalog.dart';

void main() {
  runApp(const FlutterVisionExample());
}

class FlutterVisionExample extends StatelessWidget {
  const FlutterVisionExample({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'flutter_vision',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      darkTheme: ThemeData(
        colorSchemeSeed: Colors.indigo,
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  ModelOption _model = bundledModels.first;
  Accelerator _accelerator = Accelerator.cpu;

  void _open(Widget screen) {
    Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => screen));
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('flutter_vision 3')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('Model', style: textTheme.titleMedium),
          const SizedBox(height: 8),
          RadioGroup<ModelOption>(
            groupValue: _model,
            onChanged: (value) => setState(() => _model = value!),
            child: Column(
              children: [
                for (final option in bundledModels)
                  RadioListTile<ModelOption>(
                    value: option,
                    title: Text(option.name),
                    subtitle: Text(option.task == YoloTask.segment
                        ? 'Segmentation'
                        : 'Detection'),
                  ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          Text('Hardware', style: textTheme.titleMedium),
          const SizedBox(height: 8),
          SegmentedButton<Accelerator>(
            segments: const [
              ButtonSegment(value: Accelerator.cpu, label: Text('CPU')),
              ButtonSegment(value: Accelerator.gpu, label: Text('GPU')),
              ButtonSegment(value: Accelerator.npu, label: Text('NPU')),
            ],
            selected: {_accelerator},
            onSelectionChanged: (value) =>
                setState(() => _accelerator = value.single),
          ),
          const SizedBox(height: 32),
          FilledButton.icon(
            icon: const Icon(Icons.videocam),
            label: const Text('Live camera'),
            onPressed: () =>
                _open(LiveScreen(model: _model, accelerator: _accelerator)),
          ),
          const SizedBox(height: 12),
          FilledButton.tonalIcon(
            icon: const Icon(Icons.photo),
            label: const Text('Photo from gallery'),
            onPressed: () =>
                _open(ImageScreen(model: _model, accelerator: _accelerator)),
          ),
        ],
      ),
    );
  }
}
