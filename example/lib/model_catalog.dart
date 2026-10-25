import 'package:flutter_vision/flutter_vision.dart';

/// Models bundled with the example (see `assets/` and `pubspec.yaml`).
class ModelOption {
  const ModelOption(this.name, this.path, this.task);

  final String name;
  final String path;
  final YoloTask task;
}

const List<ModelOption> bundledModels = [
  ModelOption('YOLO11n', 'assets/yolov11n.tflite', YoloTask.detect),
  ModelOption('YOLOv8n', 'assets/yolov8n.tflite', YoloTask.detect),
  ModelOption('YOLOv5n', 'assets/yolov5n.tflite', YoloTask.detect),
  ModelOption('YOLOv8n-seg', 'assets/yolov8n-seg.tflite', YoloTask.segment),
];

const String labelsAsset = 'assets/labels.txt';

/// Loads [option] with [accelerator] (falls back to CPU automatically).
Future<YoloModel> loadModel(ModelOption option, Accelerator accelerator) {
  return YoloModel.load(
    modelPath: option.path,
    labelsPath: labelsAsset,
    task: option.task,
    accelerator: accelerator,
  );
}
