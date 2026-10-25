# Migrating from 2.x to 3.0

3.0 replaces the `FlutterVision` class with one `YoloModel` object per model.

## Loading

```dart
// 2.x
final vision = FlutterVision();
await vision.loadYoloModel(
  modelPath: 'assets/yolov8n.tflite',
  labels: 'assets/labels.txt',
  modelVersion: 'yolov8',
  useGpu: true,
  numThreads: 2,
);

// 3.0
final model = await YoloModel.load(
  modelPath: 'assets/yolov8n.tflite',
  labelsPath: 'assets/labels.txt',
  accelerator: Accelerator.gpu,
  numThreads: 2,
);
```

| 2.x | 3.0 |
|---|---|
| `labels:` | `labelsPath:` (or `labels:` with a `List<String>`) |
| `modelVersion:` | Not needed: detected from the model. Use `format:` / `task:` only to force it. |
| `useGpu: true` | `accelerator: Accelerator.gpu` |
| `quantization:` | Not needed: int8 models are detected. |
| `isAsset: false` | `source: ModelSource.file` |
| `rotation:` (at load time) | `CameraFrame(rotation: ...)` on each frame |

## Detecting

```dart
// 2.x
final results = await vision.yoloOnImage(
  bytesList: bytes, imageHeight: h, imageWidth: w,
  iouThreshold: 0.4, confThreshold: 0.4, classThreshold: 0.5,
);
final left = results[0]['box'][0];

// 3.0
final result = await model.detectImage(
  bytes,
  options: const DetectionOptions(confidenceThreshold: 0.5, iouThreshold: 0.4),
);
final left = result.detections.first.box.left;
```

* `imageHeight` / `imageWidth` are no longer needed for images.
* `confThreshold` and `classThreshold` are merged into `confidenceThreshold`
  (for YOLOv5 the score is objectness × class score, as in Ultralytics).
* Results are `Detection` objects instead of maps:

| 2.x map | 3.0 |
|---|---|
| `box[0..3]` | `box` (`Rect`: `left`, `top`, `right`, `bottom`) |
| `box[4]` | `confidence` |
| `tag` | `label` (and `classIndex`) |
| `polygons` (relative to the box) | `polygon` (absolute image coordinates) |

Coordinates are in pixels of the upright image: the EXIF orientation of photos
and the rotation of camera frames are already applied. Use `BoxMapper` to
convert them to widget coordinates.

## Camera frames

```dart
// 2.x
await vision.yoloOnFrame(
  bytesList: image.planes.map((p) => p.bytes).toList(),
  imageHeight: image.height,
  imageWidth: image.width,
);

// 3.0
final result = await model.detectFrame(CameraFrame(
  width: image.width,
  height: image.height,
  format: FrameFormat.nv21, // ImageFormatGroup.nv21 on Android
  planes: [
    for (final p in image.planes)
      FramePlane(bytes: p.bytes, bytesPerRow: p.bytesPerRow,
          bytesPerPixel: p.bytesPerPixel ?? 1),
  ],
  rotation: rotation, // see CameraFrame.rotationFor
));
if (result == null) return; // previous frame still running, frame skipped
```

## Releasing

```dart
await vision.closeYoloModel(); // 2.x
await model.dispose();         // 3.0
```

## Platform setup

* Remove `aaptOptions { noCompress 'tflite' }` if you added it only for this
  plugin: it is the default.
* The plugin no longer declares the camera and storage permissions. Declare
  the ones your app needs.
* iOS: set the deployment target to 15.0 or higher and use Swift Package
  Manager. It is on by default in Flutter 3.47; on older versions run
  `flutter config --enable-swift-package-manager`.
