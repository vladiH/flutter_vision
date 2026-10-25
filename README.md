# flutter_vision

[![pub package](https://img.shields.io/pub/v/flutter_vision.svg)](https://pub.dev/packages/flutter_vision)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Real-time object detection and instance segmentation with
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models in
Flutter, on **Android and iOS**, powered by
[LiteRT](https://ai.google.dev/edge/litert) (TensorFlow Lite).

## What's new in 3.0

Version 3.0 is a complete rewrite. Highlights:

* **iOS support**, with the same results as Android.
* A new, typed API: `YoloModel.load`, `detectImage`, `detectFrame`, `dispose`,
  `Detection` results and `BoxMapper` to draw them.
* The model format is detected automatically; several models can run at once.
* LiteRT 2.3 with CPU, GPU and NPU acceleration and automatic CPU fallback.
* Correct camera frames on every device, EXIF orientation for photos,
  letterbox resizing like Ultralytics, class-aware NMS.
* OpenCV removed: apps are much smaller and the native libraries meet the
  Android 15+ 16 KB page size requirement.

Upgrading from 2.x? Read [MIGRATION.md](MIGRATION.md). Full list of changes in
[CHANGELOG.md](CHANGELOG.md).

## Platform support

| | Android | iOS |
|---|---|---|
| Minimum version | API 24 (Android 7.0) | iOS 15 |
| Detection | ✅ | ✅ |
| Segmentation | ✅ | ✅ |
| CPU / GPU | ✅ / ✅ (OpenCL, OpenGL) | ✅ / ✅ (Metal) |
| NPU | ✅ with the vendor runtime (see below) | — |
| Tested on | Android 16 emulator | iOS simulator (Xcode 26.6, Apple Silicon) |

Flutter 3.41 or newer is required (developed and tested with Flutter 3.47.8).

## Supported models

The output format is detected automatically from the model, so any
Ultralytics TFLite export works:

| Family | Models | Format |
|---|---|---|
| Anchor-based | YOLOv5, YOLOv7 | `[1, N, 5 + classes]` |
| Anchor-free | YOLOv8, YOLOv9, YOLO11, YOLOv12 | `[1, 4 + classes, N]` |
| NMS-free | YOLOv10, YOLO26 | `[1, 300, 6]` |

Detection and segmentation (`-seg`) variants are supported, with float32,
float16 and int8 models.

Results of the bundled example models on Ultralytics' sample photo (a bus and
four people). Android and iOS agree within about two points:

| Model | Bus | People found (≥ 50%) | Task |
|---|---|---|---|
| YOLO11n | 94% | 3 (90%, 85%, 83%) | detection |
| YOLOv8n | 84% | 3 (89%, 88%, 88%) | detection |
| YOLOv5n | 79% | 3 (87%, 83%, 79%) | detection |
| YOLOv8n-seg | 87% | 3 (86%, 83%, 81%) | segmentation |

## Installation

```yaml
dependencies:
  flutter_vision: ^3.0.0
```

### Android

Nothing to configure. The plugin needs `minSdk 24` (the Flutter default) and
does not add any permission to your app; declare `CAMERA` yourself if you use
the camera.

### iOS

* Set the iOS deployment target to **15.0** or higher (Xcode: Runner target >
  General > Minimum Deployments).
* The iOS side is a Swift package. Swift Package Manager is on by default in
  Flutter 3.47; on older versions run
  `flutter config --enable-swift-package-manager`.
* LiteRT ships arm64 binaries only: real iPhones and the iOS Simulator on
  Apple Silicon Macs work; the simulator on Intel Macs (x86_64) does not.
  `flutter run` on a simulator builds only arm64, but
  `flutter build ios --simulator` builds both architectures and fails to link
  unless you exclude x86_64. In Xcode, select the **Runner target** > Build
  Settings > Excluded Architectures and set "Any iOS Simulator SDK" to
  `$(inherited) x86_64` for Debug, Profile and Release. This writes
  `"EXCLUDED_ARCHS[sdk=iphonesimulator*]" = "$(inherited) x86_64";` in
  `project.pbxproj`. (At project level it has no effect: Flutter's
  `Generated.xcconfig` overrides it.)
* Add `NSCameraUsageDescription` to `Info.plist` if you use the camera.

## Preparing a model

Export your model with Ultralytics:

```bash
pip install ultralytics
yolo export model=yolo11n.pt format=tflite          # detection
yolo export model=yolo11n-seg.pt format=tflite      # segmentation
```

Use one of the generated files, for example
`yolo11n_saved_model/yolo11n_float16.tflite` (`_float32` and `_int8` also
work). Create a labels file with one class name per line, in class index
order (the `names` of your `data.yaml`; a `0: person` style is accepted too).

Add both to your app's assets:

```yaml
flutter:
  assets:
    - assets/yolo11n_float16.tflite
    - assets/labels.txt
```

## Usage

```dart
import 'package:flutter_vision/flutter_vision.dart';

final model = await YoloModel.load(
  modelPath: 'assets/yolo11n_float16.tflite',
  labelsPath: 'assets/labels.txt',
  accelerator: Accelerator.gpu, // falls back to CPU automatically
);

final result = await model.detectImage(imageBytes); // JPEG, PNG, HEIC...
for (final d in result.detections) {
  print('${d.label} ${(d.confidence * 100).round()}% at ${d.box}');
}

await model.dispose();
```

Several models can be loaded at once. Each model runs on its own native
thread, and calls to the same model run one after another. Models can also be
loaded from a file on the device with `source: ModelSource.file`.

### Options

```dart
final result = await model.detectImage(
  imageBytes,
  options: const DetectionOptions(
    confidenceThreshold: 0.4, // minimum score
    iouThreshold: 0.45,       // overlap allowed between boxes (NMS)
    maxDetections: 100,
    classes: {0, 2},          // only these class indices
    classAgnosticNms: false,
    maskThreshold: 0.5,       // segmentation only
  ),
);
```

### Live camera

Use the [camera](https://pub.dev/packages/camera) plugin with NV21 frames on
Android and BGRA on iOS:

```dart
final controller = CameraController(
  camera,
  ResolutionPreset.medium,
  enableAudio: false,
  imageFormatGroup:
      Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
);
await controller.initialize();
await controller.startImageStream((image) async {
  final result = await model.detectFrame(CameraFrame(
    width: image.width,
    height: image.height,
    format: Platform.isAndroid ? FrameFormat.nv21 : FrameFormat.bgra8888,
    planes: [
      for (final p in image.planes)
        FramePlane(
          bytes: p.bytes,
          bytesPerRow: p.bytesPerRow,
          bytesPerPixel: p.bytesPerPixel ?? 1,
        ),
    ],
    // Android delivers frames in the sensor orientation; iOS already
    // delivers them upright.
    rotation: Platform.isAndroid
        ? CameraFrame.rotationFor(
            sensorOrientation: camera.sensorOrientation,
            deviceOrientation:
                CameraFrame.degreesFor(controller.value.deviceOrientation.name),
            frontCamera: camera.lensDirection == CameraLensDirection.front,
          )
        : 0,
  ));
  if (result == null) return; // previous frame still running: frame skipped
  // update the UI with result.detections
});
```

`detectFrame` skips frames while the previous one is still running, so the
results stay real time instead of lagging behind. YUV420 frames with row
padding (Android `YUV_420_888`) are supported too.

### Drawing the results

Coordinates are pixels of the upright image (EXIF orientation and frame
rotation already applied). `BoxMapper` converts them to the widget that shows
the image; use the same `BoxFit`:

```dart
final mapper = BoxMapper(
  imageSize: result.imageSize,
  widgetSize: size,
  fit: BoxFit.cover,     // BoxFit.contain for Image widgets
  mirror: false,         // true for the Android front camera preview
);
canvas.drawRect(mapper.mapRect(detection.box), paint);
if (detection.polygon != null) {
  canvas.drawPath(Path()..addPolygon(mapper.mapPolygon(detection.polygon!), true), paint);
}
```

## Results

`YoloResult`:

| Field | |
|---|---|
| `detections` | `List<Detection>`, highest confidence first |
| `imageSize` | size of the upright image |
| `preprocessTime`, `inferenceTime`, `postprocessTime`, `totalTime` | timings |

`Detection`:

| Field | |
|---|---|
| `box` | `Rect` in image pixels |
| `label`, `classIndex` | class |
| `confidence` | 0 to 1 |
| `polygon` | outline in image pixels (segmentation models only) |

`YoloModel` also exposes `task`, `format`, `inputSize`, `labels` and
`accelerator` (the hardware actually used).

## Hardware acceleration

* `Accelerator.cpu` (default): XNNPACK, works everywhere.
* `Accelerator.gpu`: OpenCL/OpenGL on Android, Metal on iOS. The first load
  takes longer while the GPU program is compiled.
* `Accelerator.npu`: Android only. Add the LiteRT NPU runtime for your
  chipset to your app (for example
  `com.google.ai.edge.litert:litert-npu-runtime-qualcomm:2.3.0`); without it
  the model falls back to GPU, then CPU.

`model.accelerator` tells which one is actually used.

## Example app

The [example app](example) has screens for photos, segmentation and the live
camera, and lets you pick the model and the hardware. Its models are in the
GitHub repository (they are left out of the pub.dev package to keep it small).

## Testing

```bash
flutter test                                   # Dart unit tests (decoding, NMS, masks...)
cd example && flutter test integration_test    # end-to-end tests on a device or emulator
```

The integration tests load the four example models, run them on images and
camera frames, and check the detections on a real photo. The repository's
[`codemagic.yaml`](codemagic.yaml) runs them on an iOS simulator in
[Codemagic](https://codemagic.io).

## Troubleshooting

| Error | Fix |
|---|---|
| `not_found` | The path must match the asset entry in `pubspec.yaml` exactly. |
| `unsupported_model`: labels do not match | The labels file must have one line per class of the model. |
| `unsupported_model`: input or outputs | Use a YOLO TFLite export with float32 or int8 tensors (`_float16`, `_float32` or `_int8` file). |
| iOS: `Undefined symbol: _LiteRt...` for x86_64 | Exclude x86_64 for the simulator (see the iOS installation notes). |
| Boxes are shifted on screen | Use `BoxMapper` with the same `BoxFit` as the widget showing the image. |

## Known limitations

* uint8-input models (for example non-YOLO SSD models) are not supported.
* The iOS Simulator needs an Apple Silicon Mac.

## Developing the iOS side on Windows

* [Codemagic](https://codemagic.io) (cloud Macs, free tier) builds and tests
  the plugin on an iOS simulator with the included `codemagic.yaml`.
* [xcross](https://github.com/arxdeus/xcross) builds, installs and hot-reloads
  Flutter iOS apps on a real iPhone (iOS 17+) from Windows or Linux. It works
  with this plugin because the iOS side is a Swift package.

## Releases

* Versions and their changes: [CHANGELOG.md](CHANGELOG.md).
* Published versions: [pub.dev/packages/flutter_vision](https://pub.dev/packages/flutter_vision/versions).
* Source of each release: [GitHub releases](https://github.com/vladiH/flutter_vision/releases).

## Contact

- Bug reports and feature requests: [GitHub Issues](https://github.com/vladiH/flutter_vision/issues)
- Email: yurihuallpavargas@gmail.com
