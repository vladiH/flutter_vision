# flutter_vision example

Object detection and segmentation on photos and on the live camera. Pick the
model (YOLO11n, YOLOv8n, YOLOv5n or YOLOv8n-seg) and the hardware (CPU, GPU,
NPU) on the home screen.

## Models

The models live in `assets/` in the
[GitHub repository](https://github.com/vladiH/flutter_vision/tree/master/example/assets).
They are left out of the pub.dev package to keep it small, so if you got the
example from pub.dev, copy `yolov5n.tflite`, `yolov8n.tflite`,
`yolov8n-seg.tflite` and `yolov11n.tflite` from there into `assets/`, or export
your own with `yolo export model=yolo11n.pt format=tflite`.

## Run

```bash
flutter run
```

On iOS, use a real iPhone or the simulator on an Apple Silicon Mac.

## Tests

End-to-end tests on a device or emulator (one test downloads a sample photo
from ultralytics.com and is skipped when offline):

```bash
flutter test integration_test
```
