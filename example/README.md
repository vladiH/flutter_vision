# flutter_vision example

Object detection and segmentation on photos and on the live camera, with the
models in `assets/` (YOLO11n, YOLOv8n, YOLOv5n and YOLOv8n-seg).

```bash
flutter run
```

End-to-end tests on a device or emulator (one test downloads a sample photo
from ultralytics.com and is skipped when offline):

```bash
flutter test integration_test
```
