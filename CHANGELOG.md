## 3.0.0

Complete rewrite. See [MIGRATION.md](MIGRATION.md) to upgrade from 2.x.

* **iOS support** (iOS 15+, Swift Package Manager), with the same results as
  Android.
* New API: `YoloModel.load`, `detectImage`, `detectFrame`, `dispose`, typed
  `Detection` results (`Rect` box, label, confidence, polygon) and
  `BoxMapper` to draw results on screen.
* Several models can be loaded at the same time.
* The model format is detected from the model itself: YOLOv5/v7, v8, v9, v11,
  v12 and NMS-free YOLOv10 / YOLO26. `modelVersion` is gone.
* LiteRT 2.3 (`CompiledModel` API) with CPU, GPU and NPU acceleration and
  automatic fallback to CPU.
* Correct camera frames on every device: row padding and pixel strides,
  NV21, YUV420 and BGRA, and a per-frame rotation.
* `detectImage` applies the EXIF orientation and decodes large photos at the
  size the model needs (no more out-of-memory errors).
* Letterbox resizing (like Ultralytics) instead of stretching, class-aware NMS
  and correct boxes for non-square models.
* Segmentation outlines are traced in Dart: OpenCV was removed, the native
  libraries are 16 KB page aligned (Android 15+) and apps are tens of MB
  smaller.
* Models with int8 inputs/outputs and NCHW inputs are supported.
* No more crashes when a model is closed or replaced during a detection, and
  no more native memory leaks.
* The plugin no longer adds camera or storage permissions to your app.
* Requires Flutter 3.41+, Android minSdk 24, iOS 15.

## 2.0.0
* **BREAKING CHANGE**: Removed all Tesseract OCR functionality.
* Added support for YOLOv11 (YOLO11).
* Improved tensor shape handling for better model compatibility.
* Enhanced memory management and resource cleanup.
* Better error handling and thread safety.
* Updated to support Gradle 8.x and Dart 3.6.0+.
* Performance optimizations across all models.
* Now support orientation and isAsset parameters

## 1.1.4
* Resolved the YoloV8 output bug and made updates to the latest version. Please find the latest release on the Ultralytics YOLOv8 0.181 GitHub repository.

## 1.1.3
* Release of segmentation feature via YOLOv8.
* Updated example code.
* Updated README.

## 1.1.2
* GPU delegation error has been fixed.
* Coordinate representation of the box in documentation was fixed.
* Switching between models now are supported.
* Added quantization option for more efficient models at the cost of some precision.

## 1.1.1
* Bounding box error has been fixed.
* Confidence scores for Yolov8 has been fixed.

## 1.1.0
* loadOcrModel, ocrOnFrame, and closeOcrModel have been removed. Instead, Yolo and Tesseract operate independently of each other.
* Models no longer returns responseHandler as output. Instead, it returns a List<Map<String, dynamic>>.
* The Tesseract model has been updated to version 5.0.0, resulting in improved accuracy.
* New methods have been added: loadYoloModel, yoloOnFrame, yoloOnImage, closeYoloModel, loadTesseractModel, tesseractOnImage, and closeTesseractModel.
* Support is now available for both Yolov5 and Yolov8.
* Resource management has been improved, and all models now operate in the background.

## 1.0.0
* Methods for Yolov5 now is available `(loadYoloModel, yoloOnFrame, closeYoloModel)`.
* Yolov5 and OCR model now is independent one to each other.

## 0.0.2
* `best` parameter has been removed.

## 0.0.1
* Initial release.
