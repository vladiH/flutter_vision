## 1.1.4
*  Resolved the YoloV8 output bug and made updates to the latest version. Please find the latest release on the Ultralytics YOLOv8 0.181 GitHub repository
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
* Methods  for Yolov5 now is available `(loadYoloModel, yoloOnFrame, closeYoloModel)`.
* Yolov5 and OCR model now is independent one to each other.

## 0.0.2
* `best` parameter has been removed

## 0.0.1
* Initial release.
