# flutter_vision

A Flutter plugin for managing both [Yolov5](https://github.com/ultralytics/yolov5) model and [Tesseract v4](https://tesseract-ocr.github.io/tessdoc/) accessing with TensorFlow Lite 2.x. Support object detection and OCR on both iOS and Android.

# Installation
Add flutter_vision as a dependency in your pubspec.yaml file.

## Android
In `android/app/build.gradle`, add the following setting in android block.

```gradle
    android{
        aaptOptions {
            noCompress 'tflite'
            noCompress 'lite'
        }
    }
```
## iOS
Comming soon ...

# Usage
## For OCR MODEL
1. Create a `assets` folder and place your labels file and model file in it. In `pubspec.yaml` add:

```
  assets:
   - assets/labels.txt
   - assets/yolov5.tflite
```

2. You must add trained data and trained data config file to your assets directory. You can find additional language trained data files [here](https://github.com/tesseract-ocr/tessdata) Trained language files.

add tessdata folder under assets folder, add tessdata_config.json file under assets folder:

```json
{
    "files": [
      "spa.traineddata"
    ]
}
```

3. Import the library:

```dart
import 'package:flutter_vision/flutter_vision.dart';
```

4. Initialized the flutter_vision library:

```dart 
 FlutterVision vision = FlutterVision();
```

5. Load the model and labels:

```dart
final responseHandler = await vision.loadOcrModel(
    labels: 'assets/labels.txt',
    modelPath: 'assets/yolov5.tflite',
    args: {
      'psm': '11',
      'oem': '1',
      'preserve_interword_spaces': '1',
    },
    language: 'spa',
    numThreads: 1,
    useGpu: false);
```

6. Make your first detection and extract text from it if you want:
> _Make use of [camera plugin](https://pub.dev/packages/camera)_

__classIsText__ parameter set index of class which you want to extract text looking at position of text in `labels.txt` file.
```dart
final responseHandler = await vision.ocrOnFrame(
    bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
    imageHeight: cameraImage.height,
    imageWidth: cameraImage.width,
    classIsText: [0],
    iouThreshold: 0.6,
    confThreshold: 0.6);
```

7. Release resources:

```dart
await vision.closeOcrModel();
```

## For YoloV5 MODEL
1. Create a `assets` folder and place your labels file and model file in it. In `pubspec.yaml` add:

```
  assets:
   - assets/labels.txt
   - assets/yolov5.tflite
```

2. Import the library:

```dart
import 'package:flutter_vision/flutter_vision.dart';
```

3. Initialized the flutter_vision library:

```dart 
 FlutterVision vision = FlutterVision();
```

4. Load the model and labels:

```dart
final responseHandler = await vision.loadYoloModel(
    labels: 'assets/labels.txt',
    modelPath: 'assets/yolov5.tflite',
    numThreads: 1,
    useGpu: false);
```

5. Make your first detection:
> _Make use of [camera plugin](https://pub.dev/packages/camera)_

```dart
final responseHandler = await vision.yoloOnFrame(
    bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
    imageHeight: cameraImage.height,
    imageWidth: cameraImage.width,
    iouThreshold: 0.6,
    confThreshold: 0.6);
```

7. Release resources:

```dart
await vision.closeYoloModel();
```

# About responseHandler object
+ Parameters
    + type: `success` or `error`.
    + message: if type is success it is `ok`, otherwise it has information about error.
    + StackTrace: StackTrace about error.
    + data: It is a `List<Map<String, dynamic>>` on ocrOnFrame, otherwise it is a `null`.

data: Contain information about objects detected, such as confidence of detection, coordinates of detected box(x1,y1,x2,y2), detected box image, text from detected box and a tag belonging to detected class.


```dart
class ResponseHandler {
  String type;
  String message;
  StackTrace? stackTrace;
  List<Map<String, dynamic>> data;
}
```

# Example
![Screenshot_2022-04-08-23-59-05-652_com vladih dni_scanner_example](https://user-images.githubusercontent.com/32783435/164163922-2eb7c8a3-8415-491f-883e-12cc87512efe.jpg)
![Screenshot_2022-04-08-23-59-42-594_com vladih dni_scanner_example](https://user-images.githubusercontent.com/32783435/164163927-b290e46b-2af8-4b2b-a6a4-88cf4075f388.jpg)
![Screenshot_2022-04-09-00-00-53-316_com vladih dni_scanner_example](https://user-images.githubusercontent.com/32783435/164163929-4b22310a-e6f6-4453-886b-7c7b622892de.jpg)