# flutter_vision

A Flutter plugin for managing [Yolov5, Yolov8](https://github.com/ultralytics/ultralytics) and [Tesseract v5](https://tesseract-ocr.github.io/tessdoc/) accessing with TensorFlow Lite 2.x. Support object detection, segmentation and OCR on Android. iOS not updated, working in progress.

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
## For YoloV5 and YoloV8 MODEL
1. Create a `assets` folder and place your labels file and model file in it. In `pubspec.yaml` add:

```
  assets:
   - assets/labels.txt
   - assets/yolovx.tflite
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
`modelVersion`: yolov5 or yolov8 or yolov8seg
```dart
await vision.loadYoloModel(
        labels: 'assets/labelss.txt',
        modelPath: 'assets/yolov5n.tflite',
        modelVersion: "yolov5",
        quantization: false,
        numThreads: 1,
        useGpu: false);
```
### For camera live feed
5. Make your first detection:
`confThreshold` work with yolov5 other case it is omited.
> _Make use of [camera plugin](https://pub.dev/packages/camera)_

```dart
final result = await vision.yoloOnFrame(
        bytesList: cameraImage.planes.map((plane) => plane.bytes).toList(),
        imageHeight: cameraImage.height,
        imageWidth: cameraImage.width,
        iouThreshold: 0.4,
        confThreshold: 0.4,
        classThreshold: 0.5);
```

### For static image
5. Make your first detection or segmentation:

```dart
final result = await vision.yoloOnImage(
        bytesList: byte,
        imageHeight: image.height,
        imageWidth: image.width,
        iouThreshold: 0.8,
        confThreshold: 0.4,
        classThreshold: 0.7);
```

6. Release resources:

```dart
await vision.closeYoloModel();
```
## For Tesseract 5.0.0 MODEL
1. Create an `assets` folder, then create a `tessdata` directory and  `tessdata_config.json` file and place them into it.
Download trained data for tesseract from [here](https://github.com/tesseract-ocr/tessdata) and place it into tessdata directory. Then, modifie tessdata_config.json as follow.
```json
{
    "files": [
      "spa.traineddata"
    ]
}
```

2.  In `pubspec.yaml` add:
```
assets:
    - assets/
    - assets/tessdata/
```
3. Import the library:

```dart
import 'package:flutter_vision/flutter_vision.dart';
```

4. Initialized the flutter_vision library:

```dart 
 FlutterVision vision = FlutterVision();
```

5. Load the model:

```dart
await vision.loadTesseractModel(
      args: {
        'psm': '11',
        'oem': '1',
        'preserve_interword_spaces': '1',
      },
      language: 'spa',
    );
```

### For static image
6. Get Text from static image:

```dart
    final XFile? photo = await picker.pickImage(source: ImageSource.gallery);
    if (photo != null) {
      final result = await vision.tesseractOnImage(bytesList: (await photo.readAsBytes()));
    }
```

7. Release resources:

```dart
await vision.closeTesseractModel();
```
# About results
## For Yolo v5 or v8 in detection task
result is a `List<Map<String,dynamic>>` where Map have the following keys:

 ``` dart
    Map<String, dynamic>:{
     "box": [x1:left, y1:top, x2:right, y2:bottom, class_confidence]
     "tag": String: detected class
    }
```

## For YoloV8 in segmentation task
result is a `List<Map<String,dynamic>>` where Map have the following keys:

 ``` dart
    Map<String, dynamic>:{
     "box": [x1:left, y1:top, x2:right, y2:bottom, class_confidence]
     "tag": String: detected class
     "polygons": List<Map<String, double>>: [{x:coordx, y:coordy}]
    }
```

## For Tesseract
result is a `List<Map<String,dynamic>>` where Map have the following keys:

```dart
    Map<String, dynamic>:{
      "text": String
      "word_conf": List:int
      "mean_conf": int}
```

# Example
![Screenshot_2022-04-08-23-59-05-652_com vladih dni_scanner_example](https://user-images.githubusercontent.com/32783435/164163922-2eb7c8a3-8415-491f-883e-12cc87512efe.jpg)
<img src="https://github.com/vladiH/flutter_vision/assets/32783435/8fbbb9da-062e-4089-b1f6-1d4b2795664c" alt="Home" width="250" height="555">
<img src="https://github.com/vladiH/flutter_vision/assets/32783435/51098356-8ed7-4c72-bce0-cf64d56976cd" alt="Detection" width="250" height="555">
<img src="https://github.com/vladiH/flutter_vision/assets/32783435/0368ae2f-89ad-4a1a-a4c3-0548f0948421" alt="Segmentation" width="250" height="555">


# <div align="center">Contact</div>

For flutter_vision bug reports and feature requests please visit [GitHub Issues](https://github.com/vladiH/flutter_vision/issues)

<br>
<div align="center">
  <a href="https://linktr.ee/randpoint" style="text-decoration:none;">
    <img src="https://github.com/vladiH/flutter_vision/assets/32783435/75d2c677-65b5-4b9d-b332-d9d08602f024" width="3%" alt="" /></a>
</div>
