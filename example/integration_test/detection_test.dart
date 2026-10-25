import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_vision/flutter_vision.dart';
import 'package:flutter_vision_example/model_catalog.dart';
import 'package:integration_test/integration_test.dart';

/// Solid image encoded as PNG.
Future<Uint8List> syntheticPng(int width, int height) async {
  final recorder = ui.PictureRecorder();
  final canvas = Canvas(recorder);
  canvas.drawRect(
    Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
    Paint()..color = const Color(0xFF808080),
  );
  final image = await recorder.endRecording().toImage(width, height);
  final data = await image.toByteData(format: ui.ImageByteFormat.png);
  return data!.buffer.asUint8List();
}

/// Downloads Ultralytics' sample photo (a bus and four people), or null when
/// offline.
Future<Uint8List?> downloadBusPhoto() async {
  try {
    final client = HttpClient()..connectionTimeout = const Duration(seconds: 10);
    final request =
        await client.getUrl(Uri.parse('https://ultralytics.com/images/bus.jpg'));
    final response = await request.close();
    if (response.statusCode != 200) return null;
    final builder = BytesBuilder();
    await response.forEach(builder.add);
    client.close();
    return builder.takeBytes();
  } catch (_) {
    return null;
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  for (final option in bundledModels) {
    group(option.name, () {
      late YoloModel model;

      setUpAll(() async {
        model = await loadModel(option, Accelerator.cpu);
      });

      tearDownAll(() => model.dispose());

      testWidgets('loads with the expected layout', (tester) async {
        expect(model.task, option.task);
        expect(model.labels, hasLength(80));
        expect(model.inputSize, const Size(640, 640));
        expect(model.accelerator, Accelerator.cpu);
      });

      testWidgets('runs on an encoded image', (tester) async {
        final png = await syntheticPng(1280, 720);
        final result = await model.detectImage(png);
        expect(result.imageSize, const Size(1280, 720));
        expect(result.inferenceTime, greaterThan(Duration.zero));
        for (final d in result.detections) {
          expect(d.box.left, greaterThanOrEqualTo(0));
          expect(d.box.right, lessThanOrEqualTo(1280));
        }
      });

      testWidgets('runs on rotated camera frames', (tester) async {
        const width = 640;
        const height = 480;
        // NV21: Y plane then interleaved VU, mid grey.
        final nv21 = Uint8List(width * height * 3 ~/ 2)..fillRange(0, width * height * 3 ~/ 2, 128);
        final result = await model.detectFrame(CameraFrame(
          width: width,
          height: height,
          format: FrameFormat.nv21,
          planes: [FramePlane(bytes: nv21, bytesPerRow: width)],
          rotation: 90,
        ));
        expect(result, isNotNull);
        expect(result!.imageSize, const Size(480, 640));

        final bgra = Uint8List(width * height * 4)..fillRange(0, width * height * 4, 200);
        final bgraResult = await model.detectFrame(CameraFrame(
          width: width,
          height: height,
          format: FrameFormat.bgra8888,
          planes: [FramePlane(bytes: bgra, bytesPerRow: width * 4, bytesPerPixel: 4)],
        ));
        expect(bgraResult!.imageSize, const Size(640, 480));
      });

      testWidgets('finds the bus and the people in a real photo', (tester) async {
        final photo = await downloadBusPhoto();
        if (photo == null) {
          markTestSkipped('No network: cannot download the sample photo.');
          return;
        }
        final result = await model.detectImage(photo);
        final labels = result.detections.map((d) => d.label).toList();
        debugPrint('${option.name}: ${result.detections}');
        expect(result.imageSize, const Size(810, 1080));
        expect(labels, contains('bus'));
        expect(labels.where((l) => l == 'person').length, greaterThanOrEqualTo(3));
        final bus = result.detections.firstWhere((d) => d.label == 'bus');
        // The bus spans most of the photo width.
        expect(bus.box.width, greaterThan(600));
        expect(bus.confidence, greaterThan(0.5));
        if (option.task == YoloTask.segment) {
          expect(bus.polygon, isNotNull);
          expect(bus.polygon!.length, greaterThan(4));
          for (final p in bus.polygon!) {
            expect(bus.box.inflate(2).contains(p), isTrue);
          }
        }
      });
    });
  }

  testWidgets('several models at once and dispose', (tester) async {
    final a = await loadModel(bundledModels[0], Accelerator.cpu);
    final b = await loadModel(bundledModels[1], Accelerator.cpu);
    final png = await syntheticPng(320, 240);
    final results = await Future.wait([a.detectImage(png), b.detectImage(png)]);
    expect(results, hasLength(2));
    await a.dispose();
    expect(() => a.detectImage(png), throwsA(isA<FlutterVisionException>()));
    expect((await b.detectImage(png)).imageSize, const Size(320, 240));
    await b.dispose();
  });

  testWidgets('GPU request falls back gracefully', (tester) async {
    final model = await loadModel(bundledModels[0], Accelerator.gpu);
    debugPrint('GPU request ran on: ${model.accelerator}');
    final result = await model.detectImage(await syntheticPng(640, 480));
    expect(result.imageSize, const Size(640, 480));
    await model.dispose();
  });

  testWidgets('reports a missing asset', (tester) async {
    await expectLater(
      YoloModel.load(modelPath: 'assets/missing.tflite', labels: const ['a']),
      throwsA(isA<FlutterVisionException>()
          .having((e) => e.code, 'code', FlutterVisionException.notFound)),
    );
  });
}
