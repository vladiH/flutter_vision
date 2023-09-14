import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
// import 'package:flutter_vision/flutter_vision.dart';

void main() {
  const MethodChannel channel = MethodChannel('flutter_vision');

  TestWidgetsFlutterBinding.ensureInitialized();

  handler(MethodCall methodCall) async {
    if (methodCall.method == 'getAll') {
      return <String, dynamic>{
        'appName': 'myapp',
        'packageName': 'com.mycompany.myapp',
        'version': '0.0.1',
        'buildNumber': '1'
      };
    }
    return null;
  }

  TestWidgetsFlutterBinding.ensureInitialized();

  TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
      .setMockMethodCallHandler(channel, handler);
}
