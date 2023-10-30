#import "FlutterVisionPlugin.h"
#if __has_include(<flutter_vision/flutter_vision-Swift.h>)
#import <flutter_vision/flutter_vision-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "flutter_vision-Swift.h"
#endif

@implementation FlutterVisionPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftFlutterVisionPlugin registerWithRegistrar:registrar];
}
@end
