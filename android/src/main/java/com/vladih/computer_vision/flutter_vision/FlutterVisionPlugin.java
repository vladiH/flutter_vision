package com.vladih.computer_vision.flutter_vision;

import androidx.annotation.NonNull;
import com.vladih.computer_vision.flutter_vision.models.ocr;
import org.opencv.android.OpenCVLoader;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;

/** FlutterVisionPlugin */
public class FlutterVisionPlugin implements FlutterPlugin, MethodCallHandler {
  /// The MethodChannel that will the communication between Flutter and native Android
  ///
  /// This local reference serves to register the plugin with the Flutter Engine and unregister it
  /// when the Flutter Engine is detached from the Activity
  private MethodChannel channel;
  private FlutterPluginBinding binding;
  private Result result;
  private ocr scanner;
  @Override
  public void onAttachedToEngine(@NonNull FlutterPluginBinding flutterPluginBinding) {
    OpenCVLoader.initDebug();
    channel = new MethodChannel(flutterPluginBinding.getBinaryMessenger(), "flutter_vision");
    channel.setMethodCallHandler(this);
    binding = flutterPluginBinding;
  }

  @Override
  public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {
    this.result = result;
    if (call.method.equals("loadOcrModel")) {
      load_ocr_model((Map) call.arguments);
    }else if(call.method.equals("ocrOnFrame")){
      ocr_on_frame((Map) call.arguments);
    } else if(call.method.equals("closeOcrModel")){
      close_ocr_model();
    }else {
      result.notImplemented();
    }
  }

  @Override
  public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
    channel.setMethodCallHandler(null);
  }

  private void load_ocr_model(Map<String, Object> args){
    try {
      /*for(Map.Entry entry:args.entrySet()){
        System.out.println(entry.getKey());
        System.out.println(entry.getValue());
      }*/
      final String model = args.get("model_path").toString();
      final Object is_asset_obj = args.get("is_asset");
      final boolean is_asset = is_asset_obj==null?false:(boolean) is_asset_obj;
      final int num_threads = (int) args.get("num_threads");
      final boolean use_gpu = (boolean) args.get("use_gpu");
      final String label_path= args.get("label_path").toString();
      final float image_mean= (float)((double) args.get("image_mean"));
      final float image_std= (float)((double) args.get("image_std"));
      final int rotation= (int) args.get("rotation");
      final boolean best= (boolean) args.get("best");
      final String tess_data = args.get("tess_data").toString();
      final Map<String,String> arg = (Map<String,String>) args.get("arg");
      final String language = args.get("language").toString();
      scanner = new ocr(binding,
              model,
              is_asset,
              num_threads,
              use_gpu,
              label_path,
              image_mean,
              image_std,
              rotation,
              best, tess_data, arg, language);
      scanner.initialize_model();
      this.result.success("ok");
    }catch (Exception e){
      this.result.error("100", "Cannot initialize model", e);
    }
  }

  private void ocr_on_frame(Map<String, Object> args){
    try {
      List<byte[]> image = (ArrayList) args.get("bytesList");
      int image_height = (int) args.get("image_height");
      int image_width = (int) args.get("image_width");
      float iou_threshold = (float)(double)( args.get("iou_threshold"));
      float conf_threshold = (float)(double)( args.get("conf_threshold"));
      List<Integer> class_is_text = (List<Integer>) args.get("class_is_text");
      List<Map<String, Object>> result = scanner.predict(image, image_height, image_width, iou_threshold, conf_threshold,class_is_text);
      this.result.success(result);
    }catch (Exception e){
      this.result.error("100", "Prediction Error", e);
    }
  }

  private void close_ocr_model(){
    scanner.close();
  }
}
