package com.vladih.computer_vision.flutter_vision;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.annotation.NonNull;

import com.vladih.computer_vision.flutter_vision.models.Yolo;
import com.vladih.computer_vision.flutter_vision.models.Yolov8;
import com.vladih.computer_vision.flutter_vision.models.tesseract;
import com.vladih.computer_vision.flutter_vision.models.Yolov5;
import com.vladih.computer_vision.flutter_vision.utils.utils;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


import io.flutter.Log;
import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;

/** FlutterVisionPlugin */
public class FlutterVisionPlugin implements FlutterPlugin, MethodCallHandler {
  private static final String CHANNEL_NAME = "flutter_vision";
  private MethodChannel methodChannel;
  private Context context;
  private  FlutterAssets assets;
  private Yolo yolo;
  private tesseract tesseract;
  @Override
  public void onAttachedToEngine(@NonNull FlutterPluginBinding binding) {
    setupChannel(binding.getApplicationContext(), binding.getFlutterAssets(), binding.getBinaryMessenger());
  }

  @Override
  public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
    this.context = null;
    this.methodChannel.setMethodCallHandler(null);
    this.methodChannel = null;
    this.assets = null;
  }

  private void setupChannel(Context context, FlutterAssets assets, BinaryMessenger messenger) {
    OpenCVLoader.initDebug();
    this.assets = assets;
    this.context = context;
    this.methodChannel = new MethodChannel(messenger, CHANNEL_NAME);
    this.methodChannel.setMethodCallHandler(this);
  }

  @Override
  public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {
    // Handle method calls from Flutter
    if (call.method.equals("loadOcrModel")) {
      try {
        load_ocr_model((Map) call.arguments);
      } catch (Exception e) {
        result.error("100","Error on load ocr components", e);
      }
    }else if(call.method.equals("ocrOnFrame")){
      ocr_on_frame((Map) call.arguments, result);
    } else if(call.method.equals("closeOcrModel")){
      close_ocr_model(result);
    }else if(call.method.equals("loadYoloModel")){
      try {
        yolo = load_yolo_model((Map) call.arguments);
        result.success("ok");
      } catch (Exception e) {
        result.error("100","Error on load Yolov5 model", e);
      }
    }else if(call.method.equals("yoloOnFrame")){
      yolo_on_frame((Map) call.arguments, result);
    } else if(call.method.equals("yoloOnImage")){
      yolo_on_image((Map) call.arguments, result);
    } else if(call.method.equals("closeYoloModel")){
      close_yolo_model(result);
    } else if(call.method.equals("loadTesseractModel")){
      try {
        tesseract = load_tesseract_model((Map) call.arguments);
      } catch (Exception e) {
        result.error("100","Error on load tesseract model", e);
      }
    }else if(call.method.equals("tesseractOnImage")){
      tesseract_on_image((Map) call.arguments, result);
    } else if(call.method.equals("closeTesseractModel")){
      close_tesseract_model(result);
    }
    else {
      result.notImplemented();
    }
  }
  private void load_ocr_model(Map<String, Object> args) throws Exception {
    yolo = load_yolo_model(args);
    tesseract = load_tesseract_model(args);
  }

  private void ocr_on_frame(Map<String, Object> args, Result result){
    try {
      List<byte[]> image = (ArrayList) args.get("bytesList");
      int image_height = (int) args.get("image_height");
      int image_width = (int) args.get("image_width");
      float iou_threshold = (float)(double)( args.get("iou_threshold"));
      float conf_threshold = (float)(double)( args.get("conf_threshold"));
      float class_threshold = (float)(double)( args.get("class_threshold"));
      List<Integer> class_is_text = (List<Integer>) args.get("class_is_text");
      Bitmap bitmap = utils.feedInputToBitmap(context.getApplicationContext(),image,image_height, image_width, 90);
      int [] shape = yolo.getInputTensor().shape();
      ByteBuffer byteBuffer = utils.feedInputTensor(bitmap, shape[1], shape[2], image_width, image_height, 0,255);
      List<Map<String, Object>> yolo_results =  yolo.detectOnFrame(byteBuffer, image_height, image_width, iou_threshold, conf_threshold, class_threshold);
      for (Map<String, Object> yolo_result:yolo_results) {
        float [] box = (float[]) yolo_result.get("box");
        if(class_is_text.contains((int)box[5])){
          Bitmap crop = utils.crop_bitmap(bitmap,
                  box[0],box[1],box[2],box[3]);
          //utils.getScreenshotBmp(crop, "crop");
          Bitmap tmp = crop.copy(crop.getConfig(),crop.isMutable());
          yolo_result.put("text", tesseract.predict_text(tmp));
        }else{
          yolo_result.put("text", "");
        }
      }
      result.success(yolo_results);
    }catch (Exception e){
      result.error("100", "Ocr error", e);
    }
  }

  private void close_ocr_model(Result result){
    try {
      yolo.close();
      tesseract.close();
      result.success("OCR model closed succesfully");
    }catch (Exception e){
      result.error("100","Fail closed ocr model", e);
    }
  }

  private Yolo load_yolo_model(Map<String, Object> args) throws Exception {
    final String model = this.assets.getAssetFilePathByName(args.get("model_path").toString());
    final Object is_asset_obj = args.get("is_asset");
    final boolean is_asset = is_asset_obj==null?false:(boolean) is_asset_obj;
    final int num_threads = (int) args.get("num_threads");
    final boolean use_gpu = (boolean) args.get("use_gpu");
    final String label_path= this.assets.getAssetFilePathByName(args.get("label_path").toString());
    final int rotation= (int) args.get("rotation");
    final String version = args.get("model_version").toString();
    Yolo yolo = null;
    switch (version){
     case "yolov5":{
       yolo = new Yolov5(
               context,
               model,
               is_asset,
               num_threads,
               use_gpu,
               label_path,
               rotation);
       yolo.initialize_model();
       return yolo;
     }
      case "yolov8":{
        yolo = new Yolov8(
                context,
                model,
                is_asset,
                num_threads,
                use_gpu,
                label_path,
                rotation);
        yolo.initialize_model();
        return yolo;
      }
      default:{
        throw new Exception("Model version must be yolov5 or yolov8");
      }
    }

  }

  private void yolo_on_frame(Map<String, Object> args, Result result){
    Bitmap bitmap = null;
    try {
      List<byte[]> image = (ArrayList) args.get("bytesList");
      int image_height = (int) args.get("image_height");
      int image_width = (int) args.get("image_width");
      float iou_threshold = (float)(double)( args.get("iou_threshold"));
      float conf_threshold = (float)(double)( args.get("conf_threshold"));
      float class_threshold = (float)(double)( args.get("class_threshold"));
      //invert width with height, because android take a photo rotating 90 degrees
      bitmap = utils.feedInputToBitmap(context,image,image_height, image_width, 90);
      int [] shape = yolo.getInputTensor().shape();
      ByteBuffer byteBuffer = utils.feedInputTensor(bitmap, shape[1], shape[2], image_width, image_height, 0,255);
      result.success(yolo.detectOnFrame(byteBuffer, bitmap.getHeight(), bitmap.getWidth(), iou_threshold, conf_threshold, class_threshold));
    }catch (Exception e){
      result.error("100", "Detection Error", e);
    }finally {
      assert bitmap != null;
      if(!bitmap.isRecycled()){
        bitmap.recycle();
      }
    }
  }

  private void yolo_on_image(Map<String, Object> args, Result result){
    try {
      byte[] image = (byte[]) args.get("bytesList");
      int image_height = (int) args.get("image_height");
      int image_width = (int) args.get("image_width");
      float iou_threshold = (float)(double)( args.get("iou_threshold"));
      float conf_threshold = (float)(double)( args.get("conf_threshold"));
      float class_threshold = (float)(double)( args.get("class_threshold"));
      Bitmap bitmap = BitmapFactory.decodeByteArray(image, 0, image.length);
      int [] shape = yolo.getInputTensor().shape();
      ByteBuffer byteBuffer = utils.feedInputTensor(bitmap, shape[1], shape[2], image_width, image_height, 0,255);
      result.success(yolo.detectOnImage(byteBuffer, image_height, image_width, iou_threshold, conf_threshold, class_threshold));
    }catch (Exception e){
      result.error("100", "Detection Error", e);
    }
  }

  private void close_yolo_model(Result result){
    yolo.close();
    result.success("Yolo model closed succesfully");
  }

  private tesseract load_tesseract_model(Map<String, Object> args) throws Exception {
    final String tess_data = args.get("tess_data").toString();
    final Map<String,String> arg = (Map<String,String>) args.get("arg");
    final String language = args.get("language").toString();
    tesseract tss = new tesseract(tess_data, arg, language);
    tss.initialize_model();
    return tss;
  }

  private void tesseract_on_image(Map<String, Object> args, Result result){
    try {
      byte[] image = (byte[]) args.get("bytesList");
      int image_height = (int) args.get("image_height");
      int image_width = (int) args.get("image_width");
      float iou_threshold = (float)(double)( args.get("iou_threshold"));
      float conf_threshold = (float)(double)( args.get("conf_threshold"));
      List<Integer> class_is_text = (List<Integer>) args.get("class_is_text");
      String data = tesseract.predict_text(image);
      result.success(data);
    }catch (Exception e){
      result.error("100", "Prediction Error", e);
    }
  }

  private void close_tesseract_model(Result result){
    tesseract.close();
  }
}
