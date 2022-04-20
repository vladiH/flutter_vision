package com.vladih.computer_vision.flutter_vision.models;

import android.graphics.Bitmap;

import com.vladih.computer_vision.flutter_vision.utils.responses;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import com.vladih.computer_vision.flutter_vision.utils.utils;

import io.flutter.embedding.engine.plugins.FlutterPlugin;

public class ocr {
    private yolov5 yolo_model;
    private tesseract tesseract_model;
    private final FlutterPlugin.FlutterPluginBinding binding;
    private final String model_path;
    private final boolean is_assets;
    private final int num_threads;
    private final boolean use_gpu;
    private final String label_path;
    private final float image_mean;
    private final float image_std;
    private final int rotation;
    private final boolean best;
    private final String tess_data;
    private final Map<String,String> arg;
    private final String language;

    public ocr(FlutterPlugin.FlutterPluginBinding binding,
               String model_path,
               boolean is_assets,
               int num_threads,
               boolean use_gpu,
               String label_path,
               float image_mean,
               float image_std,
               int rotation,
               boolean best,
               String tess_data,
               Map<String, String> arg, String language) {
        this.binding = binding;
        this.model_path = model_path;
        this.is_assets = is_assets;
        this.num_threads = num_threads;
        this.use_gpu = use_gpu;
        this.label_path = label_path;
        this.image_mean = image_mean;
        this.image_std = image_std;
        this.rotation = rotation;
        this.best = best;
        this.tess_data = tess_data;
        this.arg = arg;
        this.language = language;
    }
    public void close(){
        yolo_model.close();
        tesseract_model.close();
    }
    public void initialize_model() throws Exception {
        try{
            yolo_model = new yolov5(
                    this.binding,
                    this.model_path,
                    this.is_assets,
                    this.num_threads,
                    this.use_gpu,
                    this.label_path,
                    this.image_mean,
                    this.image_std,
                    this.rotation,
                    this.best
            );
            responses result = yolo_model.initialize_model();
            if(result.getType()!="success"){
                throw new Exception(result.getMessage());
            }
            tesseract_model = new tesseract(
                    this.tess_data,
                    this.arg,
                    this.language
            );
            result = tesseract_model.initialize_model();
            if(result.getType()!="success"){
                throw new Exception(result.getMessage());
            }
        } catch (Exception e){
            throw  new Exception("Unexpected error: "+e);
        }
    }

    public List<Map<String, Object>> predict(List<byte[]> image,
                        int image_height,
                        int image_width,
                        float iou_threshold,
                        float conf_threshold, List<Integer> class_is_text) throws Exception {
        try{
            List<Map<String, Object>> result = new ArrayList<>();
            List<float[]> yolo_result = yolo_model.detectOnFrame(image,image_height,image_width,
                                                            iou_threshold,conf_threshold);
            Bitmap bitmap=yolo_model.get_current_bitmap();
            Vector<String> labels = yolo_model.getLabels();
            //utils.getScreenshotBmp(bitmap, "current");
            for (float [] box:yolo_result) {
                Map<String, Object> output = new HashMap<>();
                String predict_text="None";
                Bitmap crop = utils.crop_bitmap(bitmap,
                        box[0],box[1],box[2],box[3]);
                //utils.getScreenshotBmp(crop, "crop");
                Bitmap tmp = crop.copy(crop.getConfig(),crop.isMutable());
                if(class_is_text.contains((int)box[5])){
                    predict_text = tesseract_model.predict_text(tmp);
                }
                output.put("yolo",box);
                output.put("image",utils.bitmap_to_byte(crop));
                output.put("prediction",predict_text);
                output.put("tag",labels.get((int)box[5]));
                result.add(output);
            }
            bitmap.recycle();
            return result;
        } catch (Exception e){
            //System.out.println(e.getStackTrace());
            throw  new Exception("Unexpected error: "+e.getMessage());
        }
    }
}
