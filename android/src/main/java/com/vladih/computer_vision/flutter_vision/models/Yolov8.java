package com.vladih.computer_vision.flutter_vision.models;

import static java.lang.Math.min;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.schema.Buffer;
import org.tensorflow.lite.schema.ReshapeOptions;
import org.tensorflow.lite.support.image.ImageProcessor;

import java.io.File;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public class Yolov8 extends Yolo{
    public Yolov8(Context context,
                  String model_path,
                  boolean is_assets,
                  int num_threads,
                  boolean use_gpu,
                  String label_path,
                  int rotation) {
        super(context, model_path, is_assets, num_threads, use_gpu, label_path, rotation);
    }
//    @Override
//    public List<Map<String, Object>> detectOnFrame(ByteBuffer byteBuffer,
//                                                   int image_height,
//                                                   int image_width,
//                                                   float iou_threshold,
//                                                   float conf_threshold) throws Exception {
//        try{
//            int[] shape = this.interpreter.getInputTensor(0).shape();
//            this.interpreter.run(byteBuffer, this.output);
//            List<float []> boxes = filter_box(this.output,iou_threshold,conf_threshold,shape[1],shape[2]);
//            boxes = restore_size(boxes, shape[1],shape[2],image_width,image_height);
//            return out(boxes, this.labels);
//        }catch (Exception e){
//            throw e;
//        }finally {
//            byteBuffer.clear();
//        }
//    }

    @Override
    protected List<float[]>filter_box(float [][][] model_outputs, float iou_threshold,
                                      float conf_threshold, float input_width, float input_height){
        try {
            //reshape [1,box+class,detected_box] to reshape [1,detected_box,box+class]
            model_outputs = reshape(model_outputs);
            List<float[]> pre_box = new ArrayList<>();
            int class_index = 4;
            int dimension = model_outputs[0][0].length;
            int rows = model_outputs[0].length;
            float[] tmp = new float[6];
            float x1,y1,x2,y2;
            for(int i=0; i<rows;i++){
                //convert xywh to xyxy
                x1 = (model_outputs[0][i][0]-model_outputs[0][i][2]/2f)*input_width;
                y1 = (model_outputs[0][i][1]-model_outputs[0][i][3]/2f)*input_height;
                x2 = (model_outputs[0][i][0]+model_outputs[0][i][2]/2f)*input_width;
                y2 = (model_outputs[0][i][1]+model_outputs[0][i][3]/2f)*input_height;
                for(int j=class_index;j<dimension;j++){
                    if (model_outputs[0][i][j]<conf_threshold) continue;
                    System.out.println("**********************");
                    System.out.println(model_outputs[0][i][0]);
                    System.out.println(model_outputs[0][i][1]);
                    System.out.println(model_outputs[0][i][2]);
                    System.out.println(model_outputs[0][i][3]);
                    System.out.println(model_outputs[0][i][j]);
                    System.out.println(x1);
                    System.out.println(y1);
                    System.out.println(x2);
                    System.out.println(y2);
                    System.out.println((j-class_index)*1f);
                    System.out.println("**********************");
                    tmp[0]=x1;
                    tmp[1]=y1;
                    tmp[2]=x2;
                    tmp[3]=y2;
                    tmp[4]=model_outputs[0][i][j];
                    tmp[5]=(j-class_index)*1f;
                    pre_box.add(tmp);
                }
            }
            if (pre_box.isEmpty()) return new ArrayList<>();
            //for reverse orden, insteand of using .reversed method
            Comparator<float []> compareValues = (v1, v2)->Float.compare(v2[4],v1[4]);
            //Collections.sort(pre_box,compareValues.reversed());
            Collections.sort(pre_box,compareValues);
            return nms(pre_box, iou_threshold);
        }catch (Exception e){
            throw  e;
        }
    }
    @Override
    protected List<Map<String, Object>>  out(List<float[]> yolo_result, Vector<String> labels){
        try {
            List<Map<String, Object>> result = new ArrayList<>();
            for (float [] box: yolo_result) {
                Map<String, Object> output = new HashMap<>();
                output.put("box",new float[]{box[0], box[1], box[2], box[3], box[4]}); //x1,y1,x2,y2,conf_class
                output.put("tag",labels.get((int)box[5]));
                result.add(output);
            }
            return result;
        }catch (Exception e){
            throw e;
        }
    }
}
