package com.vladih.computer_vision.flutter_vision.models;

import static java.lang.Math.min;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Yolov5 extends Yolo{
    public Yolov5(Context context,
                  String model_path,
                  boolean is_assets,
                  int num_threads,
                  boolean use_gpu,
                  String label_path,
                  int rotation) {
        super(context, model_path, is_assets, num_threads, use_gpu, label_path, rotation);
    }
    @Override
    protected List<float[]>filter_box(float [][][] model_outputs, float iou_threshold,
                                          float conf_threshold, float modelx_size, float modely_size){
        try {
            List<float[]> pre_box = new ArrayList<>();
            int conf_index = 4;
            int class_index = 5;
            int dimension = model_outputs[0][0].length;
            int rows = model_outputs[0].length;
            float[] tmp = new float[7];
            float x1,y1,x2,y2,conf;
            for(int i=0; i<rows;i++){
                //if (model_outputs[0][i][class_index]<=conf_threshold) continue;
                //convert xywh to xyxy
                x1 = (model_outputs[0][i][0]-model_outputs[0][i][2]/2f)*modelx_size;
                y1 = (model_outputs[0][i][1]-model_outputs[0][i][3]/2f)*modely_size;
                x2 = (model_outputs[0][i][0]+model_outputs[0][i][2]/2f)*modelx_size;
                y2 = (model_outputs[0][i][1]+model_outputs[0][i][3]/2f)*modely_size;
                conf = model_outputs[0][i][conf_index];
                final float score = model_outputs[0][i][conf_index];
                for(int j=class_index;j<dimension;j++){
                    //change if result is poor
                    if(score<=conf_threshold) continue;
                    if (model_outputs[0][i][j]<conf_threshold) continue;
                    tmp[0]=x1;
                    tmp[1]=y1;
                    tmp[2]=x2;
                    tmp[3]=y2;
                    tmp[4]=conf;
                    tmp[5]=(j-class_index)*1f;
                    tmp[6]=score*model_outputs[0][i][j];
                    pre_box.add(tmp);
                }
            }
            if (pre_box.isEmpty()) return new ArrayList<>();
            //for reverse orden, insteand of using .reversed method
            Comparator<float []> compareValues = (v1, v2)->Float.compare(v2[6],v1[6]);
            //Collections.sort(pre_box,compareValues.reversed());
            Collections.sort(pre_box,compareValues);
            return nms(pre_box, iou_threshold);
        }catch (Exception e){
            Log.e("filter_box", e.getMessage());
            throw  e;
        }
    }
}
