package com.vladih.computer_vision.flutter_vision.models;

import static java.lang.Math.min;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import com.vladih.computer_vision.flutter_vision.utils.utils;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.schema.Buffer;
import org.tensorflow.lite.schema.ReshapeOptions;
import org.tensorflow.lite.support.image.ImageProcessor;

import java.io.File;
import java.io.FileInputStream;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.Vector;

public class Yolov8 extends Yolo {
    public Yolov8(Context context,
                  String model_path,
                  boolean is_assets,
                  int num_threads,
                  boolean quantization,
                  boolean use_gpu,
                  String label_path,
                  int rotation) {
        super(context, model_path, is_assets, num_threads, quantization, use_gpu, label_path, rotation);
    }

    @Override
    public List<Map<String, Object>> detect_task(ByteBuffer byteBuffer,
                                                 int source_height,
                                                 int source_width,
                                                 float iou_threshold,
                                                 float conf_threshold,
                                                 float class_threshold) throws Exception {
        try {
            int[] input_shape = this.interpreter.getInputTensor(0).shape();
            this.interpreter.run(byteBuffer, this.output);
            //INFO: output from detection model is not normalized
            List<float[]> boxes = filter_box(this.output, iou_threshold, conf_threshold,
                    class_threshold, input_shape[1], input_shape[2]);
            boxes = restore_size(boxes, input_shape[1], input_shape[2], source_width, source_height);
            return out(boxes, this.labels);
        } catch (Exception e) {
            throw e;
        } finally {
            byteBuffer.clear();
        }
    }

    @Override
    protected List<float[]> filter_box(float[][][] model_outputs, float iou_threshold,
                                       float conf_threshold, float class_threshold, float input_width, float input_height) {
        try {
            //model_outputs = [1,box+class,detected_box]
            List<float[]> pre_box = new ArrayList<>();
            int class_index = 4;
            int dimension = model_outputs[0][0].length;
            int rows = model_outputs[0].length;
            int max_index = 0;
            float max = 0f;
            for (int i = 0; i < dimension; i++) {
                float x1 = (model_outputs[0][0][i] - model_outputs[0][2][i] / 2f)* input_width;
                float y1 = (model_outputs[0][1][i] - model_outputs[0][3][i] / 2f)* input_height;
                float x2 = (model_outputs[0][0][i] + model_outputs[0][2][i] / 2f)* input_width;
                float y2 = (model_outputs[0][1][i] + model_outputs[0][3][i] / 2f)* input_height;

                max_index = class_index;
                max = model_outputs[0][max_index][i];

                for (int j = class_index + 1; j < rows; j++) {
                    float current = model_outputs[0][j][i];
                    if (current > max) {
                        max = current;
                        max_index = j;
                    }
                }

                if (max > class_threshold) {
                    float[] tmp = new float[6];
                    tmp[0] = x1;
                    tmp[1] = y1;
                    tmp[2] = x2;
                    tmp[3] = y2;
                    tmp[4] = max;
                    tmp[5] = (max_index - class_index) * 1f;
                    pre_box.add(tmp);
                }
            }
            if (pre_box.isEmpty()) return new ArrayList<>();
            //for reverse orden, insteand of using .reversed method
            Comparator<float[]> compareValues = (v1, v2) -> Float.compare(v2[4], v1[4]);
            //Collections.sort(pre_box,compareValues.reversed());
            Collections.sort(pre_box, compareValues);
            return nms(pre_box, iou_threshold);
        } catch (Exception e) {
            throw e;
        }
    }

    @Override
    protected List<Map<String, Object>> out(List<float[]> yolo_result, Vector<String> labels) {
        try {
            List<Map<String, Object>> result = new ArrayList<>();
            for (float[] box : yolo_result) {
                Map<String, Object> output = new HashMap<>();
                output.put("box", new float[]{box[0], box[1], box[2], box[3], box[4]}); //x1,y1,x2,y2,conf_class
                output.put("tag", labels.get((int) box[5]));
                result.add(output);
            }
            return result;
        } catch (Exception e) {
            throw e;
        }
    }
}
