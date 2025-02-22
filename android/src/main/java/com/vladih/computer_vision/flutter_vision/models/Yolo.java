package com.vladih.computer_vision.flutter_vision.models;

import static java.lang.Math.min;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;

import com.vladih.computer_vision.flutter_vision.utils.FeedInputTensorHelper;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
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
import java.util.Vector;

import io.flutter.embedding.engine.FlutterEngine;
import io.flutter.embedding.engine.plugins.FlutterPlugin;

public class Yolo {
    protected float[][][] output;
    protected Interpreter interpreter;
    protected Vector<String> labels;
    protected final Context context;
    protected final String model_path;
    protected final boolean is_assets;
    protected final int num_threads;
    protected final boolean quantization;
    protected final boolean use_gpu;
    protected final String label_path;
    protected final int rotation;

    public Yolo(Context context,
                String model_path,
                boolean is_assets,
                int num_threads,
                boolean quantization,
                boolean use_gpu,
                String label_path,
                int rotation) {
        this.context = context;
        this.model_path = model_path;
        this.is_assets = is_assets;
        this.num_threads = num_threads;
        this.quantization = quantization;
        this.use_gpu = use_gpu;
        this.label_path = label_path;
        this.rotation = rotation;
    }

    //    public Vector<String> getLabels(){return this.labels;}
    public Tensor getInputTensor() {
        return this.interpreter.getInputTensor(0);
    }

    @SuppressLint("SuspiciousIndentation")
    public void initialize_model() throws Exception {
        AssetManager asset_manager = null;
        MappedByteBuffer buffer = null;
        FileChannel file_channel = null;
        FileInputStream input_stream = null;

        try {
            if (is_assets) {
                asset_manager = context.getAssets();
                AssetFileDescriptor file_descriptor = asset_manager.openFd(this.model_path);
                input_stream = new FileInputStream(file_descriptor.getFileDescriptor());

                file_channel = input_stream.getChannel();
                buffer = file_channel.map(
                        FileChannel.MapMode.READ_ONLY, file_descriptor.getStartOffset(),
                        file_descriptor.getLength()
                );
                file_descriptor.close();
            } else {
                input_stream = new FileInputStream(new File(this.model_path));
                file_channel = input_stream.getChannel();
                buffer = file_channel.map(FileChannel.MapMode.READ_ONLY, 0, file_channel.size());
            }

            Interpreter.Options interpreterOptions = new Interpreter.Options();
            try {
                // Check if GPU support is available
                CompatibilityList compatibilityList = new CompatibilityList();
                if (use_gpu && compatibilityList.isDelegateSupportedOnThisDevice()) {
                    GpuDelegateFactory.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
                    GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions.setQuantizedModelsAllowed(this.quantization));
                    interpreterOptions.addDelegate(gpuDelegate);
                } else {
                    interpreterOptions.setNumThreads(num_threads);
                }
                // Create the interpreter
                this.interpreter = new Interpreter(buffer, interpreterOptions);
            } catch (Exception e) {
                interpreterOptions = new Interpreter.Options();
                interpreterOptions.setNumThreads(num_threads);
                // Create the interpreter
                this.interpreter = new Interpreter(buffer, interpreterOptions);
            }
            this.interpreter.allocateTensors();
            this.labels = load_labels(asset_manager, label_path);
            int[] shape = interpreter.getOutputTensor(0).shape();//3dimension
            this.output = (float [][][]) Array.newInstance(float.class, shape);
        } catch (Exception e) {
            throw e;
        } finally {
            if (buffer != null)
                buffer.clear();
            if (file_channel != null && file_channel.isOpen()) {
                file_channel.close();
                input_stream.close();
            }
        }
    }

    protected Vector<String> load_labels(AssetManager asset_manager, String label_path) throws Exception {
        BufferedReader br = null;
        try {
            if (asset_manager != null) {
                br = new BufferedReader(new InputStreamReader(asset_manager.open(label_path)));
            } else {
                br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(label_path))));
            }
            String line;
            Vector<String> labels = new Vector<>();
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            return labels;
        } catch (Exception e) {
            throw new Exception(e.getMessage());
        } finally {
            if (br != null) {
                br.close();
            }
        }
    }

    public List<Map<String, Object>> detect_task(ByteBuffer byteBuffer,
                                                 int source_height,
                                                 int source_width,
                                                 float iou_threshold,
                                                 float conf_threshold, float class_threshold) throws Exception {
        try {
            int[] input_shape = this.interpreter.getInputTensor(0).shape();
            this.interpreter.run(byteBuffer, this.output);
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

    protected List<float[]> filter_box(float[][][] model_outputs, float iou_threshold,
                                       float conf_threshold, float class_threshold, float input_width, float input_height) {
        try {
            //model_outputs = [1,box+model_conf+class,detected_box]
            List<float[]> pre_box = new ArrayList<>();
            int conf_index = 4;
            int class_index = 5;
            int dimension = model_outputs[0][0].length;
            int rows = model_outputs[0].length;
            float x1, y1, x2, y2, conf;
            int max_index = 0;
            float max = 0f;
            for (int i = 0; i < rows; i++) {
                //convert xywh to xyxy
                x1 = (model_outputs[0][i][0] - model_outputs[0][i][2] / 2f) * input_width;
                y1 = (model_outputs[0][i][1] - model_outputs[0][i][3] / 2f) * input_height;
                x2 = (model_outputs[0][i][0] + model_outputs[0][i][2] / 2f) * input_width;
                y2 = (model_outputs[0][i][1] + model_outputs[0][i][3] / 2f) * input_height;
                conf = model_outputs[0][i][conf_index];
                if (conf < conf_threshold) continue;

                max_index = class_index;
                max = model_outputs[0][i][max_index];

                for (int j = class_index + 1; j < dimension; j++) {
                    float current = model_outputs[0][i][j];
                    if (current > max) {
                        max = current;
                        max_index = j;
                    }
                }
                if (max > class_threshold){
                    float[] tmp = new float[6];
                    tmp[0] = x1;
                    tmp[1] = y1;
                    tmp[2] = x2;
                    tmp[3] = y2;
                    tmp[4] = model_outputs[0][i][max_index];
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

    protected static List<float[]> nms(List<float[]> boxes, float iou_threshold) {
        try {
            List<float[]> filteredBoxes = new ArrayList<>(boxes); // Create a copy of the input list

            for (int i = 0; i < filteredBoxes.size(); i++) {
                float[] box = filteredBoxes.get(i);
                for (int j = i + 1; j < filteredBoxes.size(); j++) {
                    float[] next_box = filteredBoxes.get(j);
                    float x1 = Math.max(next_box[0], box[0]);
                    float y1 = Math.max(next_box[1], box[1]);
                    float x2 = Math.min(next_box[2], box[2]);
                    float y2 = Math.min(next_box[3], box[3]);

                    float width = Math.max(0, x2 - x1);
                    float height = Math.max(0, y2 - y1);

                    float intersection = width * height;
                    float union = (next_box[2] - next_box[0]) * (next_box[3] - next_box[1])
                            + (box[2] - box[0]) * (box[3] - box[1]) - intersection;
                    float iou = intersection / union;
                    if (iou > iou_threshold) {
                        filteredBoxes.remove(j);
                        j--;
                    }
                }
            }
            return filteredBoxes;
        } catch (Exception e) {
            Log.e("nms", e.getMessage());
            throw e;
        }
    }

    protected List<float[]> restore_size(List<float[]> nms,
                                         int input_width,
                                         int input_height,
                                         int src_width,
                                         int src_height) {
        try {
            //restore size after scaling, larger images
            if (src_width > input_width || src_height > input_height) {
                float gainx = src_width / (float) input_width;
                float gainy = src_height / (float) input_height;
                for (int i = 0; i < nms.size(); i++) {
                    nms.get(i)[0] = min(src_width, Math.max(nms.get(i)[0] * gainx, 0));
                    nms.get(i)[1] = min(src_height, Math.max(nms.get(i)[1] * gainy, 0));
                    nms.get(i)[2] = min(src_width, Math.max(nms.get(i)[2] * gainx, 0));
                    nms.get(i)[3] = min(src_height, Math.max(nms.get(i)[3] * gainy, 0));
                }
                //restore size after padding, smaller images
            } else {
                float padx = (src_width - input_width) / 2f;
                float pady = (src_height - input_height) / 2f;
                for (int i = 0; i < nms.size(); i++) {
                    nms.get(i)[0] = min(src_width, Math.max(nms.get(i)[0] + padx, 0));
                    nms.get(i)[1] = min(src_height, Math.max(nms.get(i)[1] + pady, 0));
                    nms.get(i)[2] = min(src_width, Math.max(nms.get(i)[2] + padx, 0));
                    nms.get(i)[3] = min(src_height, Math.max(nms.get(i)[3] + pady, 0));
                }
            }
            return nms;
        } catch (Exception e) {
            throw new RuntimeException(e.getMessage());
        }
    }
    protected List<Map<String, Object>> out(List<float[]> yolo_result, Vector<String> labels) {
        try {
            List<Map<String, Object>> result = new ArrayList<>();
            //utils.getScreenshotBmp(bitmap, "current");
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

    public void close() {
        try {
            if (interpreter != null)
                interpreter.close();
        } catch (Exception e) {
            throw e;
        }
    }
}
