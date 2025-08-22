package com.vladih.computer_vision.flutter_vision.models;

import android.content.Context;

public class Yolov5 extends Yolo{
    public Yolov5(Context context,
                  String model_path,
                  boolean is_assets,
                  int num_threads,
                  boolean quantization,
                  boolean use_gpu,
                  String label_path,
                  int rotation) {
        super(context, model_path, is_assets, num_threads, quantization, use_gpu, label_path, rotation);
    }
}
