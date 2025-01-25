package com.vladih.computer_vision.flutter_vision;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.annotation.NonNull;

import com.vladih.computer_vision.flutter_vision.models.Yolo;
import com.vladih.computer_vision.flutter_vision.models.Yolov5;
import com.vladih.computer_vision.flutter_vision.models.Yolov8;
import com.vladih.computer_vision.flutter_vision.models.Yolov8Seg;
import com.vladih.computer_vision.flutter_vision.utils.utils;

import org.opencv.android.OpenCVLoader;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;
import io.flutter.plugin.common.MethodChannel.MethodCallHandler;
import io.flutter.plugin.common.MethodChannel.Result;
import io.flutter.embedding.engine.plugins.FlutterPlugin.FlutterAssets;

public class FlutterVisionPlugin implements FlutterPlugin, MethodCallHandler {
    private static final String CHANNEL_NAME = "flutter_vision";
    private MethodChannel methodChannel;
    private Context context;
    private FlutterAssets assets;
    private Yolo yolo_model;

    private ExecutorService executor;
    private boolean isDetecting = false;
    private static final ArrayList<Map<String, Object>> empty = new ArrayList<>();

    @Override
    public void onAttachedToEngine(@NonNull FlutterPluginBinding binding) {
        setupChannel(binding.getApplicationContext(), binding.getFlutterAssets(), binding.getBinaryMessenger());
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
        try {
            this.context = null;
            if (this.methodChannel != null) {
                this.methodChannel.setMethodCallHandler(null);
                this.methodChannel = null;
            }
            this.assets = null;
            close_yolo();
            if (this.executor != null) {
                this.executor.shutdownNow();
            }
        } catch (Exception e) {
            if (this.executor != null && !this.executor.isShutdown()) {
                this.executor.shutdownNow();
            }
        }
    }

    private void setupChannel(Context context, FlutterAssets assets, BinaryMessenger messenger) {
        OpenCVLoader.initDebug();
        this.assets = assets;
        this.context = context;
        this.methodChannel = new MethodChannel(messenger, CHANNEL_NAME);
        this.methodChannel.setMethodCallHandler(this);
        this.executor = Executors.newSingleThreadExecutor();
    }

    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull Result result) {
        try {
            if (call.method.equals("loadYoloModel")) {
                load_yolo_model((Map) call.arguments);
                result.success("ok");
            } else if (!isModelLoaded()) {
                result.error("MODEL_NOT_LOADED", "YOLO model not initialized", null);
            } else if (call.method.equals("yoloOnFrame")) {
                yolo_on_frame((Map) call.arguments, result);
            } else if (call.method.equals("yoloOnImage")) {
                yolo_on_image((Map) call.arguments, result);
            } else if (call.method.equals("closeYoloModel")) {
                close_yolo_model(result);
            } else {
                result.notImplemented();
            }
        } catch (Exception e) {
            result.error("OPERATION_FAILED", "Operation failed: " + e.getMessage(), null);
        }
    }

    // FlutterVisionPlugin.java
    private synchronized boolean isModelLoaded() {
        return yolo_model != null && yolo_model.isInitialized();
    }

    private void load_yolo_model(Map<String, Object> args) throws Exception {
        if (args == null) throw new Exception("Arguments cannot be null");

        String model = "";
        final Object modelPathObj = args.get("model_path");
        final Object labelPathObj = args.get("label_path");
        if (modelPathObj == null || labelPathObj == null) {
            throw new Exception("Missing model or label path");
        }

        final boolean is_asset = args.get("is_asset") != null && (boolean) args.get("is_asset");
        String label_path = "";

        if(is_asset){
            model = this.assets.getAssetFilePathByName(modelPathObj.toString());
            label_path = this.assets.getAssetFilePathByName(labelPathObj.toString());
        } else {
            model = modelPathObj.toString();
            label_path = labelPathObj.toString();
        }

        final int num_threads = (int) args.get("num_threads");
        final boolean quantization = (boolean) args.get("quantization");
        final boolean use_gpu = (boolean) args.get("use_gpu");
        final int rotation = (int) args.get("rotation");
        final String version = args.get("model_version").toString();

        switch (version) {
            case "yolov5":
                yolo_model = new Yolov5(
                        context,
                        model,
                        is_asset,
                        num_threads,
                        quantization,
                        use_gpu,
                        label_path,
                        rotation);
                break;
            case "yolov8":
                yolo_model = new Yolov8(
                        context,
                        model,
                        is_asset,
                        num_threads,
                        quantization,
                        use_gpu,
                        label_path,
                        rotation);
                break;
            case "yolov8seg":
                yolo_model = new Yolov8Seg(
                        context,
                        model,
                        is_asset,
                        num_threads,
                        quantization,
                        use_gpu,
                        label_path,
                        rotation);
                break;
            default:
                throw new Exception("Unsupported model version");
        }
        yolo_model.initialize_model();
    }

    class DetectionTask implements Runnable {
        private Yolo yolo;
        byte[] image;
        List<byte[]> frame;
        int image_height;
        int image_width;
        float iou_threshold;
        float conf_threshold;
        float class_threshold;
        String typing;
        private Result result;

        public DetectionTask(Yolo yolo, Map<String, Object> args, String typing, Result result) {
            this.typing = typing;
            this.yolo = yolo;
            if (args == null) {
                result.error("INVALID_ARGS", "Arguments cannot be null", null);
                return;
            }

            if (typing.equals("img")) {
                this.image = (byte[]) args.get("bytesList");
            } else {
                this.frame = (ArrayList) args.get("bytesList");
            }
            this.image_height = (int) args.get("image_height");
            this.image_width = (int) args.get("image_width");
            this.iou_threshold = (float) (double) (args.get("iou_threshold"));
            this.conf_threshold = (float) (double) (args.get("conf_threshold"));
            this.class_threshold = (float) (double) (args.get("class_threshold"));
            this.result = result;
        }

        @Override
        public void run() {
            try {
                if (yolo == null || !yolo.isInitialized()) {
                    throw new Exception("Model not initialized");
                }

                Bitmap bitmap;
                if (typing.equals("img")) {
                    if (image == null) throw new Exception("Invalid image data");
                    bitmap = BitmapFactory.decodeByteArray(image, 0, image.length);
                } else {
                    if (frame == null || frame.isEmpty()) throw new Exception("Invalid frame data");
                    bitmap = utils.feedInputToBitmap(context, frame, image_height, image_width, 90);
                }

                if (bitmap == null || bitmap.isRecycled()) {
                    throw new Exception("Failed to decode image");
                }

                int[] shape = yolo.getInputTensor().shape();
                int src_width = bitmap.getWidth();
                int src_height = bitmap.getHeight();
                ByteBuffer byteBuffer = utils.feedInputTensor(bitmap, shape[1], shape[2], src_width, src_height, 0, 255);
                List<Map<String, Object>> detections = yolo.detect_task(byteBuffer, src_height, src_width, iou_threshold, conf_threshold, class_threshold);
                result.success(detections != null ? detections : Collections.emptyList());
            } catch (Exception e) {
                result.error("DETECTION_FAILED", "Detection failed: " + e.getMessage(), null);
            } finally {
                isDetecting = false;
            }
        }
    }

    private synchronized void yolo_on_frame(Map<String, Object> args, Result result) {
        try {
            if (args == null || args.get("bytesList") == null) {
                result.error("INVALID_ARGS", "Missing frame data", null);
                return;
            }

            if (isDetecting) {
                result.success(empty);
            } else {
                isDetecting = true;
                DetectionTask detectionTask = new DetectionTask(yolo_model, args, "frame", result);
                executor.submit(detectionTask);
            }
        } catch (Exception e) {
            result.error("PROCESSING_ERROR", "Frame processing error: " + e.getMessage(), null);
        }
    }

    private synchronized void yolo_on_image(Map<String, Object> args, Result result) {
        try {
            if (args == null || args.get("bytesList") == null) {
                result.error("INVALID_ARGS", "Missing image data", null);
                return;
            }

            if (isDetecting) {
                result.success(empty);
            } else {
                isDetecting = true;
                DetectionTask detectionTask = new DetectionTask(yolo_model, args, "img", result);
                executor.submit(detectionTask);
            }
        } catch (Exception e) {
            result.error("PROCESSING_ERROR", "Image processing error: " + e.getMessage(), null);
        }
    }

    private void close_yolo_model(Result result) {
        try {
            close_yolo();
            result.success("YOLO model closed successfully");
        } catch (Exception e) {
            result.error("CLOSE_ERROR", "Error closing model: " + e.getMessage(), null);
        }
    }

    private void close_yolo() {
        if (yolo_model != null) {
            yolo_model.close();
            yolo_model = null;
        }
    }

}