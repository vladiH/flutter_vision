package com.vladih.computer_vision.flutter_vision;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.annotation.NonNull;

import com.vladih.computer_vision.flutter_vision.models.Yolo;
import com.vladih.computer_vision.flutter_vision.models.Yolov8;
import com.vladih.computer_vision.flutter_vision.models.Yolov5;
import com.vladih.computer_vision.flutter_vision.models.Yolov8Seg;
import com.vladih.computer_vision.flutter_vision.utils.utils;

import org.opencv.android.OpenCVLoader;

import java.nio.ByteBuffer;
import java.util.ArrayList;
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

/**
 * FlutterVisionPlugin
 */
public class FlutterVisionPlugin implements FlutterPlugin, MethodCallHandler {
    private static final String CHANNEL_NAME = "flutter_vision";
    private MethodChannel methodChannel;
    private Context context;
    private FlutterAssets assets;
    private Yolo yolo_model;

    private ExecutorService executor;

    private boolean isDetecting = false;

    private static ArrayList<Map<String, Object>> empty = new ArrayList<>();

    @Override
    public void onAttachedToEngine(@NonNull FlutterPluginBinding binding) {
        setupChannel(binding.getApplicationContext(), binding.getFlutterAssets(), binding.getBinaryMessenger());
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
        try {
            this.context = null;
            this.methodChannel.setMethodCallHandler(null);
            this.methodChannel = null;
            this.assets = null;
            close_yolo();
            this.executor.shutdownNow();
        } catch (Exception e) {
            if (!this.executor.isShutdown()) {
                this.executor.shutdownNow();
            }
//            System.out.println(e.getMessage());
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
        // Handle method calls from Flutter
        if (call.method.equals("loadYoloModel")) {
            try {
                load_yolo_model((Map) call.arguments);
                result.success("ok");
            } catch (Exception e) {
                result.error("100", "Error on load Yolov5 model", e);
            }
        } else if (call.method.equals("yoloOnFrame")) {
            yolo_on_frame((Map) call.arguments, result);
        } else if (call.method.equals("yoloOnImage")) {
            yolo_on_image((Map) call.arguments, result);
        } else if (call.method.equals("closeYoloModel")) {
            close_yolo_model(result);
        }else {
            result.notImplemented();
        }
    }

    private void load_yolo_model(Map<String, Object> args) throws Exception {
        String model = "";
        final Object is_asset_obj = args.get("isAsset"); 
        final boolean is_asset = is_asset_obj == null ? false : (boolean) is_asset_obj;
        String label_path = "";
        
        // CORREGIDO: Usar los nombres correctos que envía Dart
        if(is_asset){
            model = this.assets.getAssetFilePathByName(args.get("modelPath").toString());
            label_path = this.assets.getAssetFilePathByName(args.get("labels").toString());
        }else{
            model = args.get("modelPath").toString();
            label_path = args.get("labels").toString();
        }
        
        // CORREGIDO: Usar los nombres correctos que envía Dart
        final int num_threads = (int) args.get("numThreads");
        final boolean quantization = (boolean) args.get("quantization");
        final boolean use_gpu = (boolean) args.get("useGpu");
        final int rotation = args.get("rotation") != null ? (int) args.get("rotation") : 0;
        final String version = args.get("modelVersion").toString();
        
        switch (version) {
            case "yolov5": {
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
            }
            case "yolov8": {
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
            }

            case "yolov8seg": {
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
            }
            default: {
                throw new Exception("Model version must be yolov5, yolov8 or yolov8seg");
            }
        }
        yolo_model.initialize_model();
    }

    //https://www.baeldung.com/java-single-thread-executor-service
    class DetectionTask implements Runnable {
        //    private static volatile DetectionTasks instance;
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
            if (typing.equals("img")) {
                this.image = (byte[]) args.get("bytesList");
            } else {
                this.frame = (ArrayList) args.get("bytesList");
            }
            // CORREGIDO: Usar los nombres correctos que envía Dart
            this.image_height = (int) args.get("imageHeight");
            this.image_width = (int) args.get("imageWidth");
            this.iou_threshold = (float) (double) (args.get("iouThreshold"));
            this.conf_threshold = (float) (double) (args.get("confThreshold"));
            this.class_threshold = (float) (double) (args.get("classThreshold"));
            this.result = result;
        }
        
        @Override
        public void run() {
            try {
                Bitmap bitmap;
                if (typing.equals("img")) {
                    bitmap = BitmapFactory.decodeByteArray(image, 0, image.length);
                } else {
                    //rotate image, because android take a photo rotating 90 degrees
                    bitmap = utils.feedInputToBitmap(context, frame, image_height, image_width, yolo.getRotation());
                }
                int[] shape = yolo.getInputTensor().shape();
                int src_width = bitmap.getWidth();
                int src_height = bitmap.getHeight();
                ByteBuffer byteBuffer = utils.feedInputTensor(bitmap, shape[1], shape[2], src_width, src_height, 0, 255);
                List<Map<String, Object>> detections = yolo.detect_task(byteBuffer, src_height, src_width, iou_threshold, conf_threshold, class_threshold);
                isDetecting = false;
                result.success(detections);
            } catch (Exception e) {
                result.error("100", "Detection Error", e);
            }
        }
    }

    private void yolo_on_frame(Map<String, Object> args, Result result) {
        try {
            if (isDetecting) {
                result.success(empty);
            } else {
                isDetecting = true;
                DetectionTask detectionTask = new DetectionTask(yolo_model, args, "frame", result);
                executor.submit(detectionTask);
            }
        } catch (Exception e) {
            result.error("100", "Detection Error", e);
        }
    }

    private void yolo_on_image(Map<String, Object> args, Result result) {
        try {
            if (isDetecting) {
                result.success(empty);
            } else {
                isDetecting = true;
                DetectionTask detectionTask = new DetectionTask(yolo_model, args, "img", result);
                executor.submit(detectionTask);
            }
        } catch (Exception e) {
            result.error("100", "Detection Error", e);
        }
    }

    private void close_yolo_model(Result result) {
        try {
            close_yolo();
            result.success("Yolo model closed succesfully");
        } catch (Exception e) {
            result.error("100", "Close_yolo_model error", e);
        }
    }

    private void close_yolo(){
        if (yolo_model != null) {
            yolo_model.close();
            yolo_model = null;
        }
    }
}