package com.vladih.computer_vision.flutter_vision;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;

import androidx.annotation.NonNull;

import com.vladih.computer_vision.flutter_vision.models.Tesseract;
import com.vladih.computer_vision.flutter_vision.models.Yolo;
import com.vladih.computer_vision.flutter_vision.models.Yolov8;
import com.vladih.computer_vision.flutter_vision.models.Yolov5;
import com.vladih.computer_vision.flutter_vision.models.Yolov8Seg;
import com.vladih.computer_vision.flutter_vision.utils.utils;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

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
    private Tesseract tesseract_model;

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
            close_tesseract();
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
        if (call.method.equals("loadOcrModel")) {
            try {
                load_ocr_model((Map) call.arguments);
            } catch (Exception e) {
                result.error("100", "Error on load ocr components", e);
            }
        } else if (call.method.equals("ocrOnFrame")) {
            ocr_on_frame((Map) call.arguments, result);
        } else if (call.method.equals("closeOcrModel")) {
            close_ocr_model(result);
        } else if (call.method.equals("loadYoloModel")) {
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
        } else if (call.method.equals("loadTesseractModel")) {
            try {
                load_tesseract_model((Map) call.arguments);
                result.success("ok");
            } catch (Exception e) {
                result.error("100", "Error on load Tesseract model", e);
            }
        } else if (call.method.equals("tesseractOnImage")) {
            tesseract_on_image((Map) call.arguments, result);
        } else if (call.method.equals("closeTesseractModel")) {
            close_tesseract_model(result);
        } else {
            result.notImplemented();
        }
    }

    private void load_ocr_model(Map<String, Object> args) throws Exception {
        load_yolo_model(args);
        load_tesseract_model(args);
    }

    private void ocr_on_frame(Map<String, Object> args, Result result) {
        try {
            List<byte[]> image = (ArrayList) args.get("bytesList");
            int image_height = (int) args.get("image_height");
            int image_width = (int) args.get("image_width");
            float iou_threshold = (float) (double) (args.get("iou_threshold"));
            float conf_threshold = (float) (double) (args.get("conf_threshold"));
            float class_threshold = (float) (double) (args.get("class_threshold"));
            List<Integer> class_is_text = (List<Integer>) args.get("class_is_text");
            Bitmap bitmap = utils.feedInputToBitmap(context.getApplicationContext(), image, image_height, image_width, 90);
            int[] shape = yolo_model.getInputTensor().shape();
            ByteBuffer byteBuffer = utils.feedInputTensor(bitmap, shape[1], shape[2], image_width, image_height, 0, 255);

            List<Map<String, Object>> yolo_results = yolo_model.detect_task(byteBuffer, image_height, image_width, iou_threshold, conf_threshold, class_threshold);
            for (Map<String, Object> yolo_result : yolo_results) {
                float[] box = (float[]) yolo_result.get("box");
                if (class_is_text.contains((int) box[5])) {
                    Bitmap crop = utils.crop_bitmap(bitmap,
                            box[0], box[1], box[2], box[3]);
                    //utils.getScreenshotBmp(crop, "crop");
                    Bitmap tmp = crop.copy(crop.getConfig(), crop.isMutable());
                    yolo_result.put("text", tesseract_model.predict_text(tmp));
                } else {
                    yolo_result.put("text", "");
                }
            }
            result.success(yolo_results);
        } catch (Exception e) {
            result.error("100", "Ocr error", e);
        }
    }

    private void close_ocr_model(Result result) {
        try {
            close_tesseract();
            close_yolo();
            result.success("OCR model closed succesfully");
        } catch (Exception e) {
            result.error("100", "Fail closed ocr model", e);
        }
    }

    private void load_yolo_model(Map<String, Object> args) throws Exception {
        final String model = this.assets.getAssetFilePathByName(args.get("model_path").toString());
        final Object is_asset_obj = args.get("is_asset");
        final boolean is_asset = is_asset_obj == null ? false : (boolean) is_asset_obj;
        final int num_threads = (int) args.get("num_threads");
        final boolean quantization = (boolean) args.get("quantization");
        final boolean use_gpu = (boolean) args.get("use_gpu");
        final String label_path = this.assets.getAssetFilePathByName(args.get("label_path").toString());
        final int rotation = (int) args.get("rotation");
        final String version = args.get("model_version").toString();
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
            if (typing == "img") {
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
                Bitmap bitmap;
                if (typing == "img") {
                    bitmap = BitmapFactory.decodeByteArray(image, 0, image.length);
                } else {
                    //rotate image, because android take a photo rotating 90 degrees
                    bitmap = utils.feedInputToBitmap(context, frame, image_height, image_width, 90);
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

    private void load_tesseract_model(Map<String, Object> args) throws Exception {
        final String tess_data = args.get("tess_data").toString();
        final Map<String, String> arg = (Map<String, String>) args.get("arg");
        final String language = args.get("language").toString();
        tesseract_model = new Tesseract(tess_data, arg, language);
        tesseract_model.initialize_model();
    }

    class PredictionTask implements Runnable {
        private Tesseract tesseract;
        private Bitmap bitmap;
        private Result result;

        public PredictionTask(Tesseract tesseract, Map<String, Object> args, Result result) {
            byte[] image = (byte[]) args.get("bytesList");
            this.tesseract = tesseract;
            this.bitmap = BitmapFactory.decodeByteArray(image, 0, image.length);
            this.result = result;
        }

        @Override
        public void run() {
            try {
                Mat mat = utils.rgbBitmapToMatGray(bitmap);
                double angle = utils.computeSkewAngle(mat.clone());
                mat = utils.deskew(mat, angle);
                mat = utils.filterTextFromImage(mat);
                bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(mat, bitmap);
//        utils.getScreenshotBmp(bitmap,"TESSEREACT");
                result.success(tesseract.predict_text(bitmap));
            } catch (Exception e) {
                result.error("100", "Prediction text Error", e);
            }
        }
    }

    private void tesseract_on_image(Map<String, Object> args, Result result) {
        try {
            PredictionTask predictionTask = new PredictionTask(tesseract_model, args, result);
            executor.submit(predictionTask);
        } catch (Exception e) {
            result.error("100", "Prediction Error", e);
        }
    }

    private void close_tesseract_model(Result result) {
        try {
            close_tesseract();
            result.success("Tesseract model closed succesfully");
        } catch (Exception e) {
            result.error("100", "close_tesseract_model error", e);
        }
    }

    private void close_tesseract(){
        if (tesseract_model != null) {
            tesseract_model.close();
            tesseract_model = null;
        }
    }

    private void close_yolo(){
        if (yolo_model != null) {
            yolo_model.close();
            yolo_model = null;
        }
    }
}
