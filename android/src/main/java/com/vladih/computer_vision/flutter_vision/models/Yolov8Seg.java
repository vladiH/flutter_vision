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

//https://dev.to/andreygermanov/how-to-implement-instance-segmentation-using-yolov8-neural-network-3if9
//PAPER: https://openaccess.thecvf.com/content_ICCV_2019/papers/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.pdf
public class Yolov8Seg extends Yolo {
    public Yolov8Seg(Context context,
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
                                                 float class_threshold) {
        try {
            if (has_multiple_output()) {
                Map<Integer, Object> outputs = new HashMap<>();
                for (int i = 0; i < interpreter.getOutputTensorCount(); i++) {
                    int[] shape = interpreter.getOutputTensor(i).shape();
                    outputs.put(i, Array.newInstance(float.class, shape));
                }
                Object[] inputs = {byteBuffer};
                this.interpreter.runForMultipleInputsOutputs(inputs, outputs);

                int[] input_shape = interpreter.getInputTensor(0).shape(); // 1, 640, 640
                int[] output0_shape = interpreter.getOutputTensor(0).shape(); //1,116,2184
                int[] output1_shape = interpreter.getOutputTensor(1).shape(); //1,160,160,32

                float[][][] output0 = (float[][][]) outputs.get(0);

                //seg_boxes = coordinates[4]+classes[x=84]+masks_weight[32]
                //INFO: output from segment model return normalized values
                List<float[]> seg_boxes = filter_box(output0,
                        iou_threshold, conf_threshold, class_threshold,
                        input_shape[1], input_shape[2]);

                output0 = null;

                //it only restores the size of the boxes, nothing has been done with mask_weight
                seg_boxes = restore_size(seg_boxes, input_shape[1], input_shape[2],
                        source_width, source_height);

                float[][][][] masks = (float[][][][]) outputs.get(1);
                List<int[]> seg_boxes_mask = new ArrayList<>();
                for (float[] mask_weight : seg_boxes) {
                    seg_boxes_mask.add(compute_mask(mask_weight,
                            masks[0], (float) 0.3,
                            output1_shape[1], output1_shape[2]));
                }
                masks = null;
                List<List<Map<String, Double>>> restore_seg_mask = restore_seg_mask_size(seg_boxes,
                        seg_boxes_mask, output1_shape[1], output1_shape[2], source_height, source_width
                );
                return out_segmentation(seg_boxes, restore_seg_mask, this.labels);
            } else {
                throw new ExceptionInInitializerError("tflite model should have two outputs in segmentation mode");
            }
        } catch (Exception e) {
            throw e;
        } finally {
            byteBuffer.clear();
        }
    }

    private int[] compute_mask(float[] mask_weight,
                               float[][][] masks_protos,
                               float seg_thresh,
                               int mask_height,
                               int mask_width) {
        int prefix_box = 6;
        int numMask = mask_weight.length - prefix_box;
        int[] masks = new int[mask_height * mask_width];
        int index = 0;
        // Set all pixels to either white (255) or black (0)
        for (int h = 0; h < mask_height; h++) {
            for (int w = 0; w < mask_width; w++) {
                float sum = 0.0f;
                for (int j = 0; j < numMask; j++) {
                    sum += mask_weight[j + prefix_box] * masks_protos[h][w][j];
                }
                if (sigmoid(sum) > seg_thresh) {
                    masks[index++] = Color.WHITE;
                } else {
                    masks[index++] = Color.BLACK;
                }
            }
//            System.out.println();
        }
//        Bitmap bitmap =Bitmap.createBitmap(masks, maskHeight, maskWidth, Bitmap.Config.ARGB_8888);
//        utils.getScreenshotBmp(bitmap, UUID.randomUUID().toString());
        return masks;
    }

    float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private List<List<Map<String, Double>>> restore_seg_mask_size(List<float[]> boxes, List<int[]> seg_mask,
                                                                  int mask_height, int mask_width,
                                                                  int source_height, int source_width) {
        Bitmap bitmap = null;
        Bitmap crop = null;
        try {
            List<List<Map<String, Double>>> polygons = new ArrayList<>();
            for (int i = 0; i < boxes.size(); i++) {
                // Set the pixel data from the flattened array
                bitmap = Bitmap.createBitmap(seg_mask.get(i), mask_width, mask_height, Bitmap.Config.ARGB_8888);
//            String tag = UUID.randomUUID().toString();
//            utils.getScreenshotBmp(bitmap, tag+"0");
                crop = utils.crop_bitmap(bitmap,
                        min(mask_width, Math.max(boxes.get(i)[0] * mask_width / source_width, 0)),
                        min(mask_height, Math.max(boxes.get(i)[1] * mask_height / source_height, 0)),
                        min(mask_width, Math.max(boxes.get(i)[2] * mask_width / source_width, 0)),
                        min(mask_height, Math.max(boxes.get(i)[3] * mask_height / source_height, 0))
                );
//            utils.getScreenshotBmp(crop, tag+"1");
                List<Map<String, Double>> crop_polygon = get_polygons_from_bitmap(crop, mask_height,
                        mask_width, source_height, source_width);
                polygons.add(crop_polygon);
            }
            return polygons;
        } catch (Exception e) {
            throw e;
        } finally {
            if (bitmap != null) bitmap.recycle();
            if (crop != null) crop.recycle();
        }
    }

    public static List<Map<String, Double>> get_polygons_from_bitmap(Bitmap mask,
                                                                           int mask_height,
                                                                           int mask_width,
                                                                           int source_height,
                                                                           int source_width) {
        Mat maskMat = utils.rgbBitmapToMatGray(mask); // Convert Bitmap to Mat
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(maskMat, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        MatOfPoint largestContour = null;
        double largestArea = 0;

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > largestArea) {
                largestArea = area;
                largestContour = contour;
            }
        }
        List<Point> polygon = new ArrayList<>(largestContour.toList());
//        List<List<Point>> polygons = new ArrayList<>();
//        for (MatOfPoint contour : contours) {
//            List<Point> polygon = new ArrayList<>();
//            for (Point point : contour.toList()) {
//                polygon.add(point);
//            }
//            polygons.add(polygon);
//        }
//        List<List<Map<String, Double>>> converted_polygons = new ArrayList<>();

//        for (List<Point> polygon : polygons) {
            List<Map<String, Double>> convertedPolygon = new ArrayList<>();
            for (Point point : polygon) {
                Map<String, Double> pointMap = new HashMap<>();
                pointMap.put("x", point.x * source_width / mask_width);
                pointMap.put("y", point.y * source_height / mask_height);
                convertedPolygon.add(pointMap);
            }
//            converted_polygons.add(convertedPolygon);
//        }
//        return converted_polygons;
        return convertedPolygon;
    }

    private boolean has_multiple_output() {
        return this.interpreter.getOutputTensorCount() > 1;
    }

    @Override
    protected List<float[]> filter_box(float[][][] model_outputs, float iou_threshold,
                                       float conf_threshold, float class_threshold,
                                       float input_width, float input_height) {
        try {
            //model_outputs = [1,box+class+mask_weight,detected_box]
            List<float[]> pre_box = new ArrayList<>();
            int class_index = 4;
            int dimension = model_outputs[0][0].length;
            int rows = model_outputs[0].length;
            int index_mask = rows - 32;
            float[] mask_weight = new float[32];
            int max_index = 0;
            float max = 0f;
            for (int i = 0; i < dimension; i++) {
                // Convertir xywh a xyxy y ajustar por el ancho y alto de entrada
                float x1 = (model_outputs[0][0][i] - model_outputs[0][2][i] / 2f) * input_width;
                float y1 = (model_outputs[0][1][i] - model_outputs[0][3][i] / 2f) * input_height;
                float x2 = (model_outputs[0][0][i] + model_outputs[0][2][i] / 2f) * input_width;
                float y2 = (model_outputs[0][1][i] + model_outputs[0][3][i] / 2f) * input_height;

                max_index = class_index;
                max = model_outputs[0][max_index][i];

                for (int j = class_index + 1; j < index_mask; j++) {
                    float current = model_outputs[0][j][i];
                    if (current > max) {
                        max = current;
                        max_index = j;
                    }
                }

                if (max > class_threshold) {
                    float[] tmp = new float[38];
                    tmp[0] = x1;
                    tmp[1] = y1;
                    tmp[2] = x2;
                    tmp[3] = y2;
                    tmp[4] = max;
                    tmp[5] = (max_index - class_index) * 1f;
                    for (int j = index_mask; j < rows; j++) {
                        tmp[j - index_mask + 6] = model_outputs[0][j][i];
                    }
                    pre_box.add(tmp);
                }
            }
            if (pre_box.isEmpty()) return new ArrayList<>();
            //for reverse orden, insteand of using .reversed method
            Comparator<float[]> compareValues = (v1, v2) -> Float.compare(v2[4], v1[4]);
            //Collections.sort(pre_box,compareValues.reversed());
            Collections.sort(pre_box, compareValues);
            return nms(pre_box, iou_threshold);
//            return nms_segmentation(pre_box, iou_threshold);
        } catch (Exception e) {
            throw e;
        }
    }

    //ignore out method of super class
    protected List<Map<String, Object>> out_segmentation(List<float[]> yolo_result,
                                                         List<List<Map<String, Double>>> polygons,
                                                         Vector<String> labels) {
        try {
            List<Map<String, Object>> result = new ArrayList<>();
            for (int i = 0; i < yolo_result.size(); i++) {
                Map<String, Object> output = new HashMap<>();
                output.put("box", new float[]{yolo_result.get(i)[0], yolo_result.get(i)[1],
                        yolo_result.get(i)[2], yolo_result.get(i)[3], yolo_result.get(i)[4]}); //x1,y1,x2,y2,conf_class
                output.put("polygons", polygons.get(i));
                output.put("tag", labels.get((int) yolo_result.get(i)[5]));
                result.add(output);
            }
            return result;
        } catch (Exception e) {
            throw e;
        }
    }
}
