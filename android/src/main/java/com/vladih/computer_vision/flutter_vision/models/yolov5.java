package com.vladih.computer_vision.flutter_vision.models;

import static android.content.ContentValues.TAG;
import static java.lang.Math.min;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
import android.util.Log;
import android.util.Size;

import com.vladih.computer_vision.flutter_vision.utils.responses;
import com.vladih.computer_vision.flutter_vision.utils.utils;

import org.opencv.core.Mat;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import io.flutter.embedding.engine.plugins.FlutterPlugin;

public class yolov5 {
    private static final int byte_per_channel = 4;
    private float [][][] output;
    private Interpreter interpreter;
    private final Vector<String> labels=new Vector<>();
    private final FlutterPlugin.FlutterPluginBinding binding;
    private final String model_path;
    private final boolean is_assets;
    private final int num_threads;
    private final boolean use_gpu;
    private final String label_path;
    private final float image_mean;
    private final float image_std;
    private final int rotation;
    private Bitmap rawBitmap;
    public yolov5(FlutterPlugin.FlutterPluginBinding context,
                  String model_path,
                  boolean is_assets,
                  int num_threads,
                  boolean use_gpu,
                  String label_path,
                  float image_mean,
                  float image_std,
                  int rotation) {
        this.binding = context;
        this.model_path = model_path;
        this.is_assets = is_assets;
        this.num_threads = num_threads;
        this.use_gpu = use_gpu;
        this.label_path = label_path;
        this.image_mean = image_mean;
        this.image_std = image_std;
        this.rotation = rotation;
    }
    public Interpreter getModel() {return this.interpreter;}
    public Bitmap get_current_bitmap(){return this.rawBitmap;}
    public Vector<String> getLabels(){return this.labels;}
    public void close(){
        if (this.rawBitmap!=null && !this.rawBitmap.isRecycled())
            this.rawBitmap.recycle();
        if (interpreter!=null)
            interpreter.close();
    }
    public responses initialize_model() throws IOException {
        AssetManager asset_manager = null;
        MappedByteBuffer buffer = null;
        FileChannel file_channel = null;
        FileInputStream input_stream = null;
        try {
            if (interpreter==null){
                if(is_assets){
                    asset_manager = binding.getApplicationContext().getAssets();
                    AssetFileDescriptor file_descriptor = asset_manager.openFd(
                            binding.getFlutterAssets().getAssetFilePathByName(this.model_path));
                    input_stream = new FileInputStream(file_descriptor.getFileDescriptor());
                    file_channel = input_stream.getChannel();
                    buffer = file_channel.map(
                            FileChannel.MapMode.READ_ONLY,file_descriptor.getStartOffset(),
                            file_descriptor.getLength()
                    );
                    file_descriptor.close();

                }else{
                    input_stream = new FileInputStream(new File(this.model_path));
                    file_channel = input_stream.getChannel();
                    buffer = file_channel.map(FileChannel.MapMode.READ_ONLY,0,file_channel.size());

                }
                CompatibilityList compatibilityList = new CompatibilityList();
                Interpreter.Options interpreterOptions = new Interpreter.Options();
                interpreterOptions.setNumThreads(num_threads);
                if(use_gpu){
                    if(compatibilityList.isDelegateSupportedOnThisDevice()){
                        GpuDelegate.Options gpuOptions = compatibilityList.getBestOptionsForThisDevice();
                        interpreterOptions.addDelegate(
                                new GpuDelegate(gpuOptions.setQuantizedModelsAllowed(true)));
                    }
                }
                interpreter = new Interpreter(buffer,interpreterOptions);
                if (is_assets){
                    load_labels(asset_manager,
                            binding.getFlutterAssets().getAssetFilePathByName(label_path));
                }else{
                    load_labels(null, label_path);
                }
            }
            return responses.success("Yolo model loaded, Success");
        }catch (Exception e){
            return  responses.error("Cannot initialize yolo model: "+e.getMessage());
        }finally {
            if (asset_manager!=null)
                asset_manager.close();
            if (buffer!=null)
                buffer.clear();
            if (file_channel!=null)
                if (file_channel.isOpen())
                    file_channel.close();
            if(file_channel!=null)
                if (file_channel.isOpen())
                    input_stream.close();
        }
    }
    private void load_labels(AssetManager asset_manager, String label_path) throws Exception {
        BufferedReader br=null;
        try {
            if(asset_manager!=null){
                br = new BufferedReader(new InputStreamReader(asset_manager.open(label_path)));
            }else{
                br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(label_path))));
            }
            String line;
            while ((line=br.readLine())!=null){
                labels.add(line);
            }
        }catch (Exception e){
            throw new Exception(e.getMessage());
        }finally {
            if (br != null) {
                br.close();
            }
        }
    }

    public List<float []> detectOnFrame(List<byte[]> image,
                                      int image_height,
                                      int image_width,
                                      float iou_threshold,
                                      float conf_threshold) throws Exception {
        this.rawBitmap=null;
        ByteBuffer byteBuffer=null;
        System.out.println("Detec on Frame Height and Width: " + image_height + "x" + image_width);
        try{
            Tensor tensor = this.interpreter.getInputTensor(0);
            int[] shape = tensor.shape();
            int inputSize = shape[1];
            System.out.println("Input Size: " + inputSize);
            //float gain = Math.min(inputSize/(float)image_width, inputSize/(float)image_height);
            //float padx = (inputSize-image_width*gain)/2;
            //float pady = (inputSize-image_height*gain)/2;
            this.rawBitmap = feedInputToBitmap(image, image_height, image_width, this.rotation);
            System.out.println("Raw Bitmap size: " + rawBitmap.getHeight() + "x" + rawBitmap.getWidth());
            byteBuffer = feedInputTensor(this.rawBitmap.copy(this.rawBitmap.getConfig(), this.rawBitmap.isMutable()), this.image_mean, this.image_std);
            this.output = new float[1][25200][this.labels.size()+5];
            this.interpreter.run(byteBuffer, this.output);
            List<float []> output = utils.filter_box(this.output,iou_threshold,conf_threshold,inputSize,inputSize);
            //invert width with height, because android take a photo rotating 90 degrees
            if(this.rotation==90){
                int aux = image_width;
                image_width = image_height;
                image_height = aux;
            }
            output = restore_size(output, inputSize,image_width,image_height);
            return output;
        }catch (Exception e){
            throw e;
        }finally {
            if(byteBuffer!=null)
                byteBuffer.clear();
        }
    }

    public List<float []> detectOnImage(byte[] image,
                                        int image_height,
                                        int image_width,
                                        float iou_threshold,
                                        float conf_threshold)  throws  Exception{
        ByteBuffer byteBuffer=null;
        try{
            this.rawBitmap = null;
            int[] inputShape = this.interpreter.getInputTensor(0).shape();
            int inputSize = inputShape[1];

            this.rawBitmap = BitmapFactory.decodeByteArray(image, 0, image.length);
            byteBuffer = feedInputTensor(this.rawBitmap.copy(this.rawBitmap.getConfig(), this.rawBitmap.isMutable()), this.image_mean, this.image_std);

            this.output = new float[1][25200][this.labels.size()+5];
            interpreter.run(byteBuffer, this.output);

            List<float []> output = utils.filter_box(this.output,iou_threshold,conf_threshold,inputSize,inputSize);

            output = restore_size(output, inputSize,image_width,image_height);
            return output;


        }catch (Exception e){
            System.out.println("Gerou exceção: " + e);

            throw e;
        }
    }

    /** Loads input image, and applies preprocessing. */
    private TensorImage loadImage(final Bitmap bitmapBuffer, int imageRotationDegrees, Size tfInputSize, TensorImage tfInputBuffer) {
        // Initializes preprocessor if null
        ImageProcessor tfImageProcessor =
                new ImageProcessor.Builder()
                        .add(
                                new ResizeOp(
                                        tfInputSize.getHeight(),
                                        tfInputSize.getWidth(),
                                        ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .build();

        tfInputBuffer.load(bitmapBuffer);
        return tfImageProcessor.process(tfInputBuffer);
    }

    private List<float[]>  restore_size(List<float[]> nms,
                                        int model_size,
                                        int src_image_width,
                                        int src_image_height){
        try{
            float gain = min(model_size/(float) src_image_width,model_size/(float) src_image_height);

            float padx = (model_size-src_image_width*gain)/2f;
            float pady = (model_size-src_image_height*gain)/2f;
            //System.out.println("////////////////RESTORE SIZE");
            //System.out.println(String.valueOf(src_image_width)+" "+String.valueOf(src_image_height));
            for(int i=0;i<nms.size();i++){
                nms.get(i)[0]= min(src_image_width, Math.max((nms.get(i)[0]-padx)/gain,0));
                nms.get(i)[1]= min(src_image_height, Math.max((nms.get(i)[1]-pady)/gain,0));
                nms.get(i)[2]= min(src_image_width, Math.max((nms.get(i)[2]-padx)/gain,0));
                nms.get(i)[3]= min(src_image_height, Math.max((nms.get(i)[3]-pady)/gain,0));
            }
            return  nms;
        }catch (Exception e){
            throw new RuntimeException(e.getMessage());
        }
    }




    private Bitmap feedInputToBitmap(List<byte[]> bytesList, int imageHeight, int imageWidth, int rotation) throws Exception {

        ByteBuffer Y = ByteBuffer.wrap(bytesList.get(0));
        ByteBuffer U = ByteBuffer.wrap(bytesList.get(1));
        ByteBuffer V = ByteBuffer.wrap(bytesList.get(2));

        int Yb = Y.remaining();
        int Ub = U.remaining();
        int Vb = V.remaining();

        byte[] data = new byte[Yb + Ub + Vb];
        Y.get(data, 0, Yb);
        V.get(data, Yb, Vb);
        U.get(data, Yb + Vb, Ub);

        return createBitmap(imageWidth, imageHeight, data, rotation);
    }

    private Bitmap createBitmap(int imageWidth, int imageHeight, byte[] data, int rotation) {
        Bitmap bitmapRaw = Bitmap.createBitmap(imageWidth, imageHeight, Bitmap.Config.ARGB_8888);
        Allocation bmData = renderScriptNV21ToRGBA888(
                this.binding.getApplicationContext(),
                imageWidth,
                imageHeight,
                data);
        bmData.copyTo(bitmapRaw);
        bmData.destroy();
        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);
        return bitmapRaw;
    }

    private Allocation renderScriptNV21ToRGBA888(android.content.Context context, int width, int height, byte[] nv21) {
        try{
            // https://stackoverflow.com/a/36409748
            RenderScript rs = RenderScript.create(context);
            ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

            Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
            Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

            Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
            Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

            in.copyFrom(nv21);

            yuvToRgbIntrinsic.setInput(in);
            yuvToRgbIntrinsic.forEach(out);

            rs.destroy();
            rs.finish();
            in.destroy();
            return out;
        }
        catch (Exception e){
            throw new RuntimeException("render script error: "+e);
        }
    }
    private ByteBuffer feedInputTensor(Bitmap bitmapRaw, float mean, float std) throws Exception {
        Tensor tensor = this.interpreter.getInputTensor(0);
        int[] shape = tensor.shape();
        int inputSize = shape[1];
        int inputChannels = shape[3];
        final boolean crop = false;
        int bytePerChannel = tensor.dataType() == DataType.UINT8 ? 1 : this.byte_per_channel;
        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * inputSize * inputSize * inputChannels * bytePerChannel);
        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();
        //TODO
        //utils.getScreenshotBmp(bitmapRaw, "antes");
        Bitmap bitmap = null;
        //Bitmap newBm = bitmapRaw;
        //720=720, 1280=width
        //640*640
        if (bitmapRaw.getWidth() != inputSize || bitmapRaw.getHeight() != inputSize) {
            Matrix matrix = getTransformationMatrix(bitmapRaw.getWidth(), bitmapRaw.getHeight(),
                    inputSize, inputSize, true, crop);
            if(!crop){
                //original size bitmap
                bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(),
                        matrix, true);
            }
            bitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888);

            final Canvas canvas = new Canvas(bitmap);

            //Draw background color
            Paint paint = new Paint();
            paint.setColor(Color.rgb(114, 114, 114));
            paint.setStyle(Paint.Style.FILL);
            canvas.drawRect(0, 0, canvas.getWidth(),    canvas.getHeight(), paint);

            //Determine the screen position
            float left = 0;
            float top = 0;
            if (bitmapRaw.getWidth() > bitmapRaw.getHeight()){
                top = (float)((bitmapRaw.getWidth() - bitmapRaw.getHeight()) / 2.0);
            }
            else{
                left = (float)((bitmapRaw.getHeight() - bitmapRaw.getWidth()) / 2.0);
            }
            canvas.drawBitmap( bitmapRaw, left , top, null );
            if (!bitmapRaw.isRecycled()){
                bitmapRaw.recycle();
            }
        }

        //utils.getScreenshotBmp(bitmap, "despues");
        if (tensor.dataType() == DataType.FLOAT32) {
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    int pixelValue = bitmap.getPixel(j, i);
                    if (inputChannels > 1){
                        imgData.putFloat((((pixelValue >> 16) & 0xFF) - mean) / std);//red
                        imgData.putFloat((((pixelValue >> 8) & 0xFF) - mean) / std);//green
                        imgData.putFloat(((pixelValue & 0xFF) - mean) / std);//blue
                    } else {
                        imgData.putFloat((((pixelValue >> 16 | pixelValue >> 8 | pixelValue) & 0xFF) - mean) / std);
                    }
                }
            }
        } else {
            //System.out.println("FLOAT16");
            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    int pixelValue = bitmap.getPixel(j, i);
                    if (inputChannels > 1){
                        imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                        imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                        imgData.put((byte) (pixelValue & 0xFF));
                    } else {
                        imgData.put((byte) ((pixelValue >> 16 | pixelValue >> 8 | pixelValue) & 0xFF));
                    }
                }
            }
        }
        if(!bitmap.isRecycled()){
            bitmap.recycle();
        }
        return imgData;
    }

    private static Matrix getTransformationMatrix(final int srcWidth,
                                                  final int srcHeight,
                                                  final int dstWidth,
                                                  final int dstHeight,
                                                  final boolean maintainAspectRatio,
                                                  final boolean crop) {
        final Matrix matrix = new Matrix();

        if (srcWidth != dstWidth || srcHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) srcWidth;
            final float scaleFactorY = dstHeight / (float) srcHeight;

            if (maintainAspectRatio && crop) {
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            }else if(maintainAspectRatio){
                final float scaleFactor = min(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            }
            else {
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        matrix.invert(new Matrix());
        return matrix;
    }
    public List<Map<String, Object>> predict(List<byte[]> image,
                                             int image_height,
                                             int image_width,
                                             float iou_threshold,
                                             float conf_threshold) throws Exception {
        try{
            List<Map<String, Object>> result = new ArrayList<>();
            List<float[]> yolo_result = detectOnFrame(image,image_height,image_width,
                    iou_threshold,conf_threshold);
            Bitmap bitmap=get_current_bitmap();
            Vector<String> labels = getLabels();
            //utils.getScreenshotBmp(bitmap, "current");
            for (float [] box:yolo_result) {
                Map<String, Object> output = new HashMap<>();
                Bitmap crop = utils.crop_bitmap(bitmap,
                        box[0],box[1],box[2],box[3]);
                //utils.getScreenshotBmp(crop, "crop");
                Bitmap tmp = crop.copy(crop.getConfig(),crop.isMutable());
                output.put("yolo",box);
                output.put("image",utils.bitmap_to_byte(crop));
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

    public List<Map<String, Object>> predictStaticImage(byte[] image,
                                             int image_height,
                                             int image_width,
                                             float iou_threshold,
                                             float conf_threshold) throws Exception {
        try{
            List<Map<String, Object>> result = new ArrayList<>();
            List<float[]> yolo_result = detectOnImage(image,image_height,image_width,
                    iou_threshold,conf_threshold);
            Bitmap bitmap=get_current_bitmap();
            Vector<String> labels = getLabels();
            //utils.getScreenshotBmp(bitmap, "current");
            for (float [] box:yolo_result) {
                Map<String, Object> output = new HashMap<>();
                Bitmap crop = utils.crop_bitmap(bitmap,
                        box[0],box[1],box[2],box[3]);
                utils.getScreenshotBmp(crop, "crop");
                Bitmap tmp = crop.copy(crop.getConfig(),crop.isMutable());
                output.put("yolo",box);
                output.put("image",utils.bitmap_to_byte(bitmap));
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
