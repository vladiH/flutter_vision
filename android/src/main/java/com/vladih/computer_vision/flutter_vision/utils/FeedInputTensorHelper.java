package com.vladih.computer_vision.flutter_vision.utils;

import android.graphics.Bitmap;

import com.googlecode.leptonica.android.Scale;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

public class FeedInputTensorHelper {
    private static FeedInputTensorHelper instance;
    private TensorImage tensorImage;
    private ImageProcessor downSizeImageProcessor;
    private ImageProcessor upSizeImageProcessor;

    private int previus_width = 0;
    private  int previus_height = 0;
    private FeedInputTensorHelper(int width, int height, float mean, float std) {
        previus_width = width;
        previus_height = height;
        tensorImage = new TensorImage(DataType.FLOAT32);
        downSizeImageProcessor =
                new ImageProcessor.Builder()
                        // Resize using Bilinear or Nearest neighbour
                        .add(new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
                        // Rotation counter-clockwise in 90 degree increments
//                      .add(new Rot90Op(rotateDegrees / 90))
                        .add(new NormalizeOp(mean, std))
//                      .add(new QuantizeOp(128.0, 1/128.0))
                        .build();
        upSizeImageProcessor =
                new ImageProcessor.Builder()
                        // Center crop the image to the largest square possible
                        .add(new ResizeWithCropOrPadOp(height, width))
                        .add(new NormalizeOp(mean, std))
                        .build();
    }

    public static synchronized FeedInputTensorHelper getInstance(int width, int height, float mean, float std) {
        if (instance == null) {
            instance = new FeedInputTensorHelper(width, height,mean, std);
        }else{
            if (instance.previus_width!=width || instance.previus_height!=height){
                instance = new FeedInputTensorHelper(width, height,mean, std);
            }
        }
        return instance;
    }

    public static TensorImage getBytebufferFromBitmap(Bitmap bitmap,
                                                      int input_width,
                                                      int input_height, float mean, float std, String size_option) throws Exception {
        try{
            //https://www.tensorflow.org/lite/inference_with_metadata/lite_support
            FeedInputTensorHelper feedInputTensorHelper = getInstance(input_width, input_height, mean, std);
            feedInputTensorHelper.tensorImage.load(bitmap);
            if (size_option=="downsize"){
                return feedInputTensorHelper.downSizeImageProcessor.process(feedInputTensorHelper.tensorImage);
            }
            if (size_option=="upsize"){
                return feedInputTensorHelper.upSizeImageProcessor.process(feedInputTensorHelper.tensorImage);
            }
            throw new Exception("internal error, size_option no supported");
        }catch (Exception e){
            throw e;
        }
    }
}
