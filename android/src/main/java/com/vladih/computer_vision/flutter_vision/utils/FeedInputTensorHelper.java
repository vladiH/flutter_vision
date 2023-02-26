package com.vladih.computer_vision.flutter_vision.utils;

import android.graphics.Bitmap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

public class FeedInputTensorHelper {
    private static FeedInputTensorHelper instance;
    private TensorImage tensorImage;
    private ImageProcessor imageProcessor;

    private FeedInputTensorHelper(int width, int height, float mean, float std) {
        tensorImage = new TensorImage(DataType.FLOAT32);
        imageProcessor =
                new ImageProcessor.Builder()
                        // Center crop the image to the largest square possible
                        .add(new ResizeWithCropOrPadOp(height, width))
                        // Resize using Bilinear or Nearest neighbour
                        .add(new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
                        // Rotation counter-clockwise in 90 degree increments
//                        .add(new Rot90Op(rotateDegrees / 90))
                                .add(new NormalizeOp(mean, std))
//                                .add(new QuantizeOp(128.0, 1/128.0))
                        .build();
    }

    public static synchronized FeedInputTensorHelper getInstance(int width, int height, float mean, float std) {
        if (instance == null) {
            instance = new FeedInputTensorHelper(width, height,mean, std);
        }
        return instance;
    }

    public static TensorImage getBytebufferFromBitmap(Bitmap bitmap,
                                                      int input_width,
                                                      int input_height, float mean, float std) {
        try{
            //https://www.tensorflow.org/lite/inference_with_metadata/lite_support
            FeedInputTensorHelper feedInputTensorHelper = getInstance(input_width, input_height, mean, std);
            feedInputTensorHelper.tensorImage.load(bitmap);
            return feedInputTensorHelper.imageProcessor.process(feedInputTensorHelper.tensorImage);
        }catch (Exception e){
            throw e;
        }
    }
}
