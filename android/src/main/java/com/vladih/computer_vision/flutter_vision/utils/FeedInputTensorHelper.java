package com.vladih.computer_vision.flutter_vision.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

import java.nio.ByteBuffer;

public class FeedInputTensorHelper {
    private static FeedInputTensorHelper instance;
    private TensorImage tensorImage;
    private ImageProcessor imageProcessor;

    private FeedInputTensorHelper(int width, int height) {
        tensorImage = new TensorImage(DataType.FLOAT32);
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        // Center crop the image to the largest square possible
                        .add(new ResizeWithCropOrPadOp(width, height))
                        // Resize using Bilinear or Nearest neighbour
                        .add(new ResizeOp(height, height, ResizeOp.ResizeMethod.BILINEAR))
                        // Rotation counter-clockwise in 90 degree increments
//                        .add(new Rot90Op(rotateDegrees / 90))
//                                .add(new NormalizeOp(127.5, 127.5))
//                                .add(new QuantizeOp(128.0, 1/128.0))
                        .build();
    }

    public static synchronized FeedInputTensorHelper getInstance(int width, int height) {
        if (instance == null) {
            instance = new FeedInputTensorHelper(width, height);
        }
        return instance;
    }

    public static TensorImage getBytebufferFromBitmap(Bitmap bitmap) {
        try{
            FeedInputTensorHelper feedInputTensorHelper = getInstance(bitmap.getWidth(), bitmap.getHeight());
            feedInputTensorHelper.tensorImage.load(bitmap);
            return feedInputTensorHelper.imageProcessor.process(feedInputTensorHelper.tensorImage);
        }catch (Exception e){
            throw e;
        }
    }
}
