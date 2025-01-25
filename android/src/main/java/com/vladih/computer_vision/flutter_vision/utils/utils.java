package com.vladih.computer_vision.flutter_vision.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.support.image.TensorImage;

import java.nio.ByteBuffer;
import java.util.List;

public class utils {
    public static Bitmap crop_bitmap(Bitmap bitmap, float x1, float y1, float x2, float y2)  {
        try{
            final int x = Math.max((int)x1,0);
            final int y = Math.max((int)y1,0);
            final int width = Math.abs((int)(x2-x1));
            final int height = Math.abs((int)(y2-y1));
            return  Bitmap.createBitmap(bitmap,x,y,width,height);
        }catch (Exception e){
            throw  e;
        }
    }

    public static Mat rgbBitmapToMatGray(Bitmap bitmap){
        Mat mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC3);
        Utils.bitmapToMat(bitmap,mat);
        Imgproc.cvtColor(mat,mat, Imgproc.COLOR_RGB2GRAY);
        return mat;
    }


    public static ByteBuffer feedInputTensor(
                                            Bitmap bitmap,
                                            int input_width,
                                            int input_height,
                                            int src_width,
                                            int src_height,
                                            float mean,
                                            float std) throws Exception {
        try {
//            utils.getScreenshotBmp(bitmap, "antes");
            TensorImage tensorImage;
            if (src_width > input_width || src_height > input_height) {
                tensorImage= FeedInputTensorHelper.getBytebufferFromBitmap(bitmap, input_width, input_height, mean, std, "downsize");
            }else{
                tensorImage= FeedInputTensorHelper.getBytebufferFromBitmap(bitmap, input_width, input_height, mean, std, "upsize");
            }
//            utils.getScreenshotBmp(tensorImage.getBitmap(), "despues");
            return tensorImage.getBuffer();
        }catch (Exception e){
            throw e;

        }finally {
            assert bitmap != null;
            if(!bitmap.isRecycled()){
                bitmap.recycle();
            }
        }
    }
    public static Bitmap feedInputToBitmap(Context context,
                                           List<byte[]> bytesList,
                                           int imageHeight,
                                           int imageWidth,
                                           int rotation) throws Exception {

        int Yb = bytesList.get(0).length;
        int Ub = bytesList.get(1).length ;
        int Vb = bytesList.get(2).length ;
        // Copy YUV data to plane byte
        byte[] data = new byte[Yb+Ub+Vb];
        System.arraycopy(bytesList.get(0), 0, data, 0, Yb);
        System.arraycopy(bytesList.get(2), 0, data, Yb, Ub);
        System.arraycopy(bytesList.get(1), 0, data, Yb+Ub, Vb);

        Bitmap bitmapRaw = RenderScriptHelper.getBitmapFromNV21(context,data, imageWidth, imageHeight);
//        utils.getScreenshotBmp(bitmapRaw, "NV21");
        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        bitmapRaw = Bitmap.createBitmap(bitmapRaw, 0, 0, bitmapRaw.getWidth(), bitmapRaw.getHeight(), matrix, true);
        return bitmapRaw;
    }
}
