package com.vladih.computer_vision.flutter_vision.utils;
import static java.lang.Math.min;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Environment;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class utils {

    /*
    args:
        model_outputs:[1,25200,[xc,yc, w, h, confidence, class....]]
    out:
        List<[x1,y1, x2, y2, confidence, class]>
     */
    /*
    // test is a 3d array
        double[][][] m = {
                            {
                                {
                                  0.011428807862102985, 0.006756599526852369, 0.04274776205420494, 0.034441519528627396, 0.50012877583503723145, 0.33658933639526367, 0.4722323715686798,
                                },
                                {
                                  0.023071227595210075, 0.006947836373001337, 0.046426184475421906, 0.023744791746139526, 0.3002465546131134033, 0.29862138628959656, 0.4498370885848999,
                                },
                                {
                                  0.03636947274208069, 0.006819264497607946, 0.04913407564163208, 0.025004519149661064, 0.90013208389282226562, 0.3155967593193054, 0.4081345796585083
                                }
                                ,
                                {
                                  0.03636947274208069, 0.006819264497607946, 0.04913407564163208, 0.025004519149661064, 0.00013208389282226562, 0.3155967593193054, 0.4081345796585083
                                },
                                {
                                  0.04930267855525017, 0.007249316666275263, 0.04969717934727669, 0.023645592853426933, 0.0001222355494974181, 0.3123127520084381, 0.40113094449043274
                                }
                            }
                        };

        // for..each loop to iterate through elements of 3d array
        float[][][] save = new float[1][5][7];
        for (int i = 0; i < 1; i++)
            for (int j = 0; j < 5; j++)
                for (int k = 0; k < 7; k++)     // These 2 lines could be replaced by
                    save[i][j][k] = (float) m[i][j][k];

        final List<float []> result = Utils.filter_box(save, 0.3f, 0.3f, 1,1);

        for(float[] x:result){
            for(float v : x){
                System.out.print(v);
                System.out.print(" ");
            }
            System.out.println("");
        }
    }*/


    public static Bitmap crop_bitmap(Bitmap bitmap, float x1, float y1, float x2, float y2) throws Exception {
        try{
            final int x = Math.max((int)x1,0);
            final int y = Math.max((int)y1,0);
            final int width = Math.abs((int)(x2-x1));
            final int height = Math.abs((int)(y2-y1));
            //System.out.println("CROPPP");
            //System.out.println(String.valueOf(x)+" "+String.valueOf(y)+" "+String.valueOf(width)+" "+String.valueOf(height));

            return  Bitmap.createBitmap(bitmap,x,y,width,height);
        }catch (Exception e){
            //System.out.println("CROPPP ERROR");
            //System.out.println(String.valueOf(x1)+" "+String.valueOf(y1)+" "+String.valueOf(x2)+" "+String.valueOf(y2));
            //System.out.println(String.valueOf(bitmap.getWidth())+" "+String.valueOf(bitmap.getHeight()));
            throw  new Exception(e.getMessage());
        }
    }
    public static byte[] bitmap_to_byte(Bitmap bitmap){
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG,100,stream);
        return stream.toByteArray();
    }
    public static Bitmap getScreenshotBmp(Bitmap bitmap, String name) {
        FileOutputStream fileOutputStream = null;

        File path = Environment
                .getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);

        String uniqueID = name;

        File file = new File(path, uniqueID + ".jpg");
        try {
            fileOutputStream = new FileOutputStream(file);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fileOutputStream);

        try {
            fileOutputStream.flush();
            fileOutputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bitmap;
    }

    public static Mat deskew(Mat src, double angle) {
        Point center = new Point(src.width()/2.0, src.height()/2.0);
        Mat rotImage = Imgproc.getRotationMatrix2D(center, angle, 1.0);
        //1.0 means 100 % scale
        Size size = new Size(src.width(), src.height());
        Imgproc.warpAffine(src, src, rotImage, size,
                Imgproc.INTER_CUBIC+ Imgproc.CV_WARP_FILL_OUTLIERS,Core.BORDER_REPLICATE);
        return src;
    }

    //input:binary matrix
    public  static double computeSkewAngle(Mat img){
        //Invert the colors (because objects are represented as white pixels, and the background is represented by black pixels)
        Core.bitwise_not( img, img );
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                new Size(5, 5));
        //We can now perform our erosion, we must declare our rectangle-shaped structuring element and call the erode function
        Imgproc.dilate(img, img, element,new Point(-1,-1),4);
        Bitmap bmp = Bitmap.createBitmap((int)img.width(), (int)img.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img, bmp);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy  = new Mat();
        Imgproc.findContours(img,contours,hierarchy ,Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Collections.sort(contours,(x,y)->Double.compare(Imgproc.contourArea(y),Imgproc.contourArea(x)));
        MatOfPoint2f dst = new MatOfPoint2f();
        contours.get(0).convertTo(dst,CvType.CV_32F);
        RotatedRect rotatedRect = Imgproc.minAreaRect(dst);
        rotatedRect.angle = rotatedRect.angle < -45 ? rotatedRect.angle + 90.f : rotatedRect.angle;
        return rotatedRect.angle;
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
            //utils.getScreenshotBmp(bitmap, "antes");
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
