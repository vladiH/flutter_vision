package com.vladih.computer_vision.flutter_vision.utils;
import static java.lang.Math.min;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Environment;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;
import org.tensorflow.lite.support.image.TensorImage;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

public class utils {
    public static Bitmap crop_bitmap(Bitmap bitmap, float x1, float y1, float x2, float y2) throws Exception {
        try{
            final int x = Math.max((int)x1,0);
            final int y = Math.max((int)y1,0);
            final int width = Math.abs((int)(x2-x1));
            final int height = Math.abs((int)(y2-y1));
            return  Bitmap.createBitmap(bitmap,x,y,width,height);
        }catch (Exception e){
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
    public static Mat image_preprocessing(Bitmap bitmap){
        try {
            Mat mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC1);
            Utils.bitmapToMat(bitmap,mat);
            Photo.fastNlMeansDenoisingColored(mat,mat,10,10,3,21);
            Core.normalize(mat,mat, 0, 255, Core.NORM_MINMAX);
            Photo.fastNlMeansDenoisingColored(mat,mat,10,10,3,21);
            Imgproc.cvtColor(mat,mat, Imgproc.COLOR_RGB2GRAY);
            Imgproc.GaussianBlur(mat, mat, new Size(3,3), 1);
            Imgproc.adaptiveThreshold(mat,mat,255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,13,4);
            return mat;
        }catch (Exception e){
            throw e;
        }
    }
    public static Mat deskew(Mat image, double skewAngle) {
        try {
            Mat rotationMatrix = Imgproc.getRotationMatrix2D(new Point(image.width() / 2, image.height() / 2), -skewAngle, 1);
            Scalar borderValue = new Scalar(255); // white border value
            // Crop the output image to remove border artifacts
            Rect cropRect = new Rect(0, 0, image.width(), image.height());
//            System.out.println(image.size());
//            System.out.println(skewAngle);
            Imgproc.warpAffine(image, image, rotationMatrix, image.size(), Imgproc.INTER_CUBIC + Imgproc.WARP_FILL_OUTLIERS, Core.BORDER_CONSTANT, borderValue);
            image = image.submat(cropRect);
            return image;
        }catch (Exception e){
            throw e;
        }
    }

    //input:binary matrix
    public static double computeSkewAngle(Mat image) {
        try {
            // Apply Canny edge detection to find the edges in the image
            // or convert letter and background to white and black color respectively
//          Imgproc.Canny(image, image, 50, 200, 3);
            Core.bitwise_not(image, image);
            // Apply the Hough transform to find the lines in the image
            Mat lines = new Mat();
            Imgproc.HoughLines(image, lines, 1, Math.PI / 180, 100);

            // Compute the average angle of the lines
            double angle = 0.0;
            int numLines = lines.cols();
            if (numLines > 0) {
                for (int i = 0; i < numLines; i++) {
                    double[] data = lines.get(0, i);
                    double rho = data[0];
                    double theta = data[1];

                    // Convert the line from polar to Cartesian coordinates
                    double a = Math.cos(theta);
                    double b = Math.sin(theta);
                    double x0 = a * rho;
                    double y0 = b * rho;
                    // Compute the angle of the line in radians and add it to the total angle
                    angle += Math.atan2(y0, x0);
                }
            } else {
                // No lines were detected, so set the angle to zero
                angle = 0.0;
            }
            // Convert the angle from radians to degrees and return it
            return 90-(angle * 180 / Math.PI);
        } catch (Exception e) {
            throw e;
        }
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
