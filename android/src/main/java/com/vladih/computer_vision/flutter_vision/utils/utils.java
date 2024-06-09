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
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
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
    //These code lines work well but is so slower, now rename as filterTextFromImage
//    public static Mat image_preprocessing(Mat mat){
//        try {
//            Photo.fastNlMeansDenoising(mat,mat, new MatOfFloat(7),3,21, Core.NORM_L1);
//            Core.normalize(mat,mat, 0, 255, Core.NORM_MINMAX);
//            Imgproc.GaussianBlur(mat, mat, new Size(5,5), 1);
//            Imgproc.adaptiveThreshold(mat,mat,255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,Imgproc.THRESH_BINARY,21,15);
//            return mat;
//        }catch (Exception e){
//            throw e;
//        }
//    }
        //Accept gray Mat
        public static Mat filterTextFromImage(Mat mat){
        try {
            //find posibles box text
            List<Rect> rects = findRects(mat);
            //join posibles box text that belong to same horizontal line
            rects = mergeRects(rects);
            //remove boxes which belong to another
            rects = non_max_suppression(rects);

            Mat new_image = new Mat(mat.height(), mat.width(), CvType.CV_8UC1, new Scalar(255));
            for (Rect box : rects) {
                try{
                    //Todo: Still there are error to fix when image have black border ie. images, stains
                    //this errors are produced by findRects function, also text doesnt working when
                    // images has wave text
                    Mat crop = mat.submat(box);
                    Core.normalize(crop, crop, 0, 255, Core.NORM_MINMAX);
                    Imgproc.threshold(crop, crop, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
                    Scalar meanScalar = Core.mean(crop);
                    double meanValue = meanScalar.val[0];
                    if (meanValue<100){
                        continue;
                    }
                    Mat roi = new_image.submat(new Rect(box.x, box.y, box.width, box.height));
                    Core.bitwise_and(crop,roi,roi);
                }catch (Exception e){
                    System.err.println("Warning, vission text error filter");
                }
//                crop.copyTo(roi);
            }
            return new_image;
        }catch (Exception e){
            throw e;
        }
    }
    public static Mat rgbBitmapToMatGray(Bitmap bitmap){
        Mat mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC3);
        Utils.bitmapToMat(bitmap,mat);
        Imgproc.cvtColor(mat,mat, Imgproc.COLOR_RGB2GRAY);
        return mat;
    }

    public static List<Rect> non_max_suppression(List<Rect> boxes) {
        // Sort the list of bounding boxes in ascending order based on their y-coordinates
        Collections.sort(boxes, new Comparator<Rect>() {
            @Override
            public int compare(Rect r1, Rect r2) {
                return Integer.compare(r1.y, r2.y);
            }
        });

        // Initialize the list of selected boxes
        List<Rect> selected_boxes = new ArrayList<>();

        // Perform NMS
        while (boxes.size() > 0) {
            Rect current = boxes.get(0);
            selected_boxes.add(current);
            boxes.remove(0);

            List<Rect> next_boxes = new ArrayList<>();
            for (Rect box : boxes) {
                if (!contain(current, box) && (box.width/ box.height)>1) {
                    next_boxes.add(box);
                }
            }
            boxes = next_boxes;
        }

        return selected_boxes;
    }

    public static boolean contain(Rect box1, Rect box2) {
        // Calculate the coordinates of the intersection rectangle
        int x1 = Math.max(box1.x, box2.x);
        int y1 = Math.max(box1.y, box2.y);
        int x2 = Math.min(box1.x+box1.width, box2.x+box2.width);
        int y2 = Math.min(box1.y+box1.height, box2.y+box2.height);
        int w = Math.max(0, x2 - x1);
        int h = Math.max(0, y2 - y1);

        // Calculate the area of intersection rectangle
        int intersection = w * h;

        // Calculate the area of both bounding boxes
//        int area_box1 = box1.width * box1.height;
        int area_box2 = box2.width * box2.height;

        return area_box2 == intersection;
    }
    public static List<Rect> mergeRects(List<Rect> rects){
        Collections.sort(rects, new Comparator<Rect>() {
            @Override
            public int compare(Rect r1, Rect r2) {
                return r1.y - r2.y;
            }
        });
        List<Rect> mergedBoxes = new ArrayList<>();
        for (int i = 0; i < rects.size(); i++) {
            Rect rect = rects.get(i);
            if (mergedBoxes.isEmpty()) {
                mergedBoxes.add(rect);
            } else {
                Rect prevRect = mergedBoxes.get(mergedBoxes.size()-1);
                if (Math.abs(rect.y - prevRect.y) > prevRect.height * 0.5) {
                    mergedBoxes.add(rect);
                } else {
                    if(rect.x - (prevRect.x + prevRect.width)> 0){
                        mergedBoxes.add(rect);
                    }
                    else{
                        int newX = Math.min(rect.x, prevRect.x);
                        int newY = Math.min(rect.y, prevRect.y);
                        int newW = Math.max(rect.x + rect.width, prevRect.x + prevRect.width) - newX;
                        int newH = Math.max(rect.y + rect.height, prevRect.y + prevRect.height) - newY;
                        mergedBoxes.set(mergedBoxes.size()-1, new Rect(newX, newY, newW, newH));
                    }
                }
            }
        }
        return mergedBoxes;
    }
    public static List<Rect> findRects(Mat image){
        Mat thresh = new Mat();
        //find countours in gray image
        Imgproc.Canny(image, thresh, 50, 150, 3, false);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        List<Rect> filteredContours = new ArrayList<>();
        double height = 0;
        for (int i = 0; i < contours.size(); i++) {
            Rect rect = Imgproc.boundingRect(contours.get(i));
            double area = rect.width * rect.height;
            double aspectRatio = (double)rect.width / rect.height;
            //remove rect that doesn't satisface this rules
            if (area > 80 && aspectRatio > 0.4 && aspectRatio < 5) {
                height += rect.height;
                filteredContours.add(new Rect((int)Math.max(0, rect.x-rect.height/2), rect.y, (int)Math.min(image.width(),rect.width+rect.height), rect.height));
            }
        }
        //remove rects with large height than mean height
        height = height / Math.sqrt(filteredContours.size());
        List<Rect> rects = new ArrayList<>();
        for (int i = 0; i < filteredContours.size(); i++) {
            Rect rect = filteredContours.get(i);
            if (rect.height > height) continue;
            rects.add(rect);
        }
        return rects;
    }
    public static Mat deskew(Mat image, double skewAngle) {
        try {
            Mat rotationMatrix = Imgproc.getRotationMatrix2D(new Point(image.width() / 2, image.height() / 2), skewAngle, 1);
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
            Imgproc.Canny(image, image, 50, 150, 3);
            // Apply the Hough transform to find the lines in the image
            Mat lines = new Mat();
            Imgproc.HoughLinesP(image, lines, 1, Math.PI / 180, 100, 100, 10);

            // Compute the average angle of the lines
            double angle = 0.0;
            int numLines = lines.cols();
            if (numLines > 0) {
                // Find the longest line
                double longestLineLength = -1;
                Point[] longestLine = null;
                for (int i = 0; i < lines.cols(); i++) {
                    double[] line = lines.get(0, i);
                    Point pt1 = new Point(line[0], line[1]);
                    Point pt2 = new Point(line[2], line[3]);
                    double length = Math.sqrt(Math.pow(pt2.x - pt1.x, 2) + Math.pow(pt2.y - pt1.y, 2));
                    if (length > longestLineLength) {
                        longestLineLength = length;
                        longestLine = new Point[] { pt1, pt2 };
                    }
                }
                // Calculate angle between longest line and horizontal axis
                double dx = longestLine[1].x - longestLine[0].x;
                double dy = longestLine[1].y - longestLine[0].y;
                angle = Math.atan2(dy, dx) * 180 / Math.PI;
            }
            // Convert the angle from radians to degrees and return it
            return angle;
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
