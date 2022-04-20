package com.vladih.computer_vision.flutter_vision.models;

import android.graphics.Bitmap;

import com.googlecode.tesseract.android.TessBaseAPI;
import com.vladih.computer_vision.flutter_vision.utils.responses;
import com.vladih.computer_vision.flutter_vision.utils.utils;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.photo.Photo;
import org.tensorflow.lite.Interpreter;

import java.io.IOException;
import java.util.Map;

public class tesseract {
    private TessBaseAPI interpreter;
    private final  int default_page_seg_mode = TessBaseAPI.PageSegMode.PSM_SINGLE_BLOCK;
    private final String tess_data;
    private final Map<String,String> arg;
    private final String language;

    public tesseract(String tess_data, Map<String, String> arg, String language) {
        this.tess_data = tess_data;
        this.arg = arg;
        this.language = language;
    }
    public TessBaseAPI getModel() {return this.interpreter;};
    public void close(){
        if (interpreter!=null){
            interpreter.clear();
            interpreter.recycle();
        }
    }
    public responses initialize_model() throws IOException {
        try {
            if(interpreter==null){
                this.interpreter = new TessBaseAPI();
                this.interpreter.init(this.tess_data, this.language);
            }
            if(!this.arg.isEmpty()){
                for(Map.Entry<String, String> entry:this.arg.entrySet()){
                    interpreter.setVariable(entry.getKey(),entry.getValue());
                }
            }
            interpreter.setPageSegMode(this.default_page_seg_mode);
            return  responses.success("Tesseract model loaded success");
        }
        catch (Exception e){
            return responses.error("Cannot initialize tesseract model: "+e.getMessage());
        }
    }

    public String predict_text(Bitmap bitmap) throws Exception {
        try{
            this.interpreter.clear();
            //Mat mat=preprocess_bitmap(bitmap);
            //bitmap = min_300dpi(mat);
            Mat mat = min_300dpi(bitmap);
            bitmap=preprocess_bitmap(mat);
            //utils.getScreenshotBmp(bitmap, "preprocess");
            this.interpreter.setImage(bitmap);
            String result = this.interpreter.getUTF8Text();
            System.out.println("++++++++++++++++mean confidence++++++++++++++++++++");
            System.out.println(this.interpreter.meanConfidence());
            System.out.println("=============word confidence==============");
            for(int i: this.interpreter.wordConfidences()){
                System.out.println(i);
            }
            this.interpreter.stop();
            return result;
        }catch (Exception e){
            throw new Exception(e.getMessage());
        }finally {
            if(!bitmap.isRecycled()){
                bitmap.recycle();
            }
        }
    }

    private Mat preprocess_bitmap(Bitmap bitmap) throws Exception {
        Mat matrix1;
        Mat matrix2;
        Mat matrix1t=new Mat();
        Mat matrix2t=new Mat();
        try {
            matrix1 = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC1);
            Utils.bitmapToMat(bitmap,matrix1);
            Photo.fastNlMeansDenoisingColored(matrix1,matrix1,10,10,7,15);
            Imgproc.cvtColor(matrix1,matrix1, Imgproc.COLOR_RGB2GRAY);
            matrix2 = matrix1.clone();
            for (int i=0; i<4; i++){
                Imgproc.GaussianBlur(matrix1.clone(), matrix1t, new Size(5,5),3);
                Core.addWeighted(matrix1.clone(), 1.48, matrix1t, -0.5, 0, matrix1);

                Imgproc.GaussianBlur(matrix2.clone(), matrix2t, new Size(0,0),5);
                Core.addWeighted(matrix2.clone(), 1.5, matrix2t, -0.5, 0, matrix2);
            }
            Imgproc.GaussianBlur(matrix1,matrix1, new Size(7,7),0);
            Imgproc.adaptiveThreshold(matrix1,matrix1,255,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY,
                    101,25);
            Core.bitwise_not(matrix1,matrix1);//add
            Imgproc.medianBlur(matrix1,matrix1,3);
            /*Imgproc.erode(matrix1,matrix1, Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                    new Size(1,3), new Point(-1,-1)));*/
            Imgproc.dilate(matrix1,matrix1, Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                    new Size(1,1)),new Point(-1,-1),1);
            Core.bitwise_not(matrix1,matrix1);
            Core.bitwise_or(matrix1,matrix2, matrix1);
            return matrix1;
        }catch (Exception e){
            throw new Exception(e.getMessage());
        }
        finally {
            matrix1=null;
            matrix2=null;
            matrix1t=null;
            matrix2t=null;
            if(!bitmap.isRecycled()){
                bitmap.recycle();
            }
        }
    }
    private Bitmap preprocess_bitmap(Mat matrix1) throws Exception {
        Mat matrix2;
        Mat matrix1t=new Mat();
        Mat matrix2t=new Mat();
        try {
            //15
            Photo.fastNlMeansDenoisingColored(matrix1,matrix1,10,10,7,21);
            Imgproc.cvtColor(matrix1,matrix1, Imgproc.COLOR_RGB2GRAY);
            matrix2 = matrix1.clone();
            for (int i=0; i<4; i++){
                Imgproc.GaussianBlur(matrix1.clone(), matrix1t, new Size(5,5),3);
                //1.48
                Core.addWeighted(matrix1.clone(), 1.48, matrix1t, -0.5, 0, matrix1);

                Imgproc.GaussianBlur(matrix2.clone(), matrix2t, new Size(0,0),5);
                Core.addWeighted(matrix2.clone(), 1.5, matrix2t, -0.5, 0, matrix2);
            }
            Imgproc.GaussianBlur(matrix1,matrix1, new Size(7,7),0);
            Imgproc.adaptiveThreshold(matrix1,matrix1,255,
                    Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
                    Imgproc.THRESH_BINARY,
                    101,25);
            Core.bitwise_not(matrix1,matrix1);//add
            Imgproc.medianBlur(matrix1,matrix1,3);
            /*Imgproc.erode(matrix1,matrix1, Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                    new Size(1,3), new Point(-1,-1)));*/
            Imgproc.dilate(matrix1,matrix1, Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                    new Size(1,1)),new Point(-1,-1),1);
            Core.bitwise_not(matrix1,matrix1);
            Core.bitwise_or(matrix1,matrix2, matrix1);
            Imgproc.threshold(matrix1,matrix1,127,255,Imgproc.THRESH_BINARY);
            double angle = utils.computeSkewAngle(matrix1.clone());
            System.out.println("+++++++++++++++ANGLE+++++++++++++++++++++");
            System.out.println(angle);
            //TODO:POR CORREGIR
            if (angle!=90)
                matrix1 = utils.deskew(matrix1,angle);
            Imgproc.GaussianBlur(matrix1,matrix1, new Size(5,5),2);
            Size size = matrix1.size();
            Bitmap bitmap = Bitmap.createBitmap((int)size.width, (int) size.height, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(matrix1,bitmap);
            return bitmap;
        }catch (Exception e){
            throw new Exception(e.getMessage());
        }
        finally {
            matrix1=null;
            matrix2=null;
            matrix1t=null;
            matrix2t=null;
        }
    }
    private Bitmap min_300dpi(Mat mat) throws Exception {
        Bitmap bitmap=null;
        try {
            Size size = mat.size();
            double max = Math.min(size.width,size.height);
            int scalefactor = (int) Math.round(300/max);
            size = new Size(size.width*scalefactor, size.height*scalefactor);
            Imgproc.resize(mat, mat, size, 1.2,1.2, Imgproc.INTER_AREA);
            bitmap = Bitmap.createBitmap((int)size.width, (int) size.height, Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat,bitmap);
            return bitmap;
        }catch (Exception e){
            throw new Exception(e.getMessage());
        }finally {
            if(!bitmap.isRecycled()){
                bitmap.recycle();
            }
        }
    }
    private Mat min_300dpi(Bitmap bitmap) throws Exception {
        try {
            System.out.println(bitmap.getHeight());
            System.out.println(bitmap.getWidth());
            Mat mat = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC1);
            Utils.bitmapToMat(bitmap,mat);
            if(!bitmap.isRecycled()){
                bitmap.recycle();
            }
            Size size = mat.size();
            double max = Math.max(size.width,size.height);
            int scalefactor = (int) Math.round(300/max);
            size = new Size(size.width*scalefactor, size.height*scalefactor);
            //System.out.println(size.height);
            //System.out.println(size.width);
            Imgproc.resize(mat, mat, size, 1.2,1.2, Imgproc.INTER_AREA);
            //Bitmap bitmap = Bitmap.createBitmap((int)size.width, (int) size.height, Bitmap.Config.ARGB_8888);
            //Utils.matToBitmap(mat,bitmap);
            return mat;
        }catch (Exception e){
            throw new Exception(e.getMessage());
        }
    }
}
