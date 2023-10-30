package com.vladih.computer_vision.flutter_vision.models;

import android.graphics.Bitmap;

import com.googlecode.tesseract.android.TessBaseAPI;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

public class Tesseract {
    private TessBaseAPI interpreter;
    private final  int default_page_seg_mode = TessBaseAPI.PageSegMode.PSM_SINGLE_BLOCK;
    private final String tess_data;
    private final Map<String,String> arg;
    private final String language;

    public Tesseract(String tess_data, Map<String, String> arg, String language) {
        this.tess_data = tess_data;
        this.arg = arg;
        this.language = language;
    }
    public void close(){
        if (interpreter!=null){
            interpreter.clear();
            interpreter.recycle();
        }
    }
    public void initialize_model() throws Exception {
        try {
            if(interpreter==null){
                this.interpreter = new TessBaseAPI();
                if (!this.interpreter.init(this.tess_data, this.language)) {
                    // Error initializing Tesseract (wrong data path or language)
                    this.interpreter.recycle();
                    throw new Exception("Cannot initialize Tesseract model");
                }
                if(!this.arg.isEmpty()){
                    for(Map.Entry<String, String> entry:this.arg.entrySet()){
                        interpreter.setVariable(entry.getKey(),entry.getValue());
                    }
                }
                interpreter.setPageSegMode(this.default_page_seg_mode);
            }
        }
        catch (Exception e){
            throw e;
        }
    }

    public Map<String, Object> predict_text(Bitmap bitmap) throws Exception {
        try{
            this.interpreter.setImage(bitmap);
            String result = this.interpreter.getUTF8Text();
            return out(result, interpreter.meanConfidence(), interpreter.wordConfidences());
        }catch (Exception e){
            throw new Exception(e.getMessage());
        }finally {
            if(!bitmap.isRecycled()){
                bitmap.recycle();
            }
        }
    }

    protected Map<String, Object> out(String text, int mean, int[] word_conf){
        try {
            Map<String, Object> result = new HashMap<String,Object>();
            result.put("text",text);
            result.put("mean_conf", mean);
            result.put("word_conf",word_conf);
            return result;
        }catch (Exception e){
            throw e;
        }
    }
}
