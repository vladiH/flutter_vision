package com.vladih.computer_vision.flutter_vision.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;

public class RenderScriptHelper {
    private static RenderScriptHelper instance;

    private RenderScript rs;
    private ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic;
    private Type.Builder yuvType;
    private Type.Builder rgbaType;
    private Allocation in;
    private Allocation out;

    private RenderScriptHelper(Context context) {
        rs = RenderScript.create(context);
        yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));
    }

    public static synchronized RenderScriptHelper getInstance(Context context) {
        if (instance == null) {
            instance = new RenderScriptHelper(context);
        }
        return instance;
    }

    public Allocation renderScriptNV21ToRGBA888(int width, int height, byte[] nv21) {
        if (yuvType == null) {
            yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
        }
        if (rgbaType == null) {
            rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
        }
        // Create input allocation for YUV data
        if (in == null) {
            in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);
        }
        // Create output allocation for RGBA data
        if (out == null) {
            out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
        }

        // Convert YUV to RGBA using RenderScript intrinsic
        in.copyFrom(nv21);
        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);
        return out;
    }

    public static Bitmap getBitmapFromNV21(Context context, byte[] nv21, int width, int height) {
        RenderScriptHelper rsHelper = getInstance(context);
        //https://blog.minhazav.dev/how-to-convert-yuv-420-sp-android.media.Image-to-Bitmap-or-jpeg/
        Allocation allocation = rsHelper.renderScriptNV21ToRGBA888(width, height, nv21);

        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        allocation.copyTo(bitmap);

        return bitmap;
    }
}