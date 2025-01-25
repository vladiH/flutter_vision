package com.vladih.computer_vision.flutter_vision.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;

public class RenderScriptHelper {
    private static RenderScriptHelper instance;

    private final RenderScript rs;
    private final ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic;
    private Allocation in;
    private Allocation out;
    private int lastWidth = -1;
    private int lastHeight = -1;
    private int lastNv21Length = -1;

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
        // Recreate YUV allocation if NV21 array size changes
        if (nv21.length != lastNv21Length) {
            Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
            in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);
            lastNv21Length = nv21.length;
        }

        // Recreate RGBA allocation if dimensions change
        if (width != lastWidth || height != lastHeight) {
            Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
            out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);
            lastWidth = width;
            lastHeight = height;
        }

        // Convert YUV to RGBA
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