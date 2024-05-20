// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import static java.lang.Math.min;

import android.graphics.Rect;
import android.util.Log;

import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

class Result {
    int classIndex;
    Float score;
    Rect rect;
    float[] mask;
    public Result(int cls, Float output, Rect rect, float[] mask) {
        this.classIndex = cls;
        this.score = output;
        this.rect = rect;
        this.mask = mask;

    }

    public float[] getMask() {
        return mask;
    }

    public void setMask(float[] mask) {
        this.mask = mask;
    }
};

public class PrePostProcessor {
    // for yolov5 model, no need to apply MEAN and STD
    static float[] NO_MEAN_RGB = new float[] {0.0f, 0.0f, 0.0f};
    static float[] NO_STD_RGB = new float[] {1.0f, 1.0f, 1.0f};

    // model input image size
    static int mInputWidth = 320;
    static int mInputHeight = 320;

    static int MASK_DIM= 32;

    // model output is of size 25200*(num_of_class+5)
        private static int mOutputRow = 6300; // as decided by the YOLOv5 model for input image of size 95640*640
    private static int mOutputColumn = 38; // left, top, right, bottom, score, 32 masks, and 1 class probability
    private static float mThreshold = 0.60f; // score above which a detection is generated
    private static int mNmsLimit = 1;

    static String[] mClasses;

    // The two methods nonMaxSuppression and IOU below are ported from https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
     Removes bounding boxes that overlap too much with other boxes that have
     a higher score.
     - Parameters:
     - boxes: an array of bounding boxes and their scores
     - limit: the maximum number of boxes that will be selected
     - threshold: used to decide whether boxes overlap too much
     */
    static ArrayList<Result> nonMaxSuppression(ArrayList<Result> boxes, int limit, float threshold) {

        // Do an argsort on the confidence scores, from high to low.
        //sort from high to low
        Collections.sort(boxes, Collections.reverseOrder(new Comparator<Result>() {
            @Override
            public int compare(Result o1, Result o2) {
                return o1.score.compareTo(o2.score);
            }
        }));

        ArrayList<Result> selected = new ArrayList<>();
        boolean[] active = new boolean[boxes.size()];
        Arrays.fill(active, true);
        int numActive = active.length;

        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
        boolean done = false;
        for (int i=0; i<boxes.size() && !done; i++) {
            if (active[i]) {
                Result boxA = boxes.get(i);
                selected.add(boxA);
                if (selected.size() >= limit) break;

                for (int j=i+1; j<boxes.size(); j++) {
                    if (active[j]) {
                        Result boxB = boxes.get(j);
                        if (IOU(boxA.rect, boxB.rect) > threshold) {
                            active[j] = false;
                            numActive -= 1;
                            if (numActive <= 0) {
                                done = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
        return selected;
    }

    /**
     Computes intersection-over-union overlap between two bounding boxes.
     */
    static float IOU(Rect a, Rect b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        if (areaA <= 0.0) return 0.0f;

        float areaB = (b.right - b.left) * (b.bottom - b.top);
        if (areaB <= 0.0) return 0.0f;

        float intersectionMinX = Math.max(a.left, b.left);
        float intersectionMinY = Math.max(a.top, b.top);
        float intersectionMaxX = min(a.right, b.right);
        float intersectionMaxY = min(a.bottom, b.bottom);
        float intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    static ArrayList<Result> outputsToNMSPredictions(float[] outputs, float imgScaleX, float imgScaleY, float ivScaleX, float ivScaleY, float startX, float startY, float[]proto) {
        ArrayList<Result> results = new ArrayList<>();

        for (int i = 0; i< mOutputRow; i++) {

            if (outputs[i* mOutputColumn +4] > mThreshold) {
                float x = outputs[i* mOutputColumn];
                float y = outputs[i* mOutputColumn +1];
                float w = outputs[i* mOutputColumn +2];
                float h = outputs[i* mOutputColumn +3];
                //System.out.println("output: left-" + x + " top-" + y + " right-" + w + " bottom-" + h);
                float left = imgScaleX * (x - w/2);
                float top = imgScaleY * (y - h/2);
                float right = imgScaleX * (x + w/2);
                float bottom = imgScaleY * (y + h/2);
                float[] bbox = {left, top, right, bottom};

                float max = outputs[i* mOutputColumn +5]; //JS to look at
                int cls = 0;
                for (int j = 0; j < mOutputColumn -5; j++) {
                    if (outputs[i* mOutputColumn +5+j] > max) {
                        max = outputs[i* mOutputColumn +5+j];
                        cls = j;
                    }
                }
               //move this to only calculate mask for the final result
                float[] mask = processMaskNative(proto, Arrays.copyOfRange(outputs,i*mOutputColumn + 6,  i*mOutputColumn + mOutputColumn), bbox);
                Rect rect = new Rect((int)(startX+ivScaleX*left), (int)(startY+top*ivScaleY), (int)(startX+ivScaleX*right), (int)(startY+ivScaleY*bottom));
                Result result = new Result(cls, outputs[i*mOutputColumn+4], rect, mask);
                results.add(result);
            }
        }
        //return results;
        return nonMaxSuppression(results, mNmsLimit, mThreshold);
    }

    static float[] processMaskNative(float[] protos, float[] masksIn, float[] bbox) {

        int n = 1; //number of masks
        int mw = 80; //mask width
        int mh = 80; //mask height
        float[] result = new float[n * mh * mw];

        //Matrix multiplication
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < mh * mw; j++) {
                float sum = 0;
                for (int k = 0; k < MASK_DIM; k++) {
                    sum += protos[k * mh * mw + j] * masksIn[i * MASK_DIM + k];
                }
                result[i * mh * mw + j] = sigmoid(sum); // Apply sigmoid or required function
            }
        }
        float gain = Math.min((float)mh / mInputHeight, (float)mw / mInputWidth);
        float padWidth = Math.max(0, (mw - mInputWidth * gain) / 2);
        float padHeight = Math.max(0, (mh - mInputHeight * gain) / 2);
        //use bilinear interpolation to resize the mask
        //Log.d("maskSigmod", Arrays.toString(result));
        result = bilinearInterpolation(n, result, mw, mh, gain, padWidth, padHeight); //this is making everything equal NaN. Need to fix
        //crop the mask
        //result = cropMask(result, bbox);

        Log.d("resultshape", String.valueOf(result.length));
        Log.d("result", Arrays.toString(result));

        //get result where value > 0.5
        for (int i = 0; i < result.length; i++) {
            if (result[i] > 0.5) {
                result[i] = 1;
            } else {
                result[i] = 0;
            }
        }
        return result;
    }


    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private static float[] bilinearInterpolation(int n, float[] masksIn, int mw, int mh, float gain, float padWidth, float padHeight){
        float[] resizedMask = new float[n * mInputHeight * mInputWidth];

        for (int i = 0; i < n; i++) {
            for (int y = 0; y < mInputHeight; y++) {
                for (int x = 0; x < mInputWidth; x++) {
                    float srcX = (x + padWidth) / gain;
                    float srcY = (y + padHeight) / gain;

                    int x0 = (int) Math.floor(srcX);
                    int x1 = Math.min(x0 + 1, mw - 1); //Math.min(x0 + 1, mw - 1
                    int y0 = (int) Math.floor(srcY);
                    int y1 = Math.min(y0 + 1, mh -1);

                    float dx = srcX - x0;
                    float dy = srcY - y0;

                    float q00 = getPixelValue(masksIn, i, mw, mh, x0, y0);
                    float q01 = getPixelValue(masksIn, i, mw, mh, x0, y1);
                    float q10 = getPixelValue(masksIn, i, mw, mh, x1, y0);
                    float q11 = getPixelValue(masksIn, i, mw, mh, x1, y1);

                    float interpolatedValue = (1 - dx) * (1 - dy) * q00 + dx * (1 - dy) * q10 + (1 - dx) * dy * q01 + dx * dy * q11;

                    resizedMask[i * mInputHeight * mInputWidth + y * mInputWidth + x] = interpolatedValue;
                }
            }
        }
        return resizedMask;
    }

    static float getPixelValue(float[] masks, int maskIndex, int maskWidth, int maskHeight, int x, int y) {
        x = Math.min(Math.max(x, 0), maskWidth - 1);
        y = Math.min(Math.max(y, 0), maskHeight - 1);
        return masks[maskIndex * maskWidth * maskHeight + y * maskWidth + x];
    }



    private static float[] cropMask(float[] mask, float[] box) {
        //set all values outside the bounding box to 0
        for (int i = 0; i < mask.length; i++) {
            if (i % 80 < box[0] || i % 80 > box[2] || i / 80 < box[1] || i / 80 > box[3]) {
                mask[i] = 0;
            }
        }
        return mask;
    }
    private float[][][] reshape(float[] data, int dim1, int dim2, int dim3) {
        float[][][] result = new float[dim1][dim2][dim3];
        int index = 0;
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    result[i][j][k] = data[index++];
                }
            }
        }
        return result;
    }


}
