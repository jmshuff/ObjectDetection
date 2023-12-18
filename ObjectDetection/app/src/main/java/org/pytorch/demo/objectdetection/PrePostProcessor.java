// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

package org.pytorch.demo.objectdetection;

import static java.lang.Math.min;

import android.graphics.Rect;
import android.util.Log;

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
    private static int mOutputColumn = 38; // left, top, right, bottom, score and 1 class probability
    private static float mThreshold = 0.30f; // score above which a detection is generated
    private static int mNmsLimit = 15;

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
        Collections.sort(boxes,
                new Comparator<Result>() {
                    @Override
                    public int compare(Result o1, Result o2) {
                        return o1.score.compareTo(o2.score);
                    }
                });

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

                float max = outputs[i* mOutputColumn +5];
                int cls = 0;
                for (int j = 0; j < mOutputColumn -5; j++) {
                    if (outputs[i* mOutputColumn +5+j] > max) {
                        max = outputs[i* mOutputColumn +5+j];
                        cls = j;
                    }
                }

                float[] mask = processMaskNative(proto, Arrays.copyOfRange(outputs,i*mOutputColumn + 6,  i*mOutputColumn + mOutputColumn));
                Rect rect = new Rect((int)(startX+ivScaleX*left), (int)(startY+top*ivScaleY), (int)(startX+ivScaleX*right), (int)(startY+ivScaleY*bottom));
                Result result = new Result(cls, outputs[i*mOutputColumn+4], rect, mask);
                results.add(result);
            }
        }
        //return results;
        return nonMaxSuppression(results, mNmsLimit, mThreshold);
    }

    static float[] processMaskNative(float[] protos, float[] masksIn) {

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
            Log.d("resultshape", String.valueOf(result.length));
        }

        float gain = min(mh / mInputHeight, mw / mInputWidth);
        float padWidth = (mw - mInputWidth* gain) / 2;
        float padHeight= (mh - mInputHeight * gain) / 2;

        return result;

/*        long[] protosShape = protos.shape();
        int c = (int) protosShape[0]; // Assuming c = channels
        int mh = (int) protosShape[1]; // Mask height
        int mw = (int) protosShape[2]; // Mask width

        // masks_in @ protos.float().view(c, -1)
        Tensor masks = masksIn.mm(protos.view(c, -1).toType(masksIn.dtype())).sigmoid().view(-1, mh, mw);

        // Calculate gain
        float gain = Math.min((float) mh / shape[0], (float) mw / shape[1]); // gain  = old / new
        float padX = (mw - shape[1] * gain) / 2;
        float padY = (mh - shape[0] * gain) / 2;

        int top = Math.round(padY);
        int left = Math.round(padX);
        int bottom = mh - Math.round(padY);
        int right = mw - Math.round(padX);

        // masks[:, top:bottom, left:right]
        masks = masks.slice(1, top, bottom).slice(2, left, right);

        // F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]
        masks = Module.interpolate(masks.unsqueeze(0), shape, true).toTensor();

        // masks = crop_mask(masks, bboxes);
        // This section is approximate and should be substituted with actual cropping logic
        // This assumes cropping logic based on bounding boxes, adjust as per your requirement
        masks = customCropMaskLogic(masks, bboxes);

        // masks.gt_(0.5)
        masks = masks.gt(0.5);

        return masks;*/
    }


    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
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
