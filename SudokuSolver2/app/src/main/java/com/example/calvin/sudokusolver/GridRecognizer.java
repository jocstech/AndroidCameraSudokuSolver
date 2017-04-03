package com.example.calvin.sudokusolver;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class GridRecognizer {

    public GridRecognizer() {

    }

    public Mat HoughLines(Mat image) {
        Mat grayMat = new Mat();
        Mat cannyEdges = new Mat();
        Mat lines = new Mat();

        Imgproc.cvtColor(image, grayMat, Imgproc.COLOR_RGB2GRAY);

        Imgproc.Canny(grayMat, cannyEdges, 10, 100);

        Imgproc.HoughLinesP(cannyEdges, lines, 1, Math.PI/180, 50,20,20);

        return lines;

    }
}
