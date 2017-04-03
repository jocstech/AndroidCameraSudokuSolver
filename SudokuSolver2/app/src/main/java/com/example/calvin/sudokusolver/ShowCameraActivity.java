package com.example.calvin.sudokusolver;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Camera;
import android.graphics.Canvas;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.JavaCameraView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class ShowCameraActivity extends AppCompatActivity implements CvCameraViewListener2 {

    // Used for logging success or failure messages
    private static final String TAG = "OCVSample::Activity";

    // Loads camera view of OpenCV for us to use. This lets us see using OpenCV
    //private CameraBridgeViewBase mOpenCvCameraView;
    private PortraitCameraView mOpenCvCameraView;

    // Used in Camera selection from menu (when implemented)
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;

    private Mat                    mRgba;
    private Mat                    mIntermediateMat;
    private Mat                    mGray;
    Mat hierarchy;

    Bitmap cropBM;
    Mat cropped;
    Point points[];
    Mat sudoku;

    List<MatOfPoint> contours;
    DigitRecognizer mnist;
    GridRecognizer gridRecognizer;

    String digit_result;
    boolean running;
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mnist = new DigitRecognizer();
                    gridRecognizer = new GridRecognizer();
                    mnist.ReadMNISTData();
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public ShowCameraActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        //getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_show_camera);

        mOpenCvCameraView = (PortraitCameraView) findViewById(R.id.camera);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

        //mOpenCvCameraView.setMaxFrameSize(400, 400);


    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_10, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        digit_result = "-1";
        running = false;
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        hierarchy = new Mat();
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
        hierarchy.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        sudoku = inputFrame.gray();
        Mat temp = inputFrame.gray();
        Mat blur = new Mat();
        Imgproc.GaussianBlur(temp, blur, new Size(5,5), 0);
        Mat thresh = new Mat();
        Imgproc.adaptiveThreshold(blur, thresh, 255,1,1,11,2);

        contours = new ArrayList<>();
        hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();

        MatOfPoint2f biggest = new MatOfPoint2f();
        double max_area = 0;
        for (MatOfPoint i : contours) {
            double area = Imgproc.contourArea(i);
            if (area > 100) {
                MatOfPoint2f m = new MatOfPoint2f(i.toArray());
                double peri = Imgproc.arcLength(m, true);
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(m, approx, 0.02 * peri, true);
                if (area > max_area && approx.total() == 4) {
                    biggest = approx;
                    max_area = area;
                }
            }
        }
        Mat copyImage = inputFrame.rgba();
        points = biggest.toArray();
        Log.i("Testing", biggest.total() + "");
        cropped = new Mat();
        if (points.length >= 4) {
            Rect R = new Rect(new Point(points[0].x, points[0].y), new Point(points[2].x, points[2].y));
            cropped = new Mat(copyImage, R);
            cropBM = Bitmap.createBitmap(cropped.cols(), cropped.rows(),Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(cropped, cropBM);

            Core.line(copyImage, new Point(points[0].x, points[0].y), new Point(points[1].x, points[1].y), new Scalar(255, 0, 0), 2);
            Core.line(copyImage, new Point(points[1].x, points[1].y), new Point(points[2].x, points[2].y), new Scalar(255, 0, 0), 2);
            Core.line(copyImage, new Point(points[2].x, points[2].y), new Point(points[3].x, points[3].y), new Scalar(255, 0, 0), 2);
            Core.line(copyImage, new Point(points[3].x, points[3].y), new Point(points[0].x, points[0].y), new Scalar(255, 0, 0), 2);
        }
        //mRgba.release();
       /* mGray.release();
        mIntermediateMat.release();

        mRgba = inputFrame.gray();
        contours = new ArrayList<MatOfPoint>();
        hierarchy = new Mat();

        //Imgproc.Canny(mRgba, mIntermediateMat, 80, 100);
        Imgproc.threshold(mRgba, mIntermediateMat, 60, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.findContours(mIntermediateMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();
        for (MatOfPoint cnt : contours) {
            Rect r = Imgproc.boundingRect(cnt);
            Point pt1 = new Point(r.x, r.y);
            Point pt2 = new Point(r.x + r.width, r.y + r.height);
            Scalar color = new Scalar(0,0,255);
            Imgproc.rectangle(mRgba, pt1, pt2, color, 3);
        }*/

 /*       Mat temp = inputFrame.rgba();
        Core.rectangle(temp, new Point (temp.cols()/2 -100,
                            temp.rows() / 2 - 100), new Point(temp.cols()/2 + 100,
                            temp.rows() / 2 + 100), new Scalar(0,0,0),3);
        if (!running) {
            running = true;
            Mat digit = temp.submat(temp.rows() / 2 - 80, temp.rows() / 2 +
                    80, temp.cols() / 2 - 80, temp.cols() / 2
                    + 80).clone();

            //Core.transpose(digit, digit);

            //String s = String.valueOf(mnist.FindMatch(digit));

            new digitTask().execute(digit);

            Core.putText(temp, digit_result, new Point(temp.cols() / 2 - 100,
                    temp.rows() / 2 - 100), 2, 12.0, new Scalar(0, 0, 0));
        }
*/
        //Core.transpose(temp, temp);

/*
       Mat lines = gridRecognizer.HoughLines(cropped);

        for (int i = 0; i < lines.cols() ; i++) {
            double[] points = lines.get(0,i);
            double x1,y1,x2,y2;

            x1 = points[0];
            y1 = points[1];
            x2 = points[2];
            y2 = points[3];

            Point pt1 = new Point(x1, y1);
            Point pt2 = new Point(x2, y2);

            Core.line(copyImage, pt1, pt2, new Scalar(255,0,0),2);
        }
*/

        return copyImage;
    }

    private class digitTask extends AsyncTask<Mat, Void, String> {

        protected String doInBackground(Mat... mat) {
            return String.valueOf(mnist.FindMatch(mat[0]));
        }

        protected void onPostExecute(String result) {
            digit_result = result;
            running = false;
        }
    }

    public void solve(View v) {
        mOpenCvCameraView.setVisibility(View.GONE);
        ImageView iv = (ImageView) findViewById(R.id.solve_img);
        iv.setVisibility(View.VISIBLE);
        //Log.i("Testing", cropBM.getWidth() + " ; " + cropBM.getHeight());
        iv.setImageBitmap(cropBM);

        Imgproc.GaussianBlur(cropped, cropped,new Size(11,11),0);

        Mat lines = gridRecognizer.HoughLines(cropped);

        for (int i = 0; i < lines.cols() ; i++) {
            double[] points = lines.get(0, i);
            double x1, y1, x2, y2;

            x1 = points[0];
            y1 = points[1];
            x2 = points[2];
            y2 = points[3];

            Point pt1 = new Point(x1, y1);
            Point pt2 = new Point(x2, y2);

            Core.line(cropped, pt1, pt2, new Scalar(255, 0, 0), 2);
        }

        Bitmap bm = Bitmap.createBitmap(cropped.cols(), cropped.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(cropped, bm);
        iv.setImageBitmap(bm);

    }

}
