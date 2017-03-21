package com.example.calvin.sudokusolver;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Camera;
import android.graphics.Canvas;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;

import org.opencv.android.JavaCameraView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
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


    List<MatOfPoint> contours;
    DigitRecognizer mnist;
    GridRecognizer gridRecognizer;

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

        Mat temp = inputFrame.rgba();
        Core.rectangle(temp, new Point (temp.cols()/2 -100,
                            temp.rows() / 2 - 100), new Point(temp.cols()/2 + 100,
                            temp.rows() / 2 + 100), new Scalar(0,0,0),3);
        Mat digit = temp.submat(temp.rows()/2 - 80, temp.rows() / 2 +
                                80, temp.cols() / 2 - 80, temp.cols() / 2
                                + 80).clone();

        //Core.transpose(digit, digit);

        String s = String.valueOf(mnist.FindMatch(digit));
        Core.putText(temp, s, new Point (temp.cols()/2 -100,
                temp.rows() / 2 - 100), 2, 12.0, new Scalar(0,0,0));

        //Core.transpose(temp, temp);


     /*   Mat lines = gridRecognizer.HoughLines(temp);

        for (int i = 0; i < lines.cols() ; i++) {
            double[] points = lines.get(0,i);
            double x1,y1,x2,y2;

            x1 = points[0];
            y1 = points[1];
            x2 = points[2];
            y2 = points[3];

            Point pt1 = new Point(x1, y1);
            Point pt2 = new Point(x2, y2);

            Core.line(temp, pt1, pt2, new Scalar(255,0,0),2);
        }
*/

        return temp;
    }

}
