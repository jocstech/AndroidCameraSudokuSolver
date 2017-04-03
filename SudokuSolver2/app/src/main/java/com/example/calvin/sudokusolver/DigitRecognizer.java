package com.example.calvin.sudokusolver;

import android.os.Environment;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvKNearest;
import org.opencv.ml.CvSVM;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.ByteBuffer;

public class DigitRecognizer {

    private String images_path = "train-images-idx3-ubyte.idx3";
    private String labels_path = "train-labels-idx1-ubyte.idx1";
    CvKNearest knn;
    CvSVM svm;
    private int total_images = 0;
    private int width;
    private int height;


    public DigitRecognizer() {
        //ReadMNISTData();
    }

    public void ReadMNISTData() {
        File external_storage = Environment.getExternalStorageDirectory();
        File mnist_images_file = new File(external_storage, images_path);
        File mnist_labels_file = new File(external_storage, labels_path);
        FileInputStream images_reader = null;
        FileInputStream labels_reader = null;
        try {
           images_reader = new FileInputStream(mnist_images_file);
           labels_reader = new FileInputStream(mnist_labels_file);
        } catch (FileNotFoundException e){
            e.printStackTrace();
        }
        Mat training_images = new Mat();
        Mat training_labels = new Mat();

        if (images_reader != null) {
            try {
                byte[] header = new byte[16];

                images_reader.read(header, 0, 16);

                // Combining the bytes to form an integer
                ByteBuffer temp = ByteBuffer.wrap(header, 4, 12);
                total_images = temp.getInt();
                width = temp.getInt();
                height = temp.getInt();

                //Total number of pixels in each image
                int px_count = width * height;
                training_images = new Mat(total_images, px_count, CvType.CV_8U);

                //Read each image and store it in an array
                for (int i = 0; i < total_images; i++) {
                    byte[] image = new byte[px_count];
                    images_reader.read(image, 0, px_count);
                    training_images.put(i, 0, image);
                }
                training_images.convertTo(training_images, CvType.CV_32FC1);
                images_reader.close();

            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if (labels_reader != null) {
            // Read Labels
            byte[] labels_data;
            labels_data = new byte[total_images];

            try {
                training_labels = new Mat(total_images, 1, CvType.CV_8U);
                Mat temp_labels = new Mat(1, total_images, CvType.CV_8U);
                byte[] header = new byte[8];
                //read the header
                labels_reader.read(header,0,8);
                // read all labels at once
                labels_reader.read(labels_data, 0, total_images);
                temp_labels.put(0,0,labels_data);

                // take a transpose of the image
                Core.transpose(temp_labels, training_labels);
                training_labels.convertTo(training_labels, CvType.CV_32FC1);
                labels_reader.close();;
            } catch (IOException e) {
                e.printStackTrace();
            }
            knn = new CvKNearest();
            knn.train(training_images, training_labels, new Mat(), false, 10, false);
            //svm = new CvSVM();
            //svm.train(training_images, training_labels);
        }


    }

    int FindMatch(Mat test_image) {
        Imgproc.dilate(test_image, test_image,
                Imgproc.getStructuringElement(Imgproc.CV_SHAPE_CROSS,
                        new Size(3,3)));
        // Resize the image
        Imgproc.resize(test_image, test_image, new Size(width, height));
        // Convert the image to grayscale
        Imgproc.cvtColor(test_image, test_image, Imgproc.COLOR_RGB2GRAY);
        // Adaptive Threshold
        Imgproc.adaptiveThreshold(test_image, test_image, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C,
                Imgproc.THRESH_BINARY_INV, 15,2);

        Mat test = new Mat(1, test_image.rows() *
                        test_image.cols(), CvType.CV_32FC1);
        int count = 0;
        for (int i = 0 ; i < test_image.rows(); i++) {
            for (int j = 0; j < test_image.cols(); j++) {
                test.put(0, count, test_image.get(i, j)[0]);
                count++;
            }
        }

        Mat results = new Mat(1, 1, CvType.CV_8U);

        knn.find_nearest(test, 10, results, new Mat(), new Mat());
        Log.i("Result:", "" + results.get(0,0)[0]);

        return (int)(results.get(0,0)[0]);
    }
}
