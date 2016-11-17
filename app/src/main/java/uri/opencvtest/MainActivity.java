package uri.opencvtest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.content.Context;

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
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private static final String TAG = "MainActivity";

    private CameraBridgeViewBase mOpenCvCameraView;

    Mat displayImage;
    Mat flippedImage;
    Mat transposedImage;
    Mat resizedImage;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){ //depending on the status passed to onManagerConnected()
                case LoaderCallbackInterface.SUCCESS: //if OpenCV works
                    Log.i(TAG,"OpenCV loaded successfully"); //debug message
                    mOpenCvCameraView.enableView(); //this enables the camera view
                    break;
                default: //if Uri implemented things wrong
                    super.onManagerConnected(status); //dunno
                    break;
            }
        }
    };

    public void MainActivity_show_camera(){ //dunno
        Log.i(TAG, "Instantiated new " + this.getClass()); //debug message
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) { //when the window is created
        Log.i(TAG, "called onCreate"); //debug message
        super.onCreate(savedInstanceState); //line put in by Android Studio
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON); //sets a flag in the app window to keep the screen on
        setContentView(R.layout.activity_main); //sets content view to main activity

        mOpenCvCameraView = (JavaCameraView)findViewById(R.id.show_camera_activity_java_surface_view); //assigns the camera view object
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE); //sets camera view to be visible
        mOpenCvCameraView.setCvCameraViewListener(this); //sets the main class as the camera view listener
    }

    @Override
    public void onPause(){
        super.onPause(); //calls the parent method
        if (mOpenCvCameraView != null){ //if the camera view has been loaded
            mOpenCvCameraView.disableView(); //this disables the camera view
        }
    }

    @Override
    public void onResume(){
        super.onResume(); //calls the parent method
        if (OpenCVLoader.initDebug()){ //if the OpenCV library loaded
            Log.d(TAG, "OpenCV library found inside package. Using it!"); //debug message
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS); //dunno
        } else {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization"); //debug message
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9, this, mLoaderCallback); //dunno
        }
    }

    public void onDestroy(){
        super.onDestroy(); //calls the parent method
        if (mOpenCvCameraView != null){ //if the camera view has been loaded
            mOpenCvCameraView.disableView(); //this disables the camera view
        }
    }

    public void onCameraViewStarted(int width, int height){ //when the camera view starts or restarts (after re-orientation of camera)
        //These lines set dimensions and properties for the image objects:
        displayImage = new Mat(height, width, CvType.CV_8UC4);
        flippedImage = new Mat(height, width, CvType.CV_8UC4);
        transposedImage = new Mat(width, width, CvType.CV_8UC4);
        resizedImage = new Mat(height, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        displayImage.release(); //this releases the object
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame){
        int screenRotation = (((WindowManager)getApplicationContext().getSystemService(Context.WINDOW_SERVICE)).getDefaultDisplay()).getRotation(); //gets the screen rotation

        Mat originalImage = inputFrame.rgba(); //sets the original image to the current frame

        switch (screenRotation){
            case Surface.ROTATION_0: //the device is in portrait mode
                Log.d(TAG, "Phone is oriented normally"); //debug message
                Core.transpose(originalImage, transposedImage); //transposes the image to correct it
                Imgproc.resize(transposedImage, resizedImage, resizedImage.size()); //resizes transposed image
                Core.flip(resizedImage, displayImage, 1); //flips resized image, outputs final image
                break;
            case Surface.ROTATION_90: //the device was turned counterclockwise to landscape mode, doesn't need to transpose because the orientation is compensating
                Log.d(TAG, "Phone is oriented 90 degrees counterclockwise"); //debug message
                Imgproc.resize(originalImage, displayImage, displayImage.size()); //resizes the image, outputs final image
                break;
            case Surface.ROTATION_180: //the device is upside down (dafuq?)
                Log.d(TAG, "Phone is oriented upside down, what the hell?"); //debug message
                displayImage = originalImage;
                break;
            case Surface.ROTATION_270: //the device was turned clockwise to landscape mode
                Log.d(TAG, "Phone is oriented 90 degrees clockwise"); //debug message
                Imgproc.resize(originalImage, resizedImage, resizedImage.size()); //resizes the image
                Core.flip(resizedImage, flippedImage, 0); //flips the image vertically
                Core.flip(flippedImage, displayImage, 1); //flips the image horizontally, outputs final image
                break;
        }

        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.gears_cropped); //gets the template picture
        Mat template = new Mat(); //creates template Mat object
        Utils.bitmapToMat(bitmap, template); //converts image to Mat object

        distanceFromCamera(template); //calls the function

        return displayImage; //returns the final image
    }

    public double distanceFromCamera(Mat template){ //returns the distance of the template object from the camera

        /*
        F = (P * D) / W
        F is the perceived focal length of the camera, this should be constant
        P is the perceived width of the object in pixels
        D is the distance from the camera, in centimeters
        W is the real width of the object, in centimeters

        D' = (W * F) / P
        D' is the calculated distance
        */

        final int matchMethod = Imgproc.TM_CCORR; //the match method used in matchTemplate() is assigned here
        Mat image = new Mat(); //this initializes the image
        Imgproc.cvtColor(displayImage, image, Imgproc.COLOR_BGR2GRAY); //converts image to black and white to compare
        Imgproc.Canny(image, image, 50, 200);

        double maximumCorrelationValue = 0; //this will track how well the template matches the image, at its best point
        Point maximumLocation = new Point(); //this is where the template matches best
        double templateScaleFactor = 0; //this is how scaled the template was when it matched best

        Mat scaledTemplate = new Mat();
        {
            int i;
            double xSize = 0;
            double ySize = 0;
            for (i = 0; i <= 100; i++) { //this resizes the template to the image size
                double xSizeTemp = template.cols() * (double)i/100;
                double ySizeTemp = template.rows() * (double)i/100;
                if (xSizeTemp > image.cols() || ySizeTemp > image.rows()) {
                    break;
                } else {
                    xSize = xSizeTemp;
                    ySize = ySizeTemp;
                }
            }
            Imgproc.resize(template, scaledTemplate, new Size(xSize, ySize)); //resizes the template
            Log.d(TAG, "scaledTemplate xSize = " + xSize);
            Log.d(TAG, "scaledTemplate ySize = " + ySize);
        }

        for (int i = 10; i <= 100; i += 10){ //this iterates through various zoom levels to find the best one

            if (template.empty()){ //if the image didn't load
                Log.d(TAG, "Template is blank"); //debug message
                return -1; //error return value signifying blank template
            }

            Mat resultImage = new Mat(); //creates the result matrix
            Mat resizedTemplate = new Mat(); //creates the image to store the resized template

            {
                double xSize = (double)i/100 * scaledTemplate.cols(); //the x dimension of the new template
                double ySize = (double)i/100 * scaledTemplate.rows(); //the y dimension of the new template
                Log.d(TAG, "resizedTemplate xSize = " + xSize); //debug message
                Log.d(TAG, "resizedTemplate ySize = " + ySize); //debug message
                Imgproc.resize(template,  resizedTemplate, new Size(xSize, ySize)); //this resizes the template, look at the freakin' line, what else could it do???
            }

            if (resizedTemplate.rows() > image.rows() ||  resizedTemplate.cols() > image.cols()){ //if the template is bigger than the image
                break; //exits the loop
            }

            Imgproc.cvtColor( resizedTemplate,  resizedTemplate, Imgproc.COLOR_BGR2GRAY); //converts template to black and white
            Imgproc.Canny( resizedTemplate,  resizedTemplate, 50, 200); //detects edges in the template

            int resultRows = image.rows() -  resizedTemplate.rows(); //defines the number of rows for the result with the frame and template
            int resultColumns = image.cols() -  resizedTemplate.cols(); //defines the number of columns for the result with the frame and template
            resultImage.create(resultRows, resultColumns, image.type()); //this creates the result image matrix

            Imgproc.matchTemplate(image, resizedTemplate, resultImage, matchMethod); //this tries to match the template
            Core.normalize(resultImage, resultImage, 0, 1, Core.NORM_MINMAX, -1); //normalizes the image

            Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(resultImage); //this finds the highest matching points

            if (minMaxLocResult.maxVal > maximumCorrelationValue){ //if a new maximum correlation coefficient is found
                maximumCorrelationValue = minMaxLocResult.maxVal; //reset the tracking variable
                maximumLocation = minMaxLocResult.maxLoc; //sets the location to that of the highest matching point
                templateScaleFactor = (double)i/100; //sets the scale factor
            }
        }

        Point startRectangle = maximumLocation; //start of marking rectangle
        double endX = startRectangle.x + scaledTemplate.cols() * templateScaleFactor; //calculates end of rectange x coordinate
        double endY = startRectangle.y + scaledTemplate.rows() * templateScaleFactor; //calculates end of rectangle y coordinate
        Point endRectangle = new Point(endX, endY); //this sets the end of the rectangle

        Log.d(TAG, "templateScaleFactor = " + templateScaleFactor);
        Log.d(TAG, "startRectangle.x = " + startRectangle.x);
        Log.d(TAG, "startRectangle.y = " + startRectangle.y);
        Log.d(TAG, "endRectangle.x = " + endRectangle.x);
        Log.d(TAG, "endRectangle.y = " + endRectangle.y);

        Core.rectangle(displayImage, startRectangle, endRectangle, new Scalar(0, 0, 255)); //this displays the rectangle
        return 0;
    }
}