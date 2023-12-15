//
//  PerspectiveTracker.cpp
//  P3DI
//
//  Created by Satveer Singh on 6/06/12.
//  Copyright (c) 2012 University of Sydney. All rights reserved.
//

#include "PerspectiveTracker.h"
using namespace std;
using namespace cv;

PerspectiveTracker::PerspectiveTracker()
{
    
}

PerspectiveTracker::PerspectiveTracker(SensorMode chosenSensor, TrackingMode chosenForTracking, int frameWidth, int frameHeight, vector<double> camera2ModelVector, int numberFramesBeforeNewObject, String haarCascadeDirectory)
{
    currentSensor = chosenSensor;
    currentlyTracking = chosenForTracking; 
    currentWindowSize = NORMAL;
    FRAME_WIDTH = frameWidth;
    FRAME_HEIGHT = frameHeight;
    CAMERA_2_MODEL_VECTOR = camera2ModelVector;
    frameDiagonal = sqrt(FRAME_WIDTH*FRAME_WIDTH + FRAME_HEIGHT*FRAME_HEIGHT);
    framesBeforeNewObject = numberFramesBeforeNewObject;
    haarcascadeDirectory = haarCascadeDirectory;
    position.resize(3);
    position[0] = 0;
    position[1] = 0;
    position[2] = 1;
    TryInitTrackers();
    TryInitSensor();
    cvNamedWindow("Tracker", CV_WINDOW_NORMAL);
}

bool PerspectiveTracker::TryInitSensor()
{
    if (currentSensor == CAMERA)
    {
        try 
        {
            //-- 2. Read the video stream
            camera = cvCreateCameraCapture( CV_LOAD_IMAGE_UNCHANGED );
            cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); 
            cvSetCaptureProperty(camera, CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);         
            return true;
        } catch (Exception e) 
        {
            return false;
        }
    }
    else
    {
        //DO STUFF DEPENDING ON WHICH CAMERA IS SELECTED
        return true;
    }
    return false;
}

bool PerspectiveTracker::TryInitTrackers()
{
    try
    {
        ballTracker = BallTracker(160,10,FRAME_WIDTH,FRAME_HEIGHT);
        faceTracker = FaceTracker(framesBeforeNewObject, haarcascadeDirectory, frameDiagonal/10, FRAME_WIDTH, FRAME_HEIGHT);
        return faceTracker.TryInitFaceTracker();
    }
    catch (Exception e)
    {
        return false;
    }
}



void PerspectiveTracker::ToggleSensorMode()
{
    if (currentSensor == CAMERA)
    {
        currentSensor = KINECT;
        TryInitSensor();
    }
    else
    {
        currentSensor = CAMERA;
        TryInitSensor();
    }
}

void PerspectiveTracker::ToggleTrackingMode()
{
    if (currentlyTracking == BALL)
    {
        currentlyTracking = FACE;
    }
    else
    {
        currentlyTracking = BALL;
    }
}

vector<double> PerspectiveTracker::GetPosition()
{
    analyseFrame();
    return position;
}

bool PerspectiveTracker::analyseFrame()
{
    if( camera )
    {
        Mat frame = cvQueryFrame( camera ); 
        bool foundObject;
        //-- 3. Apply the classifier to the frame
        if( !frame.empty() )
        { 
            if (currentlyTracking == BALL)
            {
                foundObject = ballTracker.analyseFrame(frame,MIN_CIRCLE_RADIUS);
                if (foundObject)
                {
                    position = ballTracker.getLastPosition();
                    frame = ballTracker.getLastAnalysedFrame();
                }
            }
            else
            {
                foundObject = faceTracker.detectFace(frame);
                if (foundObject)
                {
                    position = faceTracker.getLastFacePosition();
                    frame = faceTracker.getLastAnalysedFrame();
                }

            }
            if (foundObject)
            {
                double mmPerPixel = distance2MMperPixel(position[2]);
                position[0] = position[0]*mmPerPixel/1000;
                position[1] = position[1]*mmPerPixel/1000;
                
                Point bottomLeftRectangle(0,FRAME_HEIGHT);
                Point bottomLeftText(bottomLeftRectangle.x +5,bottomLeftRectangle.y-5);        
                Point topRightRectangle(FRAME_WIDTH, bottomLeftRectangle.y - 1);
                
                char positionText[30];
                rectangle(frame, bottomLeftRectangle, topRightRectangle,Scalar( 0, 0, 0 ),50, 8, 0);
                sprintf(positionText,"[x=%2.4fm, y=%2.4fm, z=%2.4fm]",position[0],position[1],position[2]);
                putText(frame,positionText,bottomLeftText, FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
            }
            
            if (currentWindowSize == FULLSCREEN)
            {
                frame = makeFullscreen(frame);
            }
            imshow("Tracker", frame);
            return true;
        }
        else
        { 
            return false;
        }
    }
    return false;
}

double PerspectiveTracker::distance2MMperPixel(double distanceM)
{
    double mmPerPixel = SIDESTEP_SLOPE*distanceM + SIDESTEP_INTERCEPT;
    return mmPerPixel*480/FRAME_WIDTH; //DOUBLE CHECK
}

void PerspectiveTracker::ToggleWindowSize()
{
    if (currentWindowSize == NORMAL)
    {
        cvSetWindowProperty("Tracker", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        currentWindowSize = FULLSCREEN;
    }
    else
    {
        cvSetWindowProperty("Tracker", CV_WND_PROP_AUTOSIZE, CV_WINDOW_NORMAL);
        currentWindowSize = NORMAL;
    }
}

Mat PerspectiveTracker::makeFullscreen(Mat frame)
{
    IplImage imageIn = frame;
    IplImage* imageInPointer = &imageIn;
    cvGetWindowHandle("Tracker");
    Mat largeImage = cvCreateImage(cvSize(1440, 900), 8, 3);
    IplImage imageOut = largeImage;
    IplImage* imageOutPointer = &imageOut;
    cvResize(imageInPointer, imageOutPointer);
    largeImage = Mat(imageOutPointer);
    return largeImage;
}

//vector<double> PerspectiveTracker::GetLightPosition()
//{
//    Mat frame = cvQueryFrame( camera ); 
//    bool foundObject;
//}
