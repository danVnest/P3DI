//
//  PerspectiveTracker.h
//  P3DI
//
//  Created by Satveer Singh on 6/06/12.
//  Copyright (c) 2012 University of Sydney. All rights reserved.
//

#ifndef P3DI_PerspectiveTracker_h
#define P3DI_PerspectiveTracker_h

#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "BallTracker.h"
#include "FaceTracker.h"

using namespace std;
using namespace cv;

#define SIDESTEP_SLOPE  1.9275
#define SIDESTEP_INTERCEPT  0.2359

#define MIN_CIRCLE_RADIUS 5

enum SensorMode{
    CAMERA,
    KINECT
};

enum TrackingMode{
    BALL,
    FACE
};

enum WindowSize{
    NORMAL,
    FULLSCREEN
};

class PerspectiveTracker {
private:
    SensorMode currentSensor;
    TrackingMode currentlyTracking;
    WindowSize currentWindowSize;
    int FRAME_WIDTH;
    int FRAME_HEIGHT;
    vector<double> CAMERA_2_MODEL_VECTOR;
    double frameDiagonal;
    CvCapture* camera;
    BallTracker ballTracker;
    FaceTracker faceTracker;
    int framesBeforeNewObject;  
    vector<double> position;
    String haarcascadeDirectory;
    
    bool TryInitSensor();
    bool TryInitTrackers();
    bool analyseFrame();
    double distance2MMperPixel(double distanceM);
    Mat makeFullscreen(Mat frame);
    
public:
    PerspectiveTracker();
    PerspectiveTracker(SensorMode chosenSensor, TrackingMode chosenForTracking, int frameWidth, int frameHeight, vector<double> camera2ModelVector, int numberFramesBeforeNewObject, String haarcascadeDirectory);
    vector<double>GetPosition();
    void ToggleSensorMode();
    void ToggleTrackingMode();
    void ToggleWindowSize();
    vector<double> GetLightPosition();
    
};

#endif
