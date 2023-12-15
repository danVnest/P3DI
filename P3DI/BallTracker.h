//
//  BallTracker.h
//  P3DI
//
//  Created by Satveer Singh on 6/06/12.
//  Copyright (c) 2012 University of Sydney. All rights reserved.
//

#ifndef P3DI_ballTracker_h
#define P3DI_ballTracker_h

#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"
#include <math.h>
using namespace cv;
using namespace std;

#define DISTANCE_MAIN   9.8219
#define DISTANCE_EXP    -1.024

class BallTracker{
    
private:
    int hueColour;
    int FRAME_WIDTH;
    int FRAME_HEIGHT;
    int valuesAllowed;
    Mat lastFrame;
    vector<double> lastPosition;
    double pixels2distance(int ballRadiusPixels);
    
public:
    BallTracker();
    BallTracker(int hueToTrack, int valuesAllowed, int frameWidth, int frameHeight);
    Mat getLastAnalysedFrame();
    vector<double>getLastPosition();
    bool analyseFrame(Mat img, float minCircleSize);
    
};

#endif
