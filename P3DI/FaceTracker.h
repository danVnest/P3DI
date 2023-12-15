//
//  FaceTracker.h
//  P3DI
//
//  Created by Satveer Singh on 25/05/12.
//  Copyright (c) 2012 University of Sydney. All rights reserved.
//

#ifndef P3DI_FaceTracker_h
#define P3DI_FaceTracker_h

#include <iostream>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>

#define DISTANCE_MAIN   159.47
#define DISTANCE_EXP    -1.181

using namespace std;
using namespace cv;

class FaceTracker {
    
private:
    /** Variables */
    String haarcascadeDirectory;
    String face_cascade_name;
    String eyes_cascade_name;
    bool noFacesInWhile;
    int framesSinceLastFace;
    double facePosThreshold;
    Rect lastFace;
    int FRAMES_BEFORE_NEW_FACE;
    vector<double> facePosition;
    const char* window_name;
    bool faceDetected;
    int FRAME_WIDTH;
    int FRAME_HEIGHT;
    Mat lastFrame;
    
    /** Function Headers */
    double pixels2distance(int faceWidthPixels);
    
public:
    CascadeClassifier face_cascade;
    
    FaceTracker();
    FaceTracker(int framesBeforeNewFace, String haarDirectory, double facePosThresh, int frameWidth, int frameHeight);
    bool TryInitFaceTracker();
    bool AnalyseFrame(CvCapture*);
    Mat getLastAnalysedFrame();
    vector<double> getLastFacePosition();
    void CreateWindow(const char*);
    void mouseHandler(int event, int x, int y, int flags);
    bool detectFace( Mat frame );
    
};

#endif