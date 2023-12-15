//
//  ballTracker.cpp
//  P3DI
//
//  Created by Satveer Singh on 6/06/12.
//  Copyright (c) 2012 University of Sydney. All rights reserved.
//

#include "BallTracker.h"

using namespace std;
using namespace cv;



//Constructors
BallTracker::BallTracker()
{
}

BallTracker::BallTracker(int hueToTrack, int valuesAllowed, int frameWidth, int frameHeight)
{
    hueColour = hueToTrack;
    this->valuesAllowed = valuesAllowed;
    FRAME_WIDTH = frameWidth;
    FRAME_HEIGHT = frameHeight;
}

//Public Methods
bool BallTracker::analyseFrame(Mat img, float minCircleRadius)
{
    IplImage* thresholdedImage;
    CvMemStorage* storage = cvCreateMemStorage(0);
    IplImage hsvIMG;
    IplImage* hsvPointer = &hsvIMG;
    vector<double> result;
    Mat hsvImage;
    cvtColor(img, hsvImage, CV_BGR2HSV_FULL);   //Convert to HSV
    hsvIMG = hsvImage;
    //imshow("hsvWindow",hsvImage);
    
    CvSize size = cvSize(hsvPointer->width,hsvPointer->height);
    thresholdedImage = cvCreateImage(size, 8, 1);
    
    // Attempt Green
    //    CvScalar minColour = cvScalar(108, 130, 90);
    //    CvScalar maxColour = cvScalar(117, 180, 110);
    
    // Orange Nerf Gun Clip
        //CvScalar minColour = cvScalar(235, 100, 100);
        //CvScalar maxColour = cvScalar(260, 255, 255);
    
    // Yellow Puck
    CvScalar minColour = cvScalar(30, 100, 100);
    CvScalar maxColour = cvScalar(45, 255, 255);
    
    // Darker Yellow
//    CvScalar minColour = cvScalar(35, 90, 160);
//    CvScalar maxColour = cvScalar(45, 180, 180);
    
    cvInRangeS(hsvPointer, minColour, maxColour, thresholdedImage);
    
    // Calculate the moments to estimate the position of the ball
    Mat binaryImage = Mat(thresholdedImage);
    CvMoments ourMoment = moments(binaryImage,true);
    
    // The actual moment values
    double moment10 = ourMoment.m10;   
    double moment01 = ourMoment.m01;
    double area = ourMoment.m00;
    
    double positionX = moment10/area;
    double positionY = moment01/area;
    
    Point center( positionX, positionY );
    double circleRadius = sqrt(area/M_PI);
    //circle(binaryImage, center, 3, Scalar( 0, 0, 255 ));
    imshow("binary", binaryImage);
    
    
    IplImage* cannyImage;
    cannyImage = thresholdedImage;
    cvCanny(thresholdedImage, cannyImage, 255, 255, 3);
    
    //cvNamedWindow( "canny", 1 );
    cvShowImage( "canny", cannyImage);
    //cvWaitKey(0);
    
    CvSeq* circles = cvHoughCircles( cannyImage, storage, CV_HOUGH_GRADIENT, 2, thresholdedImage->height/4, 200, 100 );
    
    float* finalCircle;
    bool circlesFound = false;
    bool circlesLargeEnough = false;
    for(int i=0; i<circles->total; i++)
    {
        circlesFound = true;
        float* p = (float*) cvGetSeqElem( circles, i );
        if (p[2] > minCircleRadius)
        {
            finalCircle = p;
            minCircleRadius = p[2];
            center = cvPoint( cvRound( p[0] ), cvRound( p[1] ) );
            circleRadius = p[2];
        }
    }
    if (!circlesFound)
    {
        if (circleRadius > minCircleRadius)
        {
            circlesLargeEnough = true;
        }
    }
    if (circlesLargeEnough) 
    {
        ellipse( img, center, Size( 1 , 1 ), 0, 0, 360, Scalar( 0, 0, 255 ), 4, 8, 0 );
        ellipse( img, center, Size( circleRadius, circleRadius), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );
    }
    
    //cvNamedWindow( "HoughCircles", 1 );
    //cvShowImage( "HoughCircles", hsvPointer);
    //cvWaitKey(0);
    //imshow("HoughCircles",hsvImage);
    lastFrame = img;
    
    result.resize(3);
    result[2] = pixels2distance(circleRadius);
    result[0] = -(positionX - FRAME_WIDTH/2.0);
    result[1] = -(positionY - FRAME_HEIGHT/2.0);
    
    lastPosition = result;
    return circlesLargeEnough;
}

Mat BallTracker::getLastAnalysedFrame()
{
    return lastFrame;
}

vector<double> BallTracker::getLastPosition()
{
    return lastPosition;
}

//Private Methods
double BallTracker::pixels2distance(int ballRadiusPixels)
{
    double distance = DISTANCE_MAIN*pow(ballRadiusPixels*480/FRAME_WIDTH, DISTANCE_EXP); //DOUBLE CHECK
    return distance;
}