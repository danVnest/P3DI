//
//  FaceTracker.cpp
//  P3DI
//
//  Created by Satveer Singh on 25/05/12.
//  Copyright (c) 2012 University of Sydney. All rights reserved.
//

#include "FaceTracker.h"

using namespace std;
using namespace cv;

FaceTracker::FaceTracker()
{
    
}

FaceTracker::FaceTracker(int framesBeforeNewFace, String haarDirectory, double facePosThresh, int frameWidth, int frameHeight)
{
    /** Global variables */
    haarcascadeDirectory = haarDirectory;
    face_cascade_name = haarcascadeDirectory+ "haarcascade_frontalface_alt.xml";
    noFacesInWhile = true;
    framesSinceLastFace = 0;
    FRAMES_BEFORE_NEW_FACE = framesBeforeNewFace;
    facePosThreshold = facePosThresh; //The distance that consecutive face positions are allowed between frames to avoid jumping
    facePosition.resize(3);
    FRAME_WIDTH = frameWidth;
    FRAME_HEIGHT = frameHeight; 
}

bool FaceTracker::TryInitFaceTracker(){
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    { 
        printf("--(!)Error loading\n"); 
        return false; 
    }
    return true;
}

vector<double> FaceTracker::getLastFacePosition()
{
    return facePosition;
}

Mat FaceTracker::getLastAnalysedFrame()
{
    return lastFrame;
}


bool FaceTracker::detectFace( Mat frame2analyse )
{
    
    vector<Rect> faces;
    Rect chosenFace;
    Mat frame_gray;
    cvtColor( frame2analyse, frame_gray, CV_BGR2GRAY );
    //frame_gray = frame; // remove if frame is colour
    equalizeHist( frame_gray, frame_gray );
    //imshow("GrayImage",frame_gray);
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    
    if (noFacesInWhile)
    {
        //Get the index of the Largest Face
        double indexLargestFace = 0;
        double widthLargestFace = 0;
        for( int i = 0; i < faces.size(); i++ )
        {
            if (faces[i].width > widthLargestFace)
            {
                indexLargestFace = i;
            }
        }
        if (faces.size() > 0)
        {
            chosenFace = faces[indexLargestFace];
            faceDetected = true;
            lastFace = chosenFace;
            noFacesInWhile = false;
        }
        else
        {
            faceDetected = false;
            framesSinceLastFace += 1;
            if (framesSinceLastFace > FRAMES_BEFORE_NEW_FACE)
            {
                noFacesInWhile = true;
                framesSinceLastFace = 0;
            }
        }
    }
    else
    {
        double closestDistanceToFace = FRAME_WIDTH; //Initialised to a large value
        double distanceToLastFace;
        int indexOfClosestFace=0;
        for( int i = 0; i < faces.size(); i++ )
        {
            double xDisplacement = (faces[i].x - lastFace.x);
            double yDisplacement = (faces[i].y - lastFace.y);
            distanceToLastFace = sqrt(xDisplacement*xDisplacement + yDisplacement*yDisplacement);
            if (distanceToLastFace < closestDistanceToFace)
            {
                closestDistanceToFace = distanceToLastFace;
                indexOfClosestFace = i;
            }
        }
        if (closestDistanceToFace <= facePosThreshold)
        {
            faceDetected = true;
            chosenFace = faces[indexOfClosestFace];
            lastFace = chosenFace;  
        }
        else
        {
            faceDetected = false;
            framesSinceLastFace += 1;
            if (framesSinceLastFace > FRAMES_BEFORE_NEW_FACE)
            {
                noFacesInWhile = true;
                framesSinceLastFace = 0;
            }
        }
    }
    
    if (faceDetected)
    {
        Point center( chosenFace.x + chosenFace.width*0.5, chosenFace.y + chosenFace.height*0.5 );
        Point eyesMiddle( center.x, center.y - chosenFace.height/7 );
        ellipse( frame2analyse, center, Size( chosenFace.width*0.5, chosenFace.height*0.5), 0, 0, 360, Scalar( 255, 0, 0 ), 4, 8, 0 );
        circle(frame2analyse, eyesMiddle, 3, Scalar( 0, 0, 255 ));
        circle(frame2analyse, eyesMiddle, 6, Scalar( 0, 0, 255 ));
        
        facePosition[2] = pixels2distance(chosenFace.width);
        
        facePosition[0] = -(eyesMiddle.x - FRAME_WIDTH/2.0);
        facePosition[1] = -(eyesMiddle.y - FRAME_HEIGHT/2.0);
        
        lastFrame = frame2analyse;
        return true;
    }
    else
    {
        return false;
    }
}

void FaceTracker::mouseHandler(int event, int x, int y, int flags)
{
    switch(event){
        case CV_EVENT_LBUTTONUP:
            lastFace.x = x;
            lastFace.y = y;
            faceDetected = true;
            Rect currentFace = lastFace;
    }
}

double FaceTracker::pixels2distance(int faceWidthPixels)
{
    double distance = DISTANCE_MAIN*pow(faceWidthPixels*480/FRAME_WIDTH, DISTANCE_EXP); //DOUBLE CHECK
    return distance;
}

