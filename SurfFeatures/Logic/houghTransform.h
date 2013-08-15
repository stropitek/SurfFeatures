#ifndef HOUGHTRANSFORM_H
#define HOUGHTRANSFORM_H

// STD includes
#include <cassert>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

// OpenCV includes
#include <cv.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/imgproc/imgproc_c.h>
#include "opencv2/nonfree/gpu.hpp"

using namespace std;
using namespace cv;

// ===========================================
// Hough Transform
// ===========================================

#define SCALE_DIFF log((float)1.5f)
#define ORIENTATION_DIFF (20.0f*PI/180.0F)
#define TRANSLATION_ERROR (30.0f/43.0f)
#define PI 3.141592653589793

float angle_radians(float fAngle );

int compatible_poll_booths_line_segment(
  float fTrainingCol,
  float fTrainingRow,
  float fTrainingOri,
  float fTrainingScl,
  float fPollCol,
  float fPollRow,
  float fPollOri,
  float fPollScl,
  float fTranslationErrorThres = TRANSLATION_ERROR,
  float fScaleErrorThres = SCALE_DIFF,
  float fOrientationErrorThres = ORIENTATION_DIFF);


int houghTransform(
  const vector<KeyPoint>& queryKeypoints,
  const vector< vector<KeyPoint> > & trainKeypoints,
  vector<DMatch> & matches,
  int refx, int refy);



int houghTransform(
  const vector<KeyPoint>& queryKeypoints,
  const vector<KeyPoint> & trainKeypoints,
  vector<DMatch> & matches,
  int refx, int refy);

int houghTransform(
  const vector<KeyPoint>& queryKeypoints,
  const vector< vector<KeyPoint> > & trainKeypoints,
  vector< vector<DMatch> > & matches,
  int refx, int refy);

#endif