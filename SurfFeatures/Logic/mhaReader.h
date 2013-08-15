#ifndef MHA_READER_H
#define MHA_READER_H

// STD includes
#include <cassert>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>

// VTK includes
#include <vtkNew.h>
#include <vtkExtractVOI.h>
#include <vtkImageData.h>
#include <vtkImageExport.h>
#include <vtkImageImport.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkSmartPointer.h>

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

// vnl includes
#include <vnl/vnl_double_3.h>

// Slicer logic
#include "vtkSlicerSurfFeaturesLogic.h"

using namespace std;
using namespace cv;

string getDir(const string& filename);
void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames);
int readImageDimensions_mha(const string& filename, int& cols, int& rows, int& count);

void readImageTransforms_mha(const string& filename,\
                             vector<vector<float> >& transforms,\
                             vector<bool>& transformsValidity,\
                             vector<string>& filenames);

void readImages_mha(const string& filename,\
                    vector<Mat>& images,\
                    vector<int> frames);


// Will read all the data form firstFrame to lastFrame, excluding all slices with invalid transform
int read_mha(const string& filename,\
             vector <Mat>& images,vector<string>& filenames,\
             vector< vector<float> >& transforms,\
             vector<bool>& transformsValidity,\
             int& firstFrame,int& lastFrame,\
             vector<int>& frames);

int read_mha(const string& filename, vector <Mat>& images,\
             vector<string>& filenames,\
             vector< vector<float> >& transforms,\
             vector<bool>& transformsValidity,\
             vector<int> frames);



// ==============================================
// Helpers - conversion functions
// ==============================================
vnl_matrix<double> convertVnlVectorToMatrix(const vnl_double_3& v);
vnl_double_3 convertVnlMatrixToVector(const vnl_matrix<double>& m);
vnl_double_3 arrayToVnlDouble(double arr[4]);
void vnlToArrayDouble(vnl_double_3 v, double arr[4]);
std::vector<float> convertVtkMatrix4x4ToMhaTransform(vtkMatrix4x4* matrix);
void convertVnlMatrixToVtkMatrix4x4(const vnl_matrix<double> vnlMatrix , vtkMatrix4x4* vtkMatrix);
void convertMhaTransformToVtkMatrix(const std::vector<float>& vec, vtkMatrix4x4* vtkMatrix);
vnl_matrix<double> convertMhaTransformToVnlMatrix(const vector<float>& vec);


// ===============================================
// OpenCV
// ===============================================
void computeKeypointsAndDescriptors(const cv::Mat& data,\
                                    const cv::Mat& mask,\
                                    std::vector<cv::KeyPoint> &keypoints,\
                                    cv::Mat &descriptors,\
                                    int minHessian = 400);

void matchDescriptors(const Mat& queryDescriptors, vector<DMatch>& matches,\
                      Ptr<DescriptorMatcher> descriptorMatcher);

void matchDescriptorsKNN( const Mat& descriptors,\
                         vector<vector<DMatch> >& matches,\
                         Ptr<DescriptorMatcher> descriptorMatcher,\
                         int k);

int maskMatchesByMatchPresent( const vector<DMatch>& matches, vector<char>& mask );
int maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask );
vector<char> initMatchesMask(vector<vector<DMatch> >& matches);

cv::Mat getResultImage( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,\
                     const Mat& trainImage, const vector<KeyPoint>& trainKeypoints,\
                     const vector<DMatch>& matches,int iTrainImageIndex);

vector<int> countVotes(const vector<DMatch>& matches, int trainSize);
vector<DMatch> getValidMatches(const vector<vector<DMatch> >& matches);
vector<DMatch> filterValidMatches(const vector<DMatch>& matches);
vector<DMatch> filterSmoothAndThreshold(const vector<int>& votes, const vector<DMatch> matches);
vector<DMatch> filterBogus(const vector<DMatch>& matches, const vector<DMatch>& bmatches, float threshold);
void computeCentroid(const cv::Mat& mask, int& x, int& y);
cv::Mat rotateImage(const Mat& source, double angle);

// =======================================================
// Helpers - geometry functions
// =======================================================
vnl_matrix<double> getRotationMatrix(vnl_double_3 & axis, double theta);
vnl_double_3 projectPoint(vnl_double_3 point, vnl_double_3 normalToPlane, double offset);
void getPlaneFromPoints(vnl_double_3 p1, vnl_double_3 p2, vnl_double_3 p3, vnl_double_3& normal, double& d);
double getAngle(const vnl_double_3& u, const vnl_double_3& v);
void getAxisAndRotationAngle(vnl_double_3 v, vnl_double_3 v_new, vnl_double_3& axis, double& angle);
vnl_double_3 transformPoint(const vnl_double_3 in, vtkMatrix4x4* transform);
double computeMeanDistance(const std::vector<vnl_double_3>& x1, const std::vector<vnl_double_3>& x2);
vnl_double_3 computeTranslation(vnl_double_3 u, vnl_double_3 v, vnl_double_3 w, vnl_double_3 trainCentroid, vnl_double_3 queryCentroid);
void setEstimate(vtkMatrix4x4* estimate, const vnl_double_3& u, const vnl_double_3& v, const vnl_double_3& w, const vnl_double_3& t);

// ===========================================
// Helpers - other
// ===========================================
void computeCorners(std::vector<vnl_double_3>& result, vtkMatrix4x4* transform, int maxX, int maxY);

// =======================================================
// Image manipulation and conversion
// =======================================================
vtkImageData* cropData(vtkImageData* data, float cropRatios[4]);
void cropData(cv::Mat& img, float cropRatios[4]);
cv::Mat convertImage(vtkImageData* image);


void readLines( const string& filename, vector<string>& lines );
void readTab(const string& filename, vector<vector<string> >& tab);
void normalizeDir(std::string& dir);
void splitDir(const std::string& filename, std::string& dir, std::string& file, string& ext);


#endif