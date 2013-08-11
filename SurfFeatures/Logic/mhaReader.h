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

void matchDescriptorsKNN( const Mat& queryDescriptors,\
                         vector<vector<DMatch> >& matches,\
                         Ptr<DescriptorMatcher>& descriptorMatcher,\
                         int k);



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

#endif