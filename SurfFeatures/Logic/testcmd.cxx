#include "vtkSlicerSurfFeaturesLogic.h"

// ITK includes
#include <itkImageFileReader.h>
#include <itkCastImageFilter.h>
#include <itkNiftiImageIO.h>
#include "itkImageToVTKImageFilter.txx"

// VTK includes
#include <vtkImageReslice.h>

// STL includes
#include <iostream>

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

#include "mhaReader.h"
#include "surfUtil.h"
#include "houghTransform.h"

#define PI 3.141592653589793

using namespace std;

// itk typedefs
typedef float PixelType;
typedef unsigned char UCharPixelType; 
const unsigned int    Dimension = 3;
typedef itk::Image<PixelType, Dimension> ImageType;
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::Image<UCharPixelType, Dimension> UCharImageType;
typedef itk::ImageToVTKImageFilter<ImageType> imageToVTKImageFilterType;
typedef itk::CastImageFilter< ImageType, UCharImageType > CastFilterType;


// Help functions
void writeKeypointsFromMhaFile(string mhaFile, string outputDir, int minHessian);
int compareOriginalToResliced(const Mat& original, const Mat& resliced, const Mat& maskOriginal, const Mat& maskResliced);
int resliceCastAndCompareToOriginal(vtkImageData* data3d, vnl_matrix<double> cosines, vnl_matrix<double> axisOrigin, const Mat& original, const Mat& maskOriginal);

// Main functions
int mainWriteKeypoints();
int mainTransformResilience();
int mainViewMatches(string scenario);

int main()
{
  //return mainWriteKeypoints();
  //return mainTransformResilience();
  return mainViewMatches("C:\\Users\\DanK\\MProject\\data\\Results\\scenarios\\test_scenarios.txt");
}

int mainViewMatches(string scenario)
{
  vector<string> outputFiles;
  vector<string> bogusFiles;
  vector<string> queryFiles;
  vector<vector<string> > trainFiles;
  //vector<string> trainFiles;
  vector<vector<string> > tab;
  readTab(scenario, tab);

  //for(int i=0; i<tab.size(); i++)
  //{
  //  bogusFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\save\\sweep_1a");
  //  trainFiles.push_back(tab[i][0]);
  //  queryFiles.push_back(tab[i][1]);
  //  outputFiles.push_back(tab[i][2]);
  //}
  
  for(int i=0; i<tab.size(); i++)
  {
    trainFiles.push_back(vector<string>(tab[i].begin(), tab[i].end()-2));
    bogusFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\save\\sweep_1a");
    queryFiles.push_back(*(tab[i].end()-2));
    outputFiles.push_back(*(tab[i].end()-1));
  }

  for(int i=0; i<outputFiles.size(); i++)
  {
    vtkSlicerSurfFeaturesLogic* surf = vtkSlicerSurfFeaturesLogic::New();
    surf->loadKeypointsAndDescriptors(bogusFiles[i], "bogus");
    surf->loadKeypointsAndDescriptors(queryFiles[i],"query");
    surf->loadSeveralTrainKeypointsAndDescriptors(trainFiles[i]);
    //surf->loadKeypointsAndDescriptors(trainFiles[i],"train");
    computeCorrespondences(surf);
    surf->simulateAll();
    vector<vector<DMatch> > matches = surf->getAfterHoughMatches();
    int count = 0;
    bool stop = false;
    bool display = false;
    while(true)
    {
      if(count >= matches.size())
        break;
      string input;
      cin >> input;
      for(int nchar=0; nchar<input.size(); nchar++)
      {
        display = false;
        if(input[nchar] == 'n')
        {
          count++;
          if(count >= surf->getQuerySize())
            count = 0;
          continue;
        }
        else if(input[nchar] == 'p')
        {
          count--;
          if(count < 0)
            count = surf->getQuerySize()-1;
          continue;
        }
        else if(input[nchar] == 'q'){
          stop = true;
          break;
        }
        else if(input[nchar] == 'd'){-
          display = true;
        }
      }
      if(stop)
        break;


      vector<int> votes = countVotes(matches[count], surf->getTrainSize());

      cout << endl << "Query image # " << count << endl << "===============" << endl;
      printVotes(votes);
      cout << "Diff between estimation and tracking: " << surf->getPlaneDistances()[count] << " mm" << endl;

      
      if(!display)
        continue;
 
      Mat queryImage = surf->getQueryImage(count);
      for(int tidx=0; tidx<votes.size(); tidx++)
      {
        if(votes[tidx] == 0)
          continue;
        // Display image
        Mat matchesImg = getResultImage(surf->getQueryImage(count), surf->getQueryKeypoints(count), surf->getTrainImage(tidx), surf->getTrainKeypoints(tidx), matches[count], tidx);
        cvStartWindowThread();
        cv::imshow("Matches", matchesImg );
        int key = cv::waitKey(0);
        cvDestroyWindow("Matches");
        cvDestroyWindow("Matches");
        if(key==113) // ascii for 'q'
          break;
      }
    }

    surf->Delete();
  }
  return 0;
}

int mainWriteKeypoints()
{
  vector<string> files;
  vector<string> outputDirs;
  readLines("C:\\Users\\DanK\\MProject\\data\\MNI_AMIGO_mha.txt",files);
  readLines("C:\\Users\\DanK\\MProject\\data\\MNI_AMIGO_dirs.txt",outputDirs);
  int minHessian = 400;

  if(files.size() != outputDirs.size())
    return 1;

  for(int i=0; i<files.size(); i++)
    writeKeypointsFromMhaFile(files[i], outputDirs[i], minHessian);
  return 0;
}

int mainTransformResilience()
{

    // read a slice from an mha file
  string filename = "C:\\Users\\DanK\\MProject\\data\\MNI\\group1\\01\\pre\\sweep_1a\\sweep_1a.mha";
  vector<Mat> images;
  vector<string> filenames;
  vector<bool> transformsValidity;
  vector<vector<float> > transforms;
  vector<int> frames;
  frames.push_back(100);
  read_mha(filename, images, filenames, transforms, transformsValidity, frames);
  if(!transformsValidity[0])
    return 1;
  vnl_matrix<double> transformMatrix = convertMhaTransformToVnlMatrix(transforms[0]);
  vnl_matrix<double> cosines = transformMatrix;
  cosines = cosines.normalize_columns().extract(3,3);
  vnl_matrix<double> frameOrigin = transformMatrix.extract(3,1,0,3);
  vnl_matrix<double> transducerPixel(4,1);
  transducerPixel.fill(1);
  transducerPixel(0,0) = 314;
  transducerPixel(1,0) = 415;
  transducerPixel(2,0) = 0;
  vnl_matrix<double> temp = transformMatrix*transducerPixel;
  vnl_matrix<double> transducerOrigin(3,1);
  transducerOrigin(0,0) = temp(0,0);
  transducerOrigin(1,0) = temp(1,0);
  transducerOrigin(2,0) = temp(2,0);
  

  

  // Read in the 3d image
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO(itk::NiftiImageIO::New());
  reader->SetFileName("C:/Users/DanK/MProject/data/MNI/analyze3dus/1a.3dus.nii");
  reader->Update();
  ImageType::Pointer itkImage = reader->GetOutput();
 
  vnl_matrix<itk::ImageBase<Dimension>::DirectionType::InternalMatrixType::element_type> direction = itkImage->GetDirection().GetVnlMatrix();
  itk::ImageBase<Dimension>::PointType origin = itkImage->GetOrigin();
  itk::ImageBase<Dimension>::SpacingType spacing = itkImage->GetSpacing();
  vnl_matrix_fixed<double, Dimension+1, Dimension+1> intermediateMatrix;
  vnl_matrix_fixed<double, Dimension+1, Dimension+1> directionMatrix;
  vnl_matrix_fixed<double, Dimension+1, Dimension+1> LPStoRASMatrix;
  vnl_matrix_fixed<double, Dimension+1, Dimension+1> IJKtoRASMatrix;
  vnl_vector_fixed<double, Dimension+1> spacingVec;
  vnl_vector_fixed<double, Dimension+1> originVec;
  originVec.fill(1);
  originVec.update(origin.Get_vnl_vector());
  spacingVec.fill(1);
  spacingVec.update(spacing.Get_vnl_vector());
  intermediateMatrix.set_identity();
  intermediateMatrix.set_diagonal(spacingVec);
  intermediateMatrix.set_column(3,originVec.data_block());
  directionMatrix.set_identity();
  directionMatrix.update(direction);
  LPStoRASMatrix.set_identity();
  LPStoRASMatrix(0,0) = -1;
  LPStoRASMatrix(1,1) = -1;
  IJKtoRASMatrix = LPStoRASMatrix * (intermediateMatrix * directionMatrix);


  // Convert the volume from itk to vtk
  imageToVTKImageFilterType::Pointer toVtkFilter = imageToVTKImageFilterType::New();
  toVtkFilter->SetInput(reader->GetOutput());
  toVtkFilter->Update();
  vtkSmartPointer<vtkImageData> data = vtkSmartPointer<vtkImageData>::New();
  data->DeepCopy(toVtkFilter->GetOutput());
  int* dim;
  dim = data->GetDimensions();
  data->SetSpacing(IJKtoRASMatrix.get_diagonal().extract(3).data_block());
  data->SetOrigin(IJKtoRASMatrix.get_column(3).extract(3).data_block());

  Mat mniMask = cv::imread("C:\\Users\\DanK\\MProject\\data\\MNI\\mni_mask.png", CV_LOAD_IMAGE_GRAYSCALE);

  // Out of plane translation
  cout << "Out of plane translation...";
  vector<float> translations;
  vector<int> houghVotesTranslation;
  for(float translation = -5; translation<=-15.; translation+=0.5)
  {
    translations.push_back(translation);
    vnl_matrix<double> frameCenter = frameOrigin + transformMatrix.extract(3,1)*images[0].cols/2 + transformMatrix.extract(3,1,0,1)*images[0].rows/2;
    frameCenter += cosines.extract(3,1,0,2) * translation;
    houghVotesTranslation.push_back(resliceCastAndCompareToOriginal(data, cosines, frameCenter, images[0], mniMask));
  }

  // In plane rotation
  cout << "Inplane rotations...\n";
  vector<float> inplaneRotations;
  vector<int> houghVotesInplaneRotations;
  for(float angle=-PI ; angle<-10; angle+=PI/8.)
  {
    inplaneRotations.push_back(angle);
    vnl_double_3 zCosine = vnl_double_3(cosines(0,2),cosines(1,2), cosines(2,2));
    vnl_matrix<double> rotation = getRotationMatrix(zCosine, angle);
    vnl_matrix<double> newCosines =  rotation * cosines;
    cv::Mat rotated = rotateImage(images[0], angle*180./PI);
    cv::Mat rotatedMask = rotateImage(mniMask, angle*180./PI);
    houghVotesInplaneRotations.push_back(compareOriginalToResliced(images[0],rotated, mniMask, rotatedMask));
    //houghVotesInplaneRotations.push_back(resliceCastAndCompareToOriginal(data, newCosines, frameOrigin, images[0], mniMask));
  }

  // Out of plane rotation (around x axis)
  cout << "Out of plane around x" << endl;
  vector<float> outOfPlaneRotations;
  vector<int> houghVotesOutOfPlaneRotations;
  for(float angle=-PI/8; angle<=PI/8; angle+=PI/24)
  {
    outOfPlaneRotations.push_back(angle);
    // Out of plane rotation around x axis, centered
    vnl_double_3 xCosine = vnl_double_3(cosines(0,0),cosines(1,0), cosines(2,0));
    vnl_matrix<double> rotation = getRotationMatrix(xCosine, angle);
    vnl_matrix<double> newCosines =  rotation * cosines;
    houghVotesOutOfPlaneRotations.push_back(resliceCastAndCompareToOriginal(data, newCosines, transducerOrigin, images[0], mniMask));
  }

  return 0;

}

void writeKeypointsFromMhaFile(string mhaFile, string outputDir, int minHessian)
{
  vtkSlicerSurfFeaturesLogic* surf = vtkSlicerSurfFeaturesLogic::New();
  surf->setQueryFile(mhaFile);
  surf->setMinHessian(minHessian);
  surf->setQueryStartFrame(0);
  surf->setQueryStopFrame(-1);
  surf->setRansacMargin(1.5);
  computeQuery(surf);
  surf->saveKeypointsAndDescriptors(outputDir,"query");

  // Cropping, calibration matrix and mask will be automatically set
  surf->Delete();
}

vnl_matrix<double> computeReslicedTransform(vtkImageData* reslicedImage, const vnl_matrix<double>& cosines, const vnl_matrix<double>& axisOrigin)
{
  vtkIdType id = reslicedImage->FindPoint(0.,0.,0.);
  double x[3] = {0.,0.,0.};
  double pcoords[3];
  int ijk[3];
  int status = reslicedImage->ComputeStructuredCoordinates(x,ijk,pcoords);
  if(status == 0)
    cout << "The reslice axis origin is outside of the 3d volume..." << endl;
  vnl_matrix<double> newOrigin(3,1,0.);
  newOrigin -= cosines.extract(3,1)*ijk[0]*reslicedImage->GetSpacing()[0];
  newOrigin -= cosines.extract(3,1,0,1)*ijk[1]*reslicedImage->GetSpacing()[1];
  newOrigin += axisOrigin;

  vnl_matrix<double> IJKtoRASResliced;
  IJKtoRASResliced.set_size(4,4);
  IJKtoRASResliced(0,0) *= reslicedImage->GetSpacing()[0];
  IJKtoRASResliced(1,0) *= reslicedImage->GetSpacing()[0];
  IJKtoRASResliced(2,0) *= reslicedImage->GetSpacing()[0];
  IJKtoRASResliced(0,1) *= reslicedImage->GetSpacing()[1];
  IJKtoRASResliced(1,1) *= reslicedImage->GetSpacing()[1];
  IJKtoRASResliced(2,1) *= reslicedImage->GetSpacing()[1];
  IJKtoRASResliced(0,2) *= reslicedImage->GetSpacing()[2];
  IJKtoRASResliced(1,2) *= reslicedImage->GetSpacing()[2];
  IJKtoRASResliced(2,2) *= reslicedImage->GetSpacing()[2];
  IJKtoRASResliced(0,3) = 0;
  IJKtoRASResliced(1,3) = 0;
  IJKtoRASResliced(2,3) = 0;
  IJKtoRASResliced(3,3) = 1.;
  IJKtoRASResliced.update(newOrigin, 0,3);
  cout << "Transform associated to the resliced image: \n";
  IJKtoRASResliced.print(cout);
  return IJKtoRASResliced;
}

int compareOriginalToResliced(const Mat& original, const Mat& resliced, const Mat& maskOriginal, const Mat& maskResliced)
{
  // Database, the resliced image
  vector<cv::KeyPoint> keypoints;
  cv::Mat descriptor;
  computeKeypointsAndDescriptors(resliced,maskResliced,keypoints,descriptor);
  vector<cv::Mat> descriptors;
  descriptors.push_back(descriptor);
  Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("FlannBased");
  descriptorMatcher->clear();
  descriptorMatcher->add(descriptors);

  // Query, the original image
  vector<cv::KeyPoint> queryKeypoints;
  cv::Mat queryDescriptors;
  computeKeypointsAndDescriptors(original, maskOriginal, queryKeypoints, queryDescriptors);

  // Find matches
  vector<vector<DMatch> > mmatches;
  int k = 1;
  matchDescriptorsKNN(queryDescriptors, mmatches, descriptorMatcher, k);
  vector<DMatch> matches = getValidMatches(mmatches);

  // Hough filter
  vector<DMatch> houghMatches = matches;
  int xc,yc;
  computeCentroid(maskOriginal,xc,yc);
  int iMatchCount = houghTransform( queryKeypoints, keypoints, houghMatches, xc, yc );
  vector<int> houghVotes = countVotes(houghMatches, 1);
  cout << houghVotes[0] << " correspondences found after hough transform." << endl;


  // Smoothing step makes no sense here

  // Draw keypoints and matches
  cv::Mat imgWithKeypoints;
  cv::drawKeypoints(resliced, keypoints, imgWithKeypoints);

  cv::Mat matchesImg;
  //cv::drawMatches(originalImg, queryKeypoints, img, keypoints, matches, matchesImg, Scalar::all(-1), Scalar::all(-1), vector<vector<char> >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  matchesImg = getResultImage(original, queryKeypoints, resliced, keypoints, houghMatches, 0);
  

  cvStartWindowThread();
  cv::imshow("Matches", matchesImg );
  cv::waitKey(0);
  cvDestroyWindow("Matches");
  cvDestroyWindow("Matches");

  return houghVotes[0];

}

int resliceCastAndCompareToOriginal(vtkImageData* data3d, vnl_matrix<double> cosines, vnl_matrix<double> axisOrigin, const Mat& original, const Mat& maskOriginal)
{
    // Reslice the volume
    vtkSmartPointer<vtkImageReslice> imageReslice = vtkSmartPointer<vtkImageReslice>::New();
    imageReslice->SetInput(data3d);
    imageReslice->SetInformationInput(data3d);
    imageReslice->SetResliceAxesDirectionCosines(cosines.transpose().data_block());
    imageReslice->SetResliceAxesOrigin(axisOrigin.data_block());
    imageReslice->SetOutputDimensionality(2);
    imageReslice->Update();
    vtkImageData* reslicedImage = imageReslice->GetOutput();

    // Compute the image to US transform associated with the resliced image
    //vnl_matrix<double>IJKtoRASResliced = computeReslicedTransform(reslicedImage, cosines, axisOrigin);


    // Convert to unsigned char and compute the mask
    float* reslicedPointer = (float*)reslicedImage->GetScalarPointer();
    unsigned char* reslicedUChar = new unsigned char[reslicedImage->GetNumberOfPoints()];
    unsigned char* reslicedMask = new unsigned char[reslicedImage->GetNumberOfPoints()];

    for(vtkIdType i=0; i<reslicedImage->GetNumberOfPoints(); i++)
    {
      reslicedUChar[i] = (unsigned char)(reslicedPointer[i]*255.);
      if(reslicedPointer[i] == 0)
        reslicedMask[i] = 0;
      else
        reslicedMask[i] = 255;
    }

    // Convert the image and mask to opencv
    int dim[3];
    reslicedImage->GetDimensions(dim);
    Mat resliced(dim[1], dim[0], CV_8U, reslicedUChar);
    Mat maskResliced(dim[1], dim[0], CV_8U, reslicedMask);

    
    
    int result = compareOriginalToResliced(original, resliced, maskOriginal, maskResliced);

    delete [] reslicedUChar;
    delete [] reslicedMask;

    return result;
}

