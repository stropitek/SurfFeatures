#include "vtkSlicerSurfFeaturesLogic.h"
#include "surfLogicUtility.hpp"

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

int main()
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
  vnl_matrix<double> normalizedTransformMatrix = transformMatrix;
  normalizedTransformMatrix.normalize_columns();

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


  
  imageToVTKImageFilterType::Pointer toVtkFilter = imageToVTKImageFilterType::New();
  toVtkFilter->SetInput(reader->GetOutput());
  toVtkFilter->Update();
  vtkSmartPointer<vtkImageData> data = vtkSmartPointer<vtkImageData>::New();
  data->DeepCopy(toVtkFilter->GetOutput());
  int* dim;
  dim = data->GetDimensions();
  data->SetSpacing(IJKtoRASMatrix.get_diagonal().extract(3).data_block());
  data->SetOrigin(IJKtoRASMatrix.get_column(3).extract(3).data_block());
  vtkSmartPointer<vtkImageReslice> imageReslice = vtkSmartPointer<vtkImageReslice>::New();
  imageReslice->SetInput(data);
  imageReslice->SetInformationInput(data);
  vnl_matrix<double> cosines = normalizedTransformMatrix.extract(3,3);
  vnl_matrix<double> frameOrigin = transformMatrix.extract(3,1,0,3);
  vnl_matrix<double> frameCenter = frameOrigin + transformMatrix.extract(3,1)*images[0].cols/2 + transformMatrix.extract(3,1,0,1)*images[0].rows/2;
  imageReslice->SetResliceAxesDirectionCosines(cosines.transpose().data_block());
  imageReslice->SetResliceAxesOrigin(frameCenter.data_block());
  imageReslice->SetOutputDimensionality(2);
  imageReslice->Update();
  vtkImageData* reslicedImage = imageReslice->GetOutput();
  vtkIdType id = reslicedImage->FindPoint(0.,0.,0.);
  double x[3] = {0.,0.,0.};
  double pcoords[3];
  int ijk[3];
  reslicedImage->ComputeStructuredCoordinates(x,ijk,pcoords);
  vnl_matrix<double> newOrigin(3,1,0.);
  newOrigin -= cosines.extract(3,1)*ijk[0]*reslicedImage->GetSpacing()[0];
  newOrigin -= cosines.extract(3,1,0,1)*ijk[1]*reslicedImage->GetSpacing()[1];
  newOrigin += frameCenter;

  vnl_matrix<double> IJKtoRASResliced = normalizedTransformMatrix;
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
  IJKtoRASResliced.print(cout);

  vtkSmartPointer<vtkMatrix4x4> reslicedTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkMatrix4x4> originalTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  convertVnlMatrixToVtkMatrix4x4(IJKtoRASResliced, reslicedTransform);
  convertVnlMatrixToVtkMatrix4x4(transformMatrix, originalTransform);


  cout << "Transform matrix: \n";
  transformMatrix.print(cout);
  cout << endl;
  cout << "Normalized Transform Matrix: \n";
  normalizedTransformMatrix.print(cout);
  cout << endl;
  cout << "Frame origin: \n";
  frameOrigin.print(cout);
  cout << endl;
  cout << "Frame center: \n";
  frameCenter.print(cout);
  cout << endl;
  cout << "Cosines: \n";
  cosines.print(cout);
  cout << endl;
  cout << "New origin: \n";
  newOrigin.print(cout);
  cout << endl;
  cout << "IJKtoRASResliced transform: \n";
  IJKtoRASResliced.print(cout);
  cout << endl;
  
   vtkIdType nbPoints = reslicedImage->GetNumberOfPoints();
  

  float* reslicedPointer = (float*)reslicedImage->GetScalarPointer();
  float* dataPointer = (float*)data->GetScalarPointer();


  vtkIdType* increments = data->GetIncrements();
  int* dims = data->GetDimensions();
  string s1 = data->GetScalarTypeAsString();

  //double* center = reslicedImage->GetCenter();
  string s2 = reslicedImage->GetScalarTypeAsString();

  unsigned char* reslicedUChar = new unsigned char[reslicedImage->GetNumberOfPoints()];
  unsigned char* reslicedMask = new unsigned char[reslicedImage->GetNumberOfPoints()];

  for(vtkIdType i=0; i<reslicedImage->GetNumberOfPoints(); i++)
  {
    reslicedUChar[i] = (unsigned char)(reslicedPointer[i]*255.);
    if(reslicedPointer[i] == 0)
      reslicedMask[i] = 0;
    else
      reslicedMask[i] = 255;
    //double* pos = reslicedImage->GetPoint(i);
    //if(reslicedPointer != 0){
    //  cout << "hello";
    //}
  }

  // Convert the image to opencv
  dim = reslicedImage->GetDimensions();
  Mat img(dim[1], dim[0], CV_8U, reslicedUChar);
  Mat mask(dim[1], dim[0], CV_8U, reslicedMask);

  // Compute the features on the resliced image

  // Get the descriptor of the current image
  cv::SurfFeatureDetector detector(400);
  // Find Keypoints
  vector<cv::KeyPoint> keypoints;
  detector.detect(img,keypoints,mask);

  cv::Mat imgWithKeypoints;
  cv::drawKeypoints(img, keypoints, imgWithKeypoints);





  cvStartWindowThread();
  cv::imshow("Image", img );
  

  cvStartWindowThread();
  cv::imshow("Original", images[0] );
  cv::waitKey(0);
  cvDestroyWindow("Original");
  cvDestroyWindow("Image");



  delete [] reslicedUChar;
  delete [] reslicedMask;



  // ==============================================
  // write correspondences for a set of images
  // ==============================================
  //vector<string> files;
  //vector<string> outputDirs;
  //readLines("C:\\Users\\DanK\\MProject\\data\\MNI_AMIGO_mha.txt",files);
  //readLines("C:\\Users\\DanK\\MProject\\data\\MNI_AMIGO_dirs.txt",outputDirs);
  //int minHessian = 400;

  //if(files.size() != outputDirs.size())
  //  return 0;

  //for(int i=0; i<files.size(); i++)
  //  writeKeypointsFromMhaFile(files[i], outputDirs[i], minHessian);
}