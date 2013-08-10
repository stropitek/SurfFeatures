#include "vtkSlicerSurfFeaturesLogic.h"
#include "surfLogicUtility.hpp"

// ITK includes
#include <itkImageFileReader.h>
#include "itkImageToVTKImageFilter.txx"

// VTK includes
#include <vtkImageReslice.h>

#include <iostream>
#include <itkNiftiImageIO.h>

using namespace std;

// itk typedefs
typedef unsigned char PixelType;
const unsigned int    Dimension = 3;
typedef itk::Image<PixelType, Dimension> ImageType;
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::ImageToVTKImageFilter<ImageType> imageToVTKImageFilterType;



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
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetImageIO(itk::NiftiImageIO::New());
  reader->SetFileName("C:/Users/DanK/MProject/data/MNI/analyze3dus/1a.3dus.nii");
  reader->Update();
  ImageType::Pointer itkImage = reader->GetOutput();

  imageToVTKImageFilterType::Pointer toVtkFilter = imageToVTKImageFilterType::New();
  toVtkFilter->SetInput(reader->GetOutput());

  toVtkFilter->Update();
  vtkImageData* data = toVtkFilter->GetOutput();
  int* dim = data->GetDimensions();

  vtkSmartPointer<vtkImageReslice> imageReslice = vtkSmartPointer<vtkImageReslice>::New();
  imageReslice->SetInput(data);
  //imageReslice->SetInformation(data->GetInformation());
  imageReslice->SetResliceAxesDirectionCosines(1,0,0,0,1,0,0,0,1);
  imageReslice->SetResliceAxesOrigin(213.,33.,-236.8);
  imageReslice->SetOutputDimensionality(2);
  imageReslice->Update();
  vtkImageData* reslicedImage = imageReslice->GetOutput();
  vtkIdType id = reslicedImage->FindPoint(213.,33.,-230.);
  vtkIdType nbPoints = reslicedImage->GetNumberOfPoints();
  double* center = reslicedImage->GetCenter();
  string s1 = data->GetScalarTypeAsString();
  string s2 = reslicedImage->GetScalarTypeAsString();
  unsigned char* reslicedPointer = (unsigned char*)reslicedImage->GetScalarPointer();
  unsigned char* dataPointer = (unsigned char*)data->GetScalarPointer();
  vtkIdType* increments = data->GetIncrements();
  int* dims = data->GetDimensions();
  data->SetSpacing(-0.3,0.3,0.3);
  for(vtkIdType i=0; i<data->GetNumberOfPoints(); i++)
  {
    double* pos = data->GetPoint(i);
    if(dataPointer[i] != 0){
      cout << "hello";
    }
  }

  // Convert the image to opencv


  // Compute the features on the resliced image


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