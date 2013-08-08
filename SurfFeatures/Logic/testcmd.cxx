#include "vtkSlicerSurfFeaturesLogic.h"
#include "surfLogicUtility.hpp"

// ITK includes
#include <itkImageFileReader.h>
#include "itkImageToVTKImageFilter.txx"

// VTK includes
#include <vtkImageReslice.h>

#include <iostream>
#include <windows.h>

using namespace std;

// itk typedefs
typedef unsigned char PixelType;
const unsigned int    Dimension = 3;
typedef itk::Image<PixelType, Dimension> ImageType;
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::ImageToVTKImageFilter<ImageType> imageToVTKImageFilterType;



void writeKeypointFromMhaFile(string mhaFile, string outputDir, int minHessian)
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
  reader->SetFileName("C:\\Users\\DanK\\MProject\\data\\MNI\\analyze3dus\\1a.3dus");

  imageToVTKImageFilterType::Pointer toVtkFilter = imageToVTKImageFilterType::New();
  toVtkFilter->SetInput(reader->GetOutput());

  toVtkFilter->Update();
  vtkImageData* data = toVtkFilter->GetOutput();

  vtkSmartPointer<vtkImageReslice> imageReslice = vtkSmartPointer<vtkImageReslice>::New();
  imageReslice->SetInput(data);
  imageReslice->SetResliceAxesDirectionCosines(1,0,0,0,1,0,0,0,1);
  imageReslice->SetResliceAxesOrigin(0,0,0);
  vtkImageData* reslicedImage = imageReslice->GetOutput();

  // Convert the image to opencv


  // Compute the features on the resliced image

  vector<string> files;
  vector<string> outputDirs;
  readLines("C:\\Users\\DanK\\MProject\\data\\MNI_AMIGO_mha.txt",files);
  readLines("C:\\Users\\DanK\\MProject\\data\\MNI_AMIGO_dirs.txt",outputDirs);
  int minHessian = 400;

  if(files.size() != outputDirs.size())
    return 0;

  for(int i=0; i<files.size(); i++)
    writeKeypointsFromMhaFile(files[i], outputDirs[i], minHessian);
}