#include "vtkSlicerSurfFeaturesLogic.h"
#include "surfLogicUtility.hpp"

#include <iostream>
#include <windows.h>

using namespace std;

int main()
{
  vector<string> files;
  vector<string> outputDirs;
  readLines("C:\\Users\\DanK\\MProject\\data\\MNI\\mni_mha.txt",files);
  readLines("C:\\Users\\DanK\\MProject\\data\\MNI\\save_dirs.txt",outputDirs);
  if(file.size() != outputDirs.size())
    return 0;
  for(int i=0; i<files.size(); i++)
  {
    vtkSlicerSurfFeaturesLogic* surf = vtkSlicerSurfFeaturesLogic::New();
    surf->setQueryFile(files[i]);
    surf->setMinHessian(400);
    surf->setQueryStartFrame(0);
    surf->setQueryStopFrame(-1);
    surf->setRansacMargin(1.5);
    vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
    matrix->Identity();
    surf->setImageToProbeTransform(matrix);
    // Set cropping correctly...
    float cropRatios[4] = {0,1,0,1};
    surf->setCropRatios(cropRatios);
    surf->setMaskFile("C:\\Users\\DanK\\MProject\\data\\MNI\\mni_mask.png");
    computeQuery(surf);
    std::string dir, file, ext;
    splitDir(files[i],dir,file,ext);
    surf->writeKeypointsAndDescriptors(outputDirs[i]);
    //surf->loadKeypointsAndDescriptors(outputDir, "query");


    surf->Delete();
  }


  /*cv::FileStorage fs("C:\\Users\\DanK\\test.yml", cv::FileStorage::READ);
  vector<cv::KeyPoint> readKey;
  cv::Mat readDescriptor;
  if(fs.isOpened())
  {
    read(fs["keyPoints"],readKey);
    read(fs["descriptors"],readDescriptor);
  }
  fs.release();
  cout << readKey.size() << endl;
  cout << readDescriptor.rows << " " << readDescriptor.cols << endl;*/
}