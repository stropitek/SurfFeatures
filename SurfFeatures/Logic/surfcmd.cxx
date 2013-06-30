#include "vtkSlicerSurfFeaturesLogic.h"

void computeAll(vtkSlicerSurfFeaturesLogic* surf)
{
  surf->computeBogus();
  surf->computeTrain();
  surf->computeQuery();
  
  // Compute bogus
  while(true)
  {
    if(!surf->isBogusLoading())
      break;
    surf->computeNextBogus();
  }
  
  // Compute Train
  while(true)
  {
    if(!surf->isTrainLoading())
      break;
    surf->computeNextTrain();
  }
  
  // Compute query
  while(true)
  {
    if(!surf->isQueryLoading())
      break;
    surf->computeNextQuery();
  }
  
  // Compute correspondences
  surf->startInterSliceCorrespondence();
  while(true)
  {
    if(!surf->isCorrespondenceComputing())
      break;
    surf->computeNextInterSliceCorrespondence();
  }
}


int main (int argc, char const *argv[])
{
  vtkSlicerSurfFeaturesLogic* surf = vtkSlicerSurfFeaturesLogic::New();
  
  // Paremeters
  std::vector<std::string> bogusFiles;
  std::vector<std::string> trainFiles;
  std::vector<std::string> queryFiles;
  
  std::vector<int> bogusStartFrames;
  std::vector<int> bogusStopFrames;
  std::vector<int> trainStartFrames;
  std::vector<int> trainStopFrames;
  std::vector<int> queryStartFrames;
  std::vector<int> queryStopFrames;
  
  std::vector<int> minHessians;
  std::vector<float> ransacMargins;
  
  int repetitions;
  
  int iFiles = bogusFiles.size();
  if(trainFiles.size()!=iFiles || queryFiles.size()!=iFiles || bogusStartFrames.size()!=iFiles || trainStartFrames.size()!=iFiles || queryStartFrames.size()!=iFiles || bogusStopFrames.size()!=iFiles || trainStopFrames.size()!=iFiles || queryStopFrames.size()!=iFiles) {
    std::cerr << "Error: invalid parameters" << std::endl;
    return 1;
  }
  
  for(int i=0; i<iFiles; i++)
  {
    surf->setBogusFile(bogusFiles[i]);
    surf->setTrainFile(trainFiles[i]);
    surf->setTrainFile(queryFiles[i]);
    
    surf->setBogusStartFrame(bogusStartFrames[i]);
    surf->setTrainStartFrame(trainStartFrames[i]);
    surf->setQueryStartFrame(queryStartFrames[i]);
    surf->setBogusStopFrame(bogusStopFrames[i]);
    surf->setTrainStopFrame(trainStopFrames[i]);
    surf->setQueryStopFrame(queryStopFrames[i]);

    for(int j=0; j<minHessians.size(); j++)
    {
      surf->setMinHessian(minHessians[j]);
      computeAll(surf);
      
      for(int k=0; k<ransacMargins.size(); k++)
      {
        surf->setRansacMargin(ransacMargins[k]);
      }
    }
  }

  surf->Delete();
  return 0;
}