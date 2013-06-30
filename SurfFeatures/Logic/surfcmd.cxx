#include "vtkSlicerSurfFeaturesLogic.h"

int main (int argc, char const *argv[])
{
  /* code */
  vtkSlicerSurfFeaturesLogic* surf = vtkSlicerSurfFeaturesLogic::New();
  
  // surf->setMinHessian(400);
  //   surf->setBogusFile();
  //   surf->setTrainFile();
  //   surf->setQueryFile();
  //   surf->setQueryStartFrame();
  //   surf->setQueryStopFrame();
  //   surf->setTrainS
  
  // Load data
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
  
  
  surf->Delete();
  return 0;
}