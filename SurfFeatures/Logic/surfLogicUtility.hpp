#ifndef SURF_LOGIC_UTILITY_H
#define SURF_LOGIC_UTILITY_H

void computeQuery(vtkSlicerSurfFeaturesLogic* surf)
{
  surf->computeQuery();
  // Compute query
  while(true)
  {
    if(!surf->isQueryLoading())
      break;
    surf->computeNextQuery();
  }
}

void computeBogus(vtkSlicerSurfFeaturesLogic* surf)
{
  surf->computeBogus();
  // Compute bogus
  while(true)
  {
    if(!surf->isBogusLoading())
      break;
    surf->computeNextBogus();
  }
}

void computeTrain(vtkSlicerSurfFeaturesLogic* surf)
{
  surf->computeTrain();
  // Compute train
  while(true)
  {
    if(!surf->isTrainLoading())
      break;
    surf->computeNextTrain();
  }
}

void computeAll(vtkSlicerSurfFeaturesLogic* surf)
{
  computeQuery(surf);
  computeTrain(surf);
  computeBogus(surf);

  // Compute correspondences
  surf->startInterSliceCorrespondence();
  while(true)
  {
    if(!surf->isCorrespondenceComputing())
      break;
    surf->computeNextInterSliceCorrespondence();
  }  
}

#endif
