#ifndef SURF_UTIL_H
#define SURF_UTIL_H

#include "vtkSlicerSurfFeaturesLogic.h"

// ======================================================
// Some function that reuse the surffeature slicer logic
// ======================================================
void computeQuery(vtkSlicerSurfFeaturesLogic* surf);
void computeBogus(vtkSlicerSurfFeaturesLogic* surf);
void computeTrain(vtkSlicerSurfFeaturesLogic* surf);
void computeCorrespondences(vtkSlicerSurfFeaturesLogic* surf);
void computeAll(vtkSlicerSurfFeaturesLogic* surf);

#endif