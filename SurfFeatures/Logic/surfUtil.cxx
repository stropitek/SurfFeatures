#include "surfUtil.h"
#include "time.h"

// ======================================================
// Some function that reuse the surffeature slicer logic
// ======================================================
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

void computeCorrespondences(vtkSlicerSurfFeaturesLogic* surf)
{
  // Compute correspondences
  surf->startInterSliceCorrespondence();
  while(true)
  {
    if(!surf->isCorrespondenceComputing())
      break;
    surf->computeNextInterSliceCorrespondence_blabla();
  } 
}

void computeAll(vtkSlicerSurfFeaturesLogic* surf)
{
  computeQuery(surf);
  computeTrain(surf);
  computeBogus(surf);

  computeCorrespondences(surf); 
}

// Reads lines from a file into a vector
void readLines( const string& filename, vector<string>& lines )
{
    lines.clear();

    ifstream file( filename.c_str() );
    if ( !file.is_open() )
        return;

    while( !file.eof() )
    {
        string str; getline( file, str );
        if( str.empty() ) break;
        lines.push_back(str);
    }
    file.close();
} 