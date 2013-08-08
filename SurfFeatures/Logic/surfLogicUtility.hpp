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

void computeCorrespondences(vtkSlicerSurfFeaturesLogic* surf)
{
  // Compute correspondences
  surf->startInterSliceCorrespondence();
  while(true)
  {
    if(!surf->isCorrespondenceComputing())
      break;
    surf->computeNextInterSliceCorrespondence();
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

void readTab(const string& filename, vector<vector<string> >& tab)
{

}

// Normalize dir by adding the appropriate delimiter at the end, if not present
void normalizeDir(std::string& dir)
{
  #ifdef WIN32
  const char dlmtr = '\\';
  #else
  const char dlmtr = '/';
  #endif

  if(!dir.empty()){
    if(*(dir.end()-1) != dlmtr)
      dir = dir + dlmtr;
  }

}

// Get the directory given a filename.
void splitDir(const std::string& filename, std::string& dir, std::string& file, string& ext)
{
  #ifdef WIN32
  const char dlmtr = '\\';
  #else
  const char dlmtr = '/';
  #endif

  size_t pos = filename.rfind(dlmtr);
  dir = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;
  file = pos == string::npos ? "" : filename.substr(pos+1);

  if(file.empty()){
    ext=""; return;
  }

  pos = file.find_last_of(".");
  ext = pos == string::npos ? "" : file.substr(pos+1);
  file = pos == string::npos ? file : file.substr(0,pos);

}

#endif
