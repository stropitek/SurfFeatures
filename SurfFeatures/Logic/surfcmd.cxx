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

void writeVarForMatlab(std::ofstream& ofs, std::string name, int var)
{
  ofs << name << " %d 1 " << var << std::endl;
}

void writeVarForMatlib(std::ofstream& ofs, std::string name, float var)
{
  ofs << name << " %f 1 " << var << std::endl;
}

void writeVarForMatlib(std::ofstream& ofs, std::string name, double var)
{
  ofs << name << " %f 1 " << var << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::string var)
{
  ofs << name << " %s 1 " << var << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<int> var)
{
  ofs << name << " %d " << var.size();
  for(int i=0; i<var.size(); i++)
  {
    ofs << " " << var[i];
  }
  ofs << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<double> var)
{
  ofs << name << " %f " << var.size();
  for(int i=0; i<var.size(); i++)
  {
    ofs << " " << var[i];
  }
  ofs << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<float> var)
{
  ofs << name << " %f " << var.size();
  for(int i=0; i<var.size(); i++)
  {
    ofs << " " << var[i];
  }
  ofs << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<std::string> var)
{
  ofs << name << " %s " << var.size();
  for(int i=0; i<var.size(); i++)
  {
    ofs << " " << var[i];
  }
  ofs << std::endl;
}

int main (int argc, char const *argv[])
{
  vtkSlicerSurfFeaturesLogic* surf = vtkSlicerSurfFeaturesLogic::New();


  // Parameters init
  std::vector<std::string> bogusFiles;
  std::vector<std::string> trainFiles;
  std::vector<std::string> queryFiles;
  #ifdef WIN32
  bogusFiles.push_back("C:\\Users\\DanK\\MProject\\data\\US\\Plus_test\\TrackedImageSequence_20121210_162606.mha");
  trainFiles.push_back("C:\\Users\\DanK\\MProject\\data\\US\\decemberUS\\TrackedImageSequence_20121211_095535.mha");
  queryFiles.push_back("C:\\Users\\DanK\\MProject\\data\\US\\decemberUS\\TrackedImageSequence_20121211_095535.mha");
  #else
  bogusFiles.push_back("/Users/dkostro/Projects/MProject/data/US/Plus_test/TrackedImageSequence_20121210_162606.mha");
  trainFiles.push_back("/Users/dkostro/Projects/MProject/data/US/decemberUS/TrackedImageSequence_20121211_095535.mha");
  queryFiles.push_back("/Users/dkostro/Projects/MProject/data/US/decemberUS/TrackedImageSequence_20121211_095535.mha");
  #endif

  std::vector<int> bogusStartFrames;
  std::vector<int> bogusStopFrames;
  std::vector<int> trainStartFrames;
  std::vector<int> trainStopFrames;
  std::vector<int> queryStartFrames;
  std::vector<int> queryStopFrames;

  bogusStartFrames.push_back(0);
  bogusStopFrames.push_back(200);
  trainStartFrames.push_back(0);
  trainStopFrames.push_back(902);
  queryStartFrames.push_back(903);
  queryStopFrames.push_back(-1);


  std::vector<int> minHessians;
  std::vector<float> ransacMargins;
  minHessians.push_back(300);
  minHessians.push_back(200);
  minHessians.push_back(400);
  minHessians.push_back(500);
  minHessians.push_back(600);
  
  ransacMargins.push_back(1.0);
  ransacMargins.push_back(1.5);
  ransacMargins.push_back(2.0);
  ransacMargins.push_back(2.5);
  ransacMargins.push_back(3.0);

  int repetitions = 20;

  int iFiles = bogusFiles.size();
  if(trainFiles.size()!=iFiles || queryFiles.size()!=iFiles || bogusStartFrames.size()!=iFiles || trainStartFrames.size()!=iFiles || queryStartFrames.size()!=iFiles || bogusStopFrames.size()!=iFiles || trainStopFrames.size()!=iFiles || queryStopFrames.size()!=iFiles) {
    std::cerr << "Error: invalid parameters" << std::endl;
    return 1;
  }

  int count = 0;
  #ifdef WIN32
  std::ofstream ofs("C:\\Users\\DanK\\MProject\\data\\Results\\surfResults.txt");
  #else
  std::ofstream ofs("/Users/dkostro/Projects/MProject/data/US/Results/surfResults.txt");
  #endif
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
        for(int l=0; l<repetitions; l++) {
          count++;
          surf->setRansacMargin(ransacMargins[k]);
          surf->simulateAll();
          std::ostringstream oss;
          oss << "x{" << count << "}";
          writeVarForMatlab(ofs, oss.str()+".bogusFile", bogusFiles[i]);
          writeVarForMatlab(ofs, oss.str()+".trainFile", trainFiles[i]);
          writeVarForMatlab(ofs, oss.str()+".queryFile", queryFiles[i]);
          writeVarForMatlab(ofs, oss.str()+".ransacMargin", ransacMargins[k]);
          writeVarForMatlab(ofs, oss.str()+".minHessian", minHessians[j]);
          writeVarForMatlab(ofs, oss.str()+".repetition", l);
          writeVarForMatlab(ofs, oss.str()+".iInliers", surf->getIInliers());
          writeVarForMatlab(ofs, oss.str()+".iOutliers", surf->getIOutliers());
          writeVarForMatlab(ofs, oss.str()+".matchDistance", surf->getMatchDistances());
          writeVarForMatlab(ofs, oss.str()+".planeDistance", surf->getPlaneDistances());
        }
      }
    }
  }
  surf->Delete();
  return 0;
}