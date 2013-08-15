#include "vtkSlicerSurfFeaturesLogic.h"
#include "mhaReader.h"
#include "surfUtil.h"
#include "matlabWriter.h"

#define MATLAB_LOG_RESULTS std::ostringstream oss;\
    oss << "x{" << l+1 << "}";\
    writeVarForMatlab(ofs, oss.str()+".bogusFile", bogusFile);\
    writeVarForMatlab(ofs, oss.str()+".trainFile", trainFile);\
    writeVarForMatlab(ofs, oss.str()+".queryFile", queryFile);\
    writeVarForMatlab(ofs, oss.str()+".ransacMargin", ransacMargin);\
    writeVarForMatlab(ofs, oss.str()+".minHessian", minHessian);\
    writeVarForMatlab(ofs, oss.str()+".repetition", l+1);\
    writeVarForMatlab(ofs, oss.str()+".iInliers", surf->getIInliers());\
    writeVarForMatlab(ofs, oss.str()+".iOutliers", surf->getIOutliers());\
    writeVarForMatlab(ofs, oss.str()+".matchDistance", surf->getMatchDistances());\
    writeVarForMatlab(ofs, oss.str()+".planeDistance", surf->getPlaneDistances());\
    writeVarForMatlab(ofs, oss.str()+".allMatches", surf->getAllMatches());\
    writeVarForMatlab(ofs, oss.str()+".afterBogusMatches", surf->getAfterBogusMatches());\
    writeVarForMatlab(ofs, oss.str()+".afterHoughMatches", surf->getAfterHoughMatches());\
    writeVarForMatlab(ofs, oss.str()+".afterSmoothMatches", surf->getAfterSmoothMatches());




void planeFittingFromFiles(string bogusFile, string trainFile, string queryFile, string outputFile,\
                               int minHessian = 400, float ransacMargin = 1.5, int repetitions = 20,\
                               int bogusStart = 0, int bogusStop = -1,\
                               int trainStart = 0, int trainStop = -1,\
                               int queryStart = 0, int queryStop = -1)
{
  ofstream ofs(outputFile.c_str());
  if(!ofs.is_open()){
    cerr << "Open output file " << outputFile << " failed\n";
    return;
  }

  vtkSlicerSurfFeaturesLogic* surf = vtkSlicerSurfFeaturesLogic::New();
  surf->setBogusFile(bogusFile);
  surf->setTrainFile(trainFile);
  surf->setQueryFile(queryFile);

  surf->setBogusStartFrame(bogusStart);
  surf->setBogusStopFrame(bogusStop);
  surf->setTrainStartFrame(trainStart);
  surf->setTrainStopFrame(trainStop);
  surf->setQueryStartFrame(queryStart);
  surf->setQueryStopFrame(queryStop);

  surf->setMinHessian(minHessian);
  surf->setRansacMargin(ransacMargin);

  // We don't set mask file, crop ratios and calibration matrices, they are set automatically based on resolution of file...

  computeAll(surf);

  for(int l=0; l<repetitions; l++) {
    surf->simulateAll();
    MATLAB_LOG_RESULTS
  }

  surf->Delete();
}

void planeFittingFromSavedKeypoints(string bogusFile, string trainFile, string queryFile, string outputFile,\
                                    float ransacMargin = 1.5, int repetitions = 1)
{
  ofstream ofs(outputFile.c_str());
  if(!ofs.is_open()){
    cerr << "Open output file " << outputFile << " failed\n";
    return;
  }

  vtkSlicerSurfFeaturesLogic* surf = vtkSlicerSurfFeaturesLogic::New();
  
  // Load given the directory where keypoints and descriptors are stored
  surf->loadKeypointsAndDescriptors(bogusFile, "bogus");
  surf->loadKeypointsAndDescriptors(trainFile, "train");
  surf->loadKeypointsAndDescriptors(queryFile, "query");

  // min hessian cannot be changed because keypoints depend on this parameters
  surf->setRansacMargin(ransacMargin);
  int minHessian = surf->getMinHessian();

  // Compute correspondences
  computeCorrespondences(surf);

  for(int l=0; l<repetitions; l++) {
    surf->simulateAll();
    MATLAB_LOG_RESULTS
  }

  surf->Delete();
}



int main (int argc, char const *argv[])
{
  vector<string> outputFiles;
  vector<string> bogusFiles;
  vector<string> trainFiles;
  vector<string> queryFiles;

  #ifdef WIN32
  //bogusFiles.push_back("C:\\Users\\DanK\\MProject\\data\\US\\Plus_test\\TrackedImageSequence_20121210_162606.mha");
  //bogusFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\group1\\01\\pre\\sweep_1a\\sweep_1a.mha");
  bogusFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\save\\sweep_1a");
  //trainFiles.push_back("C:\\Users\\DanK\\MProject\\data\\US\\decemberUS\\TrackedImageSequence_20121211_095535.mha");
  //trainFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\group1\\06\\pre\\sweep_6c\\sweep_6c.mha");
  trainFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\save\\sweep_6c");
  //queryFiles.push_back("C:\\Users\\DanK\\MProject\\data\\US\\decemberUS\\TrackedImageSequence_20121211_095535.mha";)
  //queryFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\group1\\06\\pre\\sweep_6d\\sweep_6d.mha");
  queryFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\save\\sweep_6d");
  outputFiles.push_back("C:\\Users\\DanK\\MProject\\data\\Results\\preop_6c_6d_4.txt");

  #else
  bogusFiles.push_back("/Users/dkostro/Projects/MProject/data/US/Plus_test/TrackedImageSequence_20121210_162606.mha");
  trainFiles.push_back("/Users/dkostro/Projects/MProject/data/US/decemberUS/TrackedImageSequence_20121211_095535.mha");
  queryFiles.push_back("/Users/dkostro/Projects/MProject/data/US/decemberUS/TrackedImageSequence_20121211_095535.mha");
  outputFiles.push_back("/Users/dkostro/Projects/MProject/data/Results/preop_6c_6d.txt");
  #endif
 
  for(int i=0; i<bogusFiles.size(); i++)
  {
    planeFittingFromSavedKeypoints(bogusFiles[i], trainFiles[i], queryFiles[i], outputFiles[i]);
    //planeFittingFromSavedKeypoints(bogusFiles[i], trainFiles[i], queryFiles[i], outputFiles[i]);
  }
  
  return 0;
}