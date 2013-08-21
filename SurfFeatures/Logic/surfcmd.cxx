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
    //MATLAB_LOG_RESULTS
  }

  surf->Delete();
}
 

void planeFittingFromSavedKeypoints(string bogusFile, string trainFile, string queryFile, string outputFile,
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


void planeFittingFromSavedKeypoints(string bogusFile, vector<string> trainFile, string queryFile, string outputFile,
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
  surf->loadSeveralTrainKeypointsAndDescriptors(trainFile);
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

void findTrainPlaneClosestToQueryMni(const vector<string>& trainFiles, const vector<string>& queryFiles, const vector<string>& outputFiles)
{
  for(int i=0; i<trainFiles.size(); i++)
  {
    vector<vector<float> > trainTransforms;
    vector<bool> trainTransformsValidity;
    vector<string> trainFilenames;
    vector<vector<float> > queryTransforms;
    vector<bool> queryTransformsValidity;
    vector<string> queryFilenames;
    readImageTransforms_mha(trainFiles[i],trainTransforms, trainTransformsValidity, trainFilenames);
    readImageTransforms_mha(queryFiles[i],queryTransforms, queryTransformsValidity, queryFilenames);    

    vector<vector<vnl_double_3> > queryPlanePoints(queryTransforms.size(), vector<vnl_double_3>());
    vector<vector<vnl_double_3> > trainPlanePoints(trainTransforms.size(), vector<vnl_double_3>());

    // For MNI, no need to recalibrate, no need to crop
    cv::Mat mni_mask = cv::imread("C:\\Users\\DanK\\MProject\\data\\MNI\\mni_mask.png", CV_LOAD_IMAGE_GRAYSCALE);
    int step = 4;
    for(int r=0; r<mni_mask.rows; r+=step) {
      for(int c=0; c<mni_mask.cols; c+=step) {
        if(mni_mask.at<unsigned char>(r,c) == 0)
          continue;
        for(int k=0; k<queryTransforms.size(); k++)
        {
          vtkSmartPointer<vtkMatrix4x4> qtransform = vtkSmartPointer<vtkMatrix4x4>::New();
          convertMhaTransformToVtkMatrix(queryTransforms[k], qtransform);
          vnl_double_3 point(r,c,0.0);
          queryPlanePoints[k].push_back(transformPoint(point,qtransform));
        }
        for(int k=0; k<trainTransforms.size(); k++)
        {
          vtkSmartPointer<vtkMatrix4x4> ttransform = vtkSmartPointer<vtkMatrix4x4>::New();
          convertMhaTransformToVtkMatrix(trainTransforms[k], ttransform);
          vnl_double_3 point(r,c,0.0);
          trainPlanePoints[k].push_back(transformPoint(point,ttransform));
        }
      }
    }
    vector<vector<double> > meanPlaneDistances;
    for(int j=0; j<queryPlanePoints.size(); j++)
    {
      vector<double> dists;
      for(int k=0; k<trainPlanePoints.size(); k++)
      {
        dists.push_back(computeMeanDistance(queryPlanePoints[j], trainPlanePoints[k]));
      }
      meanPlaneDistances.push_back(dists);
    }
    ofstream ofs(outputFiles[i].c_str());
    for(int j=0; j<meanPlaneDistances.size(); j++)
    {
      ostringstream oss;
      oss << "dist{" << j+1 << "}";\
      writeVarForMatlab(ofs, oss.str(), meanPlaneDistances[j]);
    }
    cout << "Written " << i+1 << "/" << trainFiles.size() << endl;
  }
}


void mainClosestPlane(string);
void mainFittingOneTrain(string);
void mainFittingSeveralTrain(string);
void mainFittingPerformance();
int main (int argc, char const *argv[])
{
  //mainClosestPlane("c:\\users\\dank\\mproject\\data\\results\\scenarios\\planedist_scenarios.txt");
  //mainFittingOneTrain("C:\\Users\\DanK\\MProject\\data\\Results\\scenarios\\postpre_scenarios.txt");
  //mainFittingSeveralTrain("C:\\Users\\DanK\\MProject\\data\\Results\\scenarios\\postpre1_scenarios.txt");
  mainFittingPerformance();
  return 0;
}

void mainFittingPerformance()
{
  string scenario = "c:\\users\\dank\\mproject\\data\\results\\scenarios\\test_scenarios.txt";
  
  vector<string> outputFiles;
  vector<string> bogusFiles;
  vector<string> queryFiles;
  vector<string> trainFiles;
  vector<vector<string> > tab;

  readTab(scenario, tab);
  for(int i=0; i<tab.size(); i++)
  {
    bogusFiles.push_back("Z:\\neuro_ultrasound\\20130730_crani\\us\\TrackedImageSequence_20130730_101437.mha");
    trainFiles.push_back(tab[i][0]);
    queryFiles.push_back(tab[i][1]);
    outputFiles.push_back(tab[i][2]);
  }
  for(int i=0; i<bogusFiles.size(); i++)
  {
    planeFittingFromFiles(bogusFiles[i], trainFiles[i], queryFiles[i], outputFiles[i]);
  }
}


void mainClosestPlane(string filename)
{
  vector<string> outputFiles;
  vector<string> bogusFiles;
  vector<string> queryFiles;
  vector<string> trainFiles;
  vector<vector<string> > tab;

  readTab(filename, tab);
  for(int i=0; i<tab.size(); i++)
  {
    trainFiles.push_back(tab[i][0]);
    queryFiles.push_back(tab[i][1]);
    outputFiles.push_back(tab[i][2]);
  }
  findTrainPlaneClosestToQueryMni(trainFiles, queryFiles, outputFiles);
}

void mainFittingOneTrain(string filename)
{

  vector<string> outputFiles;
  vector<string> bogusFiles;
  vector<string> queryFiles;
  vector<string> trainFiles;
  vector<vector<string> > tab;
  
  readTab(filename, tab);
  for(int i=0; i<tab.size(); i++)
  {
    bogusFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\save\\sweep_1a");
    trainFiles.push_back(tab[i][0]);
    queryFiles.push_back(tab[i][1]);
    outputFiles.push_back(tab[i][2]);
  }

  for(int i=0; i<bogusFiles.size(); i++)
    planeFittingFromSavedKeypoints(bogusFiles[i], trainFiles[i], queryFiles[i], outputFiles[i]);

}

void mainFittingSeveralTrain(string scenario)
{
  vector<string> outputFiles;
  vector<string> bogusFiles;
  vector<string> queryFiles;
  vector<vector<string> > trainFiles;
  vector<vector<string> > tab;
  readTab(scenario, tab);
  
  for(int i=0; i<tab.size(); i++)
  {
    trainFiles.push_back(vector<string>(tab[i].begin(), tab[i].end()-2));
    bogusFiles.push_back("C:\\Users\\DanK\\MProject\\data\\MNI\\save\\sweep_1a");
    queryFiles.push_back(*(tab[i].end()-2));
    outputFiles.push_back(*(tab[i].end()-1));
  }

  for(int i=0; i<bogusFiles.size(); i++)
    planeFittingFromSavedKeypoints(bogusFiles[i], trainFiles[i], queryFiles[i], outputFiles[i]);

}