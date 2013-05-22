/*==============================================================================

  Program: 3D Slicer

  Portions (c) Copyright Brigham and Women's Hospital (BWH) All Rights Reserved.

  See COPYRIGHT.txt
  or http://www.slicer.org/copyright/copyright.txt for details.

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

==============================================================================*/

// .NAME vtkSlicerSurfFeaturesLogic - slicer logic class for volumes manipulation
// .SECTION Description
// This class manages the logic associated with reading, saving,
// and changing propertied of the volumes


#ifndef __vtkSlicerSurfFeaturesLogic_h
#define __vtkSlicerSurfFeaturesLogic_h

// Slicer includes
#include "vtkSlicerModuleLogic.h"

// MRML includes
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkMRMLLinearTransformNode.h"

// STD includes
#include <cstdlib>
#include <fstream>
#include <ctime>

// OpenCV includes
#include "opencv2/core/core.hpp"
#include <opencv2/nonfree/features2d.hpp>

// Qt includes
#include <QTextEdit>
#include "vtkSlicerSurfFeaturesModuleLogicExport.h"

#include "util_macros.h"

using namespace std;
using namespace cv;


/// \ingroup Slicer_QtModules_ExtensionTemplate
class VTK_SLICER_SURFFEATURES_MODULE_LOGIC_EXPORT vtkSlicerSurfFeaturesLogic :
  public vtkSlicerModuleLogic
{
public:

  static vtkSlicerSurfFeaturesLogic *New();
  vtkTypeMacro(vtkSlicerSurfFeaturesLogic, vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);

  void next();
  void setConsole(QTextEdit* console);
  void setMinHessian(int);
  void setBogusFile(std::string);
  void setTrainFile(std::string);
  void setQueryFile(std::string);
  

  // parameter setters
  void setTrainStartFrame(int startFrame);
  void setBogusStartFrame(int startFrame);
  void setTrainStopFrame(int stopFrame);
  void setBogusStopFrame(int stopFrame);
  SET(int,queryStartFrame,QueryStartFrame);
  SET(int,queryStopFrame,QueryStopFrame);

  void computeBogus();
  void computeTrain();
  void computeQuery();

protected:
  vtkSlicerSurfFeaturesLogic();
  virtual ~vtkSlicerSurfFeaturesLogic();

  virtual void SetMRMLSceneInternal(vtkMRMLScene* newScene);
  /// Register MRML Node classes to Scene. Gets called automatically when the MRMLScene is attached to this logic class.
  virtual void RegisterNodes();
  virtual void UpdateFromMRMLScene();
  virtual void OnMRMLSceneNodeAdded(vtkMRMLNode* node);
  virtual void OnMRMLSceneNodeRemoved(vtkMRMLNode* node);
private:

  vtkSlicerSurfFeaturesLogic(const vtkSlicerSurfFeaturesLogic&); // Not implemented
  void operator=(const vtkSlicerSurfFeaturesLogic&);               // Not implemented

  void stopWatchWrite(std::ostringstream& oss);
  void resetConsoleFont();
  vtkImageData* cropData(vtkImageData* data);
  void cropData(cv::Mat& img);
  cv::Mat convertImage(vtkImageData* data);                                                  
  void showImage(vtkMRMLNode* node);
  
  bool isTracked(vtkMRMLNode* node);
  void initBogusDatabase();
  void findClosestSlice(vtkMRMLNode* queryNode);
  void updateDescriptorMatcher();
  void computeKeypointsAndDescriptors(const cv::Mat& data,\
                                      std::vector<cv::KeyPoint>& keypoints,\
                                      cv::Mat& descriptors);
  void readAndComputeFeaturesOnMhaFile(const std::string& file,\
                                       std::vector<cv::Mat>& images,\
                                       std::vector<std::string>& imagesNames,\
                                       std::vector<std::vector<float> >& imagesTransform,\
                                       std::vector<std::vector<cv::KeyPoint> >& keypoints,\
                                       std::vector<cv::Mat>& descriptors,\
                                       cv::Ptr<cv::DescriptorMatcher> descriptorMatcher,\
                                       int startFrame, int stopFrame);

  // Attributes
private:
  // Node
  vtkMRMLScalarVolumeNode* node;

  // Clock
  clock_t initTime;
  clock_t lastStopWatch;

  // State
  unsigned int step;

  // Action Flags


  // train images, keypoints, descriptors and matcher
  std::vector<cv::Mat> trainImages;
  std::vector<std::string> trainImagesNames;
  std::vector< std::vector<float> > trainImagesTransform;
  std::vector<std::vector<cv::KeyPoint> > trainKeypoints;
  std::vector<cv::Mat> trainDescriptors;
  cv::Ptr<cv::DescriptorMatcher> trainDescriptorMatcher;
  

  // bogus images, keypoints, descriptors and matcher
  std::vector<cv::Mat> bogusImages;
  std::vector<std::string> bogusImagesNames;
  std::vector< std::vector<float> > bogusImagesTransform;
  std::vector<std::vector<cv::KeyPoint> > bogusKeypoints;
  std::vector<cv::Mat> bogusDescriptors;
  cv::Ptr<cv::DescriptorMatcher> bogusDescriptorMatcher;

  // query images, keypoints, descriptors and matcher
  std::vector<cv::Mat> queryImages;
  std::vector<std::string> queryImagesNames;
  std::vector< std::vector<float> > queryImagesTransform;
  std::vector<std::vector<cv::KeyPoint> > queryKeypoints;
  std::vector<cv::Mat> queryDescriptors;
  cv::Ptr<cv::DescriptorMatcher> queryDescriptorMatcher;
  

  // Console
  QTextEdit* console;

  // Parameters
  int minHessian;
  std::string matcherType;
  std::string trainFile;
  std::string bogusFile;
  std::string queryFile;
  int trainStartFrame;
  int trainStopFrame;
  int bogusStartFrame;
  int bogusStopFrame;
  int queryStartFrame;
  int queryStopFrame;
  float cropRatios[4];
};



#endif
