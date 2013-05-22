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

  void displayFeatures(vtkMRMLNode*);
  void setObservedNode(vtkMRMLScalarVolumeNode* snode);
  void setObservedNode(vtkMRMLLinearTransformNode* tnode);
  void removeObservedVolume();
  void removeObservedTransform();
  void toggleRecord();
  void setConsole(QTextEdit* console);
  void setMinHessian(int);
  void showNextImage();
  void matchWithNextImage();
  void changeBogusFile(std::string);

protected:
  vtkSlicerSurfFeaturesLogic();
  virtual ~vtkSlicerSurfFeaturesLogic();

  virtual void SetMRMLSceneInternal(vtkMRMLScene* newScene);
  /// Register MRML Node classes to Scene. Gets called automatically when the MRMLScene is attached to this logic class.
  virtual void RegisterNodes();
  virtual void UpdateFromMRMLScene();
  virtual void OnMRMLSceneNodeAdded(vtkMRMLNode* node);
  virtual void OnMRMLSceneNodeRemoved(vtkMRMLNode* node);

  virtual void ProcessMRMLNodesEvents(vtkObject* caller, unsigned long event, void * callData);
private:

  vtkSlicerSurfFeaturesLogic(const vtkSlicerSurfFeaturesLogic&); // Not implemented
  void operator=(const vtkSlicerSurfFeaturesLogic&);               // Not implemented

  void recordData(vtkMRMLNode* node);
  void stopWatchWrite(std::ostringstream& oss);
  void resetConsoleFont();
  vtkImageData* cropData(vtkImageData* data);
  cv::Mat convertImage(vtkImageData* data);                                                  
  void showImage(vtkMRMLNode* node);
  void showMatchWithImage(vtkMRMLNode* node);
  void computeKeypointsAndDescriptors(const cv::Mat& data, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
  bool isTracked(vtkMRMLNode* node);
  void initBogusDatabase();
  void findClosestSlice(vtkMRMLNode* queryNode);
  void updateDescriptorMatcher();

  // Attributes
private:
  // Nodes
  vtkMRMLScalarVolumeNode* observedVolume;
  vtkMRMLLinearTransformNode* observedTransform;

  // Clock
  clock_t lastImageModified;
  clock_t initSurf;
  clock_t initTime;
  clock_t lastStopWatch;

  // Action Flags
  bool recording;
  bool matchNext;
  bool showNextImg;
  bool matchWithNextImg;

  // opencv keypoints, descriptors and matchers
  std::vector<cv::Mat> descriptorDatabase;
  std::vector<cv::Mat> descriptorBogus;
  std::vector<std::vector<cv::KeyPoint> > keypointsDatabase;
  std::vector<std::vector<cv::KeyPoint> > keypointsBogus;
  cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;
  cv::Ptr<cv::DescriptorMatcher> descriptorMatcherBogus;

  // Buffer
  cv::Mat lastDescriptor;
  cv::Mat lastImage;
  std::vector<cv::KeyPoint> lastKeypoints;

  // Console
  QTextEdit* console;

  // Parameters
  int minHessian;
  std::string matcherType;
  std::string bogusFile;
};



#endif
