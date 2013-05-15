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
  void match();
  void setConsole(QTextEdit* console);

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
  void matchImageToDatabase(vtkMRMLNode* node);
  void stopWatchWrite(std::ostringstream& oss);
  void resetConsoleFont();
  vtkImageData* cropData(vtkImageData* data);
  cv::Mat convertImage(vtkImageData* data);

  // Attributes
private:
  vtkMRMLScalarVolumeNode* observedVolume;
  vtkMRMLLinearTransformNode* observedTransform;
  clock_t lastImageModified;
  clock_t initSurf;
  clock_t initTime;
  clock_t lastStopWatch;
  bool recording;
  bool matchNext;
  cv::FlannBasedMatcher flannMatcher;
  std::vector<cv::Mat> descriptorDatabase;
  QTextEdit* console;
  int minHessian;
};

#endif
