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

// STD includes
#include <cstdlib>
#include <fstream>

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

  // Attributes
private:
  vtkMRMLScalarVolumeNode* observedNode;
  std::ofstream ofs;
};

#endif
