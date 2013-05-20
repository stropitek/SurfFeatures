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

#ifndef __qSlicerSurfFeaturesModuleWidget_h
#define __qSlicerSurfFeaturesModuleWidget_h

// SlicerQt includes
#include "qSlicerAbstractModuleWidget.h"

#include "qSlicerSurfFeaturesModuleExport.h"

class qSlicerSurfFeaturesModuleWidgetPrivate;
class vtkMRMLNode;

/// \ingroup Slicer_QtModules_ExtensionTemplate
class Q_SLICER_QTMODULES_SURFFEATURES_EXPORT qSlicerSurfFeaturesModuleWidget :
  public qSlicerAbstractModuleWidget
{
  Q_OBJECT

public:

  typedef qSlicerAbstractModuleWidget Superclass;
  qSlicerSurfFeaturesModuleWidget(QWidget *parent=0);
  virtual ~qSlicerSurfFeaturesModuleWidget();

public slots:
  void onVolumeSelect(vtkMRMLNode*);
  void onTrackerSelect(vtkMRMLNode*);
  void toggleRecord();
  void match();
  void setMinHessian(int);
  void showNextImage();
  void matchWithNextImage();

protected:
  QScopedPointer<qSlicerSurfFeaturesModuleWidgetPrivate> d_ptr;
  
  virtual void setup();

private:
  Q_DECLARE_PRIVATE(qSlicerSurfFeaturesModuleWidget);
  Q_DISABLE_COPY(qSlicerSurfFeaturesModuleWidget);
};

#endif
