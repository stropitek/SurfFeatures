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

#include <QString>
#include <QTimer>

class qSlicerSurfFeaturesModuleWidgetPrivate;
class qSlicerSurfFeaturesModuleWidget;
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
  void onRansacMarginChanged(double);
  void onMinHessianChanged(int);

  void onLeftCropChanged(double);
  void onTopCropChanged(double);
  void onRightCropChanged(double);
  void onBottomCropChanged(double);

  void onShowCrop();

  void onBogusPathChanged(const QString&);
  void onTrainPathChanged(const QString&);
  void onQueryPathChanged(const QString&);

  void onTrainStartFrameChanged(int);
  void onTrainStopFrameChanged(int);
  void onBogusStartFrameChanged(int);
  void onBogusStopFrameChanged(int);
  void onQueryStartFrameChanged(int);
  void onQueryStopFrameChanged(int);

  void onComputeBogus();
  void onComputeTrain();
  void onComputeQuery();

  void onPreviousImage();
  void onNextImage();
  void onShowCurrentImage();
  void onComputeCorrespondences();
  void onSaveCurrentImage();

  void updateParameters();

  void onTogglePlay();
  void onPlayImage();
  void onPlayIntervalChanged(int);
  
  void onQueryTimer();
  void onTrainTimer();
  void onBogusTimer();
  void onCorrespondenceTimer();

  void onLogMatches();
  void onMaskFile();

  void onSaveKeypoints();
  void onLoadKeypoints();

protected:
  QScopedPointer<qSlicerSurfFeaturesModuleWidgetPrivate> d_ptr;
  
  virtual void setup();

private:
  Q_DECLARE_PRIVATE(qSlicerSurfFeaturesModuleWidget);
  Q_DISABLE_COPY(qSlicerSurfFeaturesModuleWidget);
};

#endif
