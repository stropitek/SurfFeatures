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

// Qt includes
#include <QDebug>
#include <QTimer>

// SlicerQt includes
#include "qSlicerSurfFeaturesModuleWidget.h"
#include "ui_qSlicerSurfFeaturesModuleWidget.h"

// stl includes
#include <fstream>

// Logic
#include "vtkSlicerSurfFeaturesLogic.h"
#include "util_macros.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerSurfFeaturesModuleWidgetPrivate: public Ui_qSlicerSurfFeaturesModuleWidget
{
  Q_DECLARE_PUBLIC(qSlicerSurfFeaturesModuleWidget)
protected:
  qSlicerSurfFeaturesModuleWidget* const q_ptr;
  QTimer* timer;
  QTimer* trainTimer;
  QTimer* bogusTimer;
public:
  ~qSlicerSurfFeaturesModuleWidgetPrivate();
  qSlicerSurfFeaturesModuleWidgetPrivate(qSlicerSurfFeaturesModuleWidget& object);

  vtkSlicerSurfFeaturesLogic* logic() const;
};

//-----------------------------------------------------------------------------
// qSlicerSurfFeaturesModuleWidgetPrivate methods

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesModuleWidgetPrivate::~qSlicerSurfFeaturesModuleWidgetPrivate()
{
  delete timer;
  delete trainTimer;
  delete bogusTimer;
}

qSlicerSurfFeaturesModuleWidgetPrivate::qSlicerSurfFeaturesModuleWidgetPrivate(qSlicerSurfFeaturesModuleWidget& object): q_ptr(&object)
{
  timer = new QTimer;
  trainTimer = new QTimer;
  bogusTimer = new QTimer;
  
  timer->setInterval(100);
  trainTimer->setInterval(0);
  bogusTimer->setInterval(0);
}

vtkSlicerSurfFeaturesLogic* qSlicerSurfFeaturesModuleWidgetPrivate::logic() const
{
  Q_Q(const qSlicerSurfFeaturesModuleWidget);
  return vtkSlicerSurfFeaturesLogic::SafeDownCast(q->logic());
}

//-----------------------------------------------------------------------------
// qSlicerSurfFeaturesModuleWidget methods

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesModuleWidget::qSlicerSurfFeaturesModuleWidget(QWidget* _parent)
  : Superclass( _parent )
  , d_ptr( new qSlicerSurfFeaturesModuleWidgetPrivate(*this) )
{
}

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesModuleWidget::~qSlicerSurfFeaturesModuleWidget()
{
}

//-----------------------------------------------------------------------------
void qSlicerSurfFeaturesModuleWidget::setup()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  d->setupUi(this);
  this->Superclass::setup();
  
  connect(d->inputVolumeNodeComboBox, SIGNAL(currentNodeChanged(vtkMRMLNode*)), this, SLOT(onInputVolumeSelect(vtkMRMLNode*)));

  connect(d->ransacMarginSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onRansacMarginChanged(double)));
  connect(d->minHessianSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onMinHessianChanged(int)));

  connect(d->leftCropSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onLeftCropChanged(double)));
  connect(d->topCropSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onTopCropChanged(double)));
  connect(d->rightCropSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onRightCropChanged(double)));
  connect(d->bottomCropSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onBottomCropChanged(double)));

  connect(d->showCropButton, SIGNAL(clicked()), this, SLOT(onShowCrop()));

  connect(d->bogusPathLineEdit, SIGNAL(currentPathChanged(const QString&)), this, SLOT(onBogusPathChanged(const QString&)));
  connect(d->trainPathLineEdit, SIGNAL(currentPathChanged(const QString&)), this, SLOT(onTrainPathChanged(const QString&)));

  connect(d->bogusFromSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onBogusStartFrameChanged(int)));
  connect(d->bogusToSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onBogusStopFrameChanged(int)));
  connect(d->trainFromSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onTrainStartFrameChanged(int)));
  connect(d->trainToSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onTrainStopFrameChanged(int)));

  connect(d->computeBogusButton, SIGNAL(clicked()), this, SLOT(onComputeBogus()));
  connect(d->computeTrainButton, SIGNAL(clicked()), this, SLOT(onComputeTrain()));


  connect(d->timer, SIGNAL(timeout()), this, SLOT(onPlayImage()));
  connect(d->registerButton, SIGNAL(clicked()), this, SLOT(onRegister()));
  
  connect(d->trainTimer, SIGNAL(timeout()), this, SLOT(onTrainTimer()));
  connect(d->bogusTimer, SIGNAL(timeout()), this, SLOT(onBogusTimer()));
  

  // call some slots to make them effective from start


  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setConsole(d->consoleDebug);

  qvtkConnect(logic, vtkCommand::ModifiedEvent, this, SLOT(updateParameters()));

  this->updateParameters();

}

void qSlicerSurfFeaturesModuleWidget::onInputVolumeSelect(vtkMRMLNode* node)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  Q_ASSERT(d->InputVolumeComboBox);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  vtkMRMLScalarVolumeNode* snode = vtkMRMLScalarVolumeNode::SafeDownCast(node);
  logic->setObservedVolume(snode);
}

SLOTDEF_1(int,onMinHessianChanged, setMinHessian);
SLOTDEF_1(double, onRansacMarginChanged, setRansacMargin);

void qSlicerSurfFeaturesModuleWidget::onLeftCropChanged(double val)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setLeftCrop(val);
  logic->showCropFirstImage();
}

void qSlicerSurfFeaturesModuleWidget::onTopCropChanged(double val)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setTopCrop(val);
  logic->showCropFirstImage();
}

void qSlicerSurfFeaturesModuleWidget::onRightCropChanged(double val)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setRightCrop(val);
  logic->showCropFirstImage();
}

void qSlicerSurfFeaturesModuleWidget::onBottomCropChanged(double val)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setBottomCrop(val);
  logic->showCropFirstImage();
}

SLOTDEF_0(onShowCrop, showCropFirstImage);


SLOTDEF_1(int,onTrainStartFrameChanged,setTrainStartFrame);
SLOTDEF_1(int,onBogusStartFrameChanged,setBogusStartFrame);
SLOTDEF_1(int,onTrainStopFrameChanged,setTrainStopFrame);
SLOTDEF_1(int,onBogusStopFrameChanged,setBogusStopFrame);

SLOTDEF_0(onRegister, play);
SLOTDEF_0(onTrainTimer, computeNextTrain);
SLOTDEF_0(onBogusTimer, computeNextBogus);

void qSlicerSurfFeaturesModuleWidget::onComputeTrain()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  if(d->trainTimer->isActive())
    return;
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->computeTrain();
  d->trainTimer->start();
}

void qSlicerSurfFeaturesModuleWidget::onComputeBogus()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  if(d->bogusTimer->isActive())
    return;
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->computeBogus();
  d->bogusTimer->start();
}


void qSlicerSurfFeaturesModuleWidget::onTrainPathChanged(const QString& path)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setTrainFile(path.toStdString());
}


void qSlicerSurfFeaturesModuleWidget::onBogusPathChanged(const QString& path)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setBogusFile(path.toStdString());
}

void qSlicerSurfFeaturesModuleWidget::updateParameters()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
    
  if(d->trainTimer->isActive() && !logic->isTrainLoading())
    d->trainTimer->stop();
  
  if(d->bogusTimer->isActive() && !logic->isBogusLoading())
    d->bogusTimer->stop();
    

  d->trainFromSpinBox->setValue(logic->getTrainStartFrame());
  d->trainToSpinBox->setValue(logic->getTrainStopFrame());
  d->bogusFromSpinBox->setValue(logic->getBogusStartFrame());
  d->bogusToSpinBox->setValue(logic->getBogusStopFrame());

  d->trainProgressBar->setValue(logic->getTrainProgress());
  d->bogusProgressBar->setValue(logic->getBogusProgress());

  d->leftCropSpinBox->setValue(logic->getLeftCrop());
  d->topCropSpinBox->setValue(logic->getTopCrop());
  d->rightCropSpinBox->setValue(logic->getRightCrop());
  d->bottomCropSpinBox->setValue(logic->getBottomCrop());

  d->bogusPathLineEdit->setCurrentPath(logic->getBogusFile().c_str());
  d->trainPathLineEdit->setCurrentPath(logic->getTrainFile().c_str());
  
  d->minHessianSpinBox->setValue(logic->getMinHessian());
}