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
#include <QFileDialog>

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
  QTimer* queryTimer;
  QTimer* trainTimer;
  QTimer* bogusTimer;
  QTimer* correspondenceTimer;
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
  delete queryTimer;
  delete trainTimer;
  delete bogusTimer;
  delete correspondenceTimer;
}

qSlicerSurfFeaturesModuleWidgetPrivate::qSlicerSurfFeaturesModuleWidgetPrivate(qSlicerSurfFeaturesModuleWidget& object): q_ptr(&object)
{
  timer = new QTimer;
  queryTimer = new QTimer;
  trainTimer = new QTimer;
  bogusTimer = new QTimer;
  correspondenceTimer = new QTimer;
  
  timer->setInterval(100);
  queryTimer->setInterval(0);
  trainTimer->setInterval(0);
  bogusTimer->setInterval(0);
  correspondenceTimer->setInterval(0);
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

  connect(d->ransacMarginSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onRansacMarginChanged(double)));
  connect(d->minHessianSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onMinHessianChanged(int)));

  connect(d->leftCropSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onLeftCropChanged(double)));
  connect(d->topCropSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onTopCropChanged(double)));
  connect(d->rightCropSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onRightCropChanged(double)));
  connect(d->bottomCropSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onBottomCropChanged(double)));

  connect(d->showCropButton, SIGNAL(clicked()), this, SLOT(onShowCrop()));

  connect(d->bogusPathLineEdit, SIGNAL(currentPathChanged(const QString&)), this, SLOT(onBogusPathChanged(const QString&)));
  connect(d->trainPathLineEdit, SIGNAL(currentPathChanged(const QString&)), this, SLOT(onTrainPathChanged(const QString&)));
  connect(d->queryPathLineEdit, SIGNAL(currentPathChanged(const QString&)), this, SLOT(onQueryPathChanged(const QString&)));

  connect(d->bogusFromSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onBogusStartFrameChanged(int)));
  connect(d->bogusToSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onBogusStopFrameChanged(int)));
  connect(d->trainFromSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onTrainStartFrameChanged(int)));
  connect(d->trainToSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onTrainStopFrameChanged(int)));
  connect(d->queryFromSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onQueryStartFrameChanged(int)));
  connect(d->queryToSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onQueryStopFrameChanged(int)));

  connect(d->computeBogusButton, SIGNAL(clicked()), this, SLOT(onComputeBogus()));
  connect(d->computeTrainButton, SIGNAL(clicked()), this, SLOT(onComputeTrain()));
  connect(d->computeQueryButton, SIGNAL(clicked()), this, SLOT(onComputeQuery()));

  connect(d->previousImageButton, SIGNAL(clicked()), this, SLOT(onPreviousImage()));
  connect(d->nextImageButton, SIGNAL(clicked()), this, SLOT(onNextImage()));
  connect(d->showCurrentImageButton, SIGNAL(clicked()), this, SLOT(onShowCurrentImage())); 
  connect(d->saveImageButton, SIGNAL(clicked()), this, SLOT(onSaveCurrentImage()));

  connect(d->computeCorrespondencesButton, SIGNAL(clicked()), this, SLOT(onComputeCorrespondences()));

  connect(d->timer, SIGNAL(timeout()), this, SLOT(onPlayImage()));
  connect(d->playButton, SIGNAL(clicked()), this, SLOT(onTogglePlay()));
  connect(d->playIntervalSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onPlayIntervalChanged(int)));
  
  connect(d->queryTimer, SIGNAL(timeout()), this, SLOT(onQueryTimer()));
  connect(d->trainTimer, SIGNAL(timeout()), this, SLOT(onTrainTimer()));
  connect(d->bogusTimer, SIGNAL(timeout()), this, SLOT(onBogusTimer()));
  connect(d->correspondenceTimer, SIGNAL(timeout()), this, SLOT(onCorrespondenceTimer()));
  
  connect(d->logMatchesButton, SIGNAL(clicked()), this, SLOT(onLogMatches()));
  connect(d->maskFileButton, SIGNAL(clicked()), this, SLOT(onMaskFile()));

  connect(d->saveKeypointsButton, SIGNAL(clicked()), this, SLOT(onSaveKeypoints()));
  connect(d->loadKeypointsButton, SIGNAL(clicked()), this, SLOT(onLoadKeypoints()));

  // call some slots to make them effective from start


  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setConsole(d->consoleDebug);

  qvtkConnect(logic, vtkCommand::ModifiedEvent, this, SLOT(updateParameters()));

  this->updateParameters();

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
SLOTDEF_1(int,onQueryStartFrameChanged,setQueryStartFrame);
SLOTDEF_1(int,onTrainStopFrameChanged,setTrainStopFrame);
SLOTDEF_1(int,onBogusStopFrameChanged,setBogusStopFrame);
SLOTDEF_1(int,onQueryStopFrameChanged,setQueryStopFrame);

SLOTDEF_0(onPlayImage, play);

SLOTDEF_0(onPreviousImage, previousImage);
SLOTDEF_0(onNextImage,nextImage);
SLOTDEF_0(onShowCurrentImage,showCurrentImage);

SLOTDEF_0(onLogMatches, writeMatches);

SLOTDEF_0(onQueryTimer, computeNextQuery);
SLOTDEF_0(onTrainTimer, computeNextTrain);
SLOTDEF_0(onBogusTimer, computeNextBogus);
SLOTDEF_0(onCorrespondenceTimer, computeNextInterSliceCorrespondence);

void qSlicerSurfFeaturesModuleWidget::onComputeQuery()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  if(d->queryTimer->isActive())
    return;
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->computeQuery();
  d->queryTimer->start();
}

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

void qSlicerSurfFeaturesModuleWidget::onComputeCorrespondences()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  if(d->correspondenceTimer->isActive())
    return;
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->startInterSliceCorrespondence();
  d->correspondenceTimer->start();
}

void qSlicerSurfFeaturesModuleWidget::onTogglePlay()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  if(d->timer->isActive()){
    d->playButton->setText(QString("Start Playing"));
    d->timer->stop();
  }
  else {
    d->playButton->setText(QString("Stop Playing"));
    d->timer->start();
  }
}

void qSlicerSurfFeaturesModuleWidget::onPlayIntervalChanged(int value)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  d->timer->setInterval(value);
}

void qSlicerSurfFeaturesModuleWidget::onTrainPathChanged(const QString& path)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setTrainFile(path.toStdString());
}

void qSlicerSurfFeaturesModuleWidget::onQueryPathChanged(const QString& path)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setQueryFile(path.toStdString());
}

void qSlicerSurfFeaturesModuleWidget::onBogusPathChanged(const QString& path)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setBogusFile(path.toStdString());
}

void qSlicerSurfFeaturesModuleWidget::onMaskFile()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  QString fileName = QFileDialog::getOpenFileName(this,
     tr("Open Image"), "/home/maskfile", tr("Image Files (*.png *.jpg *.bmp)"));
  std::string fn = fileName.toStdString();
  if(!fn.empty())
    d->logic()->setMaskFile(fn);
}

void qSlicerSurfFeaturesModuleWidget::onSaveKeypoints()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  QString directory = QFileDialog::getExistingDirectory(this, tr("Save To Directory"), "/home");
  d->logic()->saveKeypointsAndDescriptors(directory.toStdString(), d->saveLoadTypeComboBox->currentText().toLower().toStdString());

}

void qSlicerSurfFeaturesModuleWidget::onLoadKeypoints()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  QString directory = QFileDialog::getExistingDirectory(this, tr("Save To Directory"), "/home");
  d->logic()->loadKeypointsAndDescriptors(directory.toStdString(), d->saveLoadTypeComboBox->currentText().toLower().toStdString());
}

void qSlicerSurfFeaturesModuleWidget::onSaveCurrentImage()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  QString fileName = QFileDialog::getSaveFileName(this, tr("Open Image"), "/home/slice", tr("Image Files (*.png *.jpg *.bmp)"));
  d->logic()->saveCurrentImage(fileName.toStdString());
}


void qSlicerSurfFeaturesModuleWidget::updateParameters()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  
  if(d->queryTimer->isActive() && !logic->isQueryLoading())
    d->queryTimer->stop();
    
  if(d->trainTimer->isActive() && !logic->isTrainLoading())
    d->trainTimer->stop();
  
  if(d->bogusTimer->isActive() && !logic->isBogusLoading())
    d->bogusTimer->stop();
    
  if(d->correspondenceTimer->isActive() && !logic->isCorrespondenceComputing()) {
    d->correspondenceTimer->stop();
    logic->resetResults();
  }

  d->queryFromSpinBox->setValue(logic->getQueryStartFrame());
  d->queryToSpinBox->setValue(logic->getQueryStopFrame());
  d->trainFromSpinBox->setValue(logic->getTrainStartFrame());
  d->trainToSpinBox->setValue(logic->getTrainStopFrame());
  d->bogusFromSpinBox->setValue(logic->getBogusStartFrame());
  d->bogusToSpinBox->setValue(logic->getBogusStopFrame());

  d->queryProgressBar->setValue(logic->getQueryProgress());
  d->trainProgressBar->setValue(logic->getTrainProgress());
  d->bogusProgressBar->setValue(logic->getBogusProgress());
  d->correspondenceProgressBar->setValue(logic->getCorrespondenceProgress());

  d->leftCropSpinBox->setValue(logic->getLeftCrop());
  d->topCropSpinBox->setValue(logic->getTopCrop());
  d->rightCropSpinBox->setValue(logic->getRightCrop());
  d->bottomCropSpinBox->setValue(logic->getBottomCrop());

  d->bogusPathLineEdit->setCurrentPath(logic->getBogusFile().c_str());
  d->trainPathLineEdit->setCurrentPath(logic->getTrainFile().c_str());
  d->queryPathLineEdit->setCurrentPath(logic->getQueryFile().c_str());
  
  d->minHessianSpinBox->setValue(logic->getMinHessian());
  d->maskFileLabel->setText(logic->getMaskFile().c_str());
}