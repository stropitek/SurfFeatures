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
}

qSlicerSurfFeaturesModuleWidgetPrivate::qSlicerSurfFeaturesModuleWidgetPrivate(qSlicerSurfFeaturesModuleWidget& object): q_ptr(&object)
{
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

  connect(d->minHessianSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onMinHessianChanged(int)));

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
  

  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setConsole(d->consoleDebug);
}


SLOTDEF_1(int,onMinHessianChanged, setMinHessian);

SLOTDEF_1(int,onTrainStartFrameChanged,setTrainStartFrame);
SLOTDEF_1(int,onBogusStartFrameChanged,setBogusStartFrame);
SLOTDEF_1(int,onQueryStartFrameChanged,setQueryStartFrame);
SLOTDEF_1(int,onTrainStopFrameChanged,setTrainStopFrame);
SLOTDEF_1(int,onBogusStopFrameChanged,setBogusStopFrame);
SLOTDEF_1(int,onQueryStopFrameChanged,setQueryStopFrame);


SLOTDEF_0(onComputeBogus,computeBogus);
SLOTDEF_0(onComputeTrain,computeTrain);
SLOTDEF_0(onComputeQuery,computeQuery);

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
