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

  connect(d->InputVolumeComboBox, SIGNAL(currentNodeChanged(vtkMRMLNode*)), this, SLOT(onVolumeSelect(vtkMRMLNode*)));
  connect(d->InputTrackerComboBox, SIGNAL(currentNodeChanged(vtkMRMLNode*)),this, SLOT(onTrackerSelect(vtkMRMLNode*)));
  connect(d->minHessianSpinBox, SIGNAL(valueChanged(int)), this, SLOT(setMinHessian(int)));
  connect(d->recordButton, SIGNAL(clicked()),this, SLOT(toggleRecord()));
  connect(d->matchButton, SIGNAL(clicked()), this, SLOT(match()));
  connect(d->showNextImageButton, SIGNAL(clicked()), this, SLOT(showNextImage()));
  connect(d->matchWithNextButton, SIGNAL(clicked()), this, SLOT(matchWithNextImage()));

  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setConsole(d->consoleDebug);
}

void qSlicerSurfFeaturesModuleWidget::onVolumeSelect(vtkMRMLNode* node)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  Q_ASSERT(d->InputVolumeComboBox);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  vtkMRMLScalarVolumeNode* snode = vtkMRMLScalarVolumeNode::SafeDownCast(node);
  logic->setObservedNode(snode);
}

void qSlicerSurfFeaturesModuleWidget::onTrackerSelect(vtkMRMLNode* node)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  Q_ASSERT(d->InputTrackerComboBox);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  vtkMRMLLinearTransformNode* tnode = vtkMRMLLinearTransformNode::SafeDownCast(node);
  logic->setObservedNode(tnode);
}

void qSlicerSurfFeaturesModuleWidget::toggleRecord()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->toggleRecord();
  if(d->recordButton->isChecked())
    d->recordButton->setText("Stop Recording");
  else
    d->recordButton->setText("Start Recording");
}

void qSlicerSurfFeaturesModuleWidget::match()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->match();
}

void qSlicerSurfFeaturesModuleWidget::setMinHessian(int minHessian)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->setMinHessian(minHessian);
}

void qSlicerSurfFeaturesModuleWidget::showNextImage()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->showNextImage();
}

void qSlicerSurfFeaturesModuleWidget::matchWithNextImage()
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  vtkSlicerSurfFeaturesLogic* logic = d->logic();
  logic->matchWithNextImage();
}