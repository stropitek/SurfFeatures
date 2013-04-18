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

// MRML includes
#include <vtkMRMLVolumeNode.h>
#include <vtkMRMLScalarVolumeNode.h>
#include <vtkImageData.h>

// stl includes
#include <fstream>

// OpenCV includes
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cxmisc.h"
#include "cv.h"

//-----------------------------------------------------------------------------
/// \ingroup Slicer_QtModules_ExtensionTemplate
class qSlicerSurfFeaturesModuleWidgetPrivate: public Ui_qSlicerSurfFeaturesModuleWidget
{
public:
  qSlicerSurfFeaturesModuleWidgetPrivate();
};

//-----------------------------------------------------------------------------
// qSlicerSurfFeaturesModuleWidgetPrivate methods

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesModuleWidgetPrivate::qSlicerSurfFeaturesModuleWidgetPrivate()
{
}

//-----------------------------------------------------------------------------
// qSlicerSurfFeaturesModuleWidget methods

//-----------------------------------------------------------------------------
qSlicerSurfFeaturesModuleWidget::qSlicerSurfFeaturesModuleWidget(QWidget* _parent)
  : Superclass( _parent )
  , d_ptr( new qSlicerSurfFeaturesModuleWidgetPrivate )
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
}

void qSlicerSurfFeaturesModuleWidget::onVolumeSelect(vtkMRMLNode* node)
{
  Q_D(qSlicerSurfFeaturesModuleWidget);
  Q_ASSERT(d->InputVolumeComboBox);
  if(!node)
    return;
  if(strcmp(node->GetClassName(),"vtkMRMLScalarVolumeNode"))
    return;
  vtkMRMLScalarVolumeNode* sv_node = vtkMRMLScalarVolumeNode::SafeDownCast(node);
  cv::FileNodeIterator mser;
  vtkImageData* data = sv_node->GetImageData();
  if(!data)
    return;
  int dims[3];
  data->GetDimensions(dims);
  std::cout << "data dimensions " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
  std::cout << node->GetClassName() << std::endl;
}
