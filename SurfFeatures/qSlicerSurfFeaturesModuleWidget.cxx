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
#include <vtkImageExport.h>

// stl includes
#include <fstream>

// OpenCV includes

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/features2d.hpp>


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
  int minHessian = 400;
  cv::SurfFeatureDetector detector( minHessian );
  vtkImageData* data = sv_node->GetImageData();
  if(!data)
    return;
  int dims[3];
  data->GetDimensions(dims);
  std::ofstream os("C:\\DK\\hello.txt");
  os << "Node class name: " << node->GetClassName() << std::endl;
  os << "Data dimensions " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
  os << "Data scalar type: " << data->GetScalarTypeAsString() << std::endl << "Data type int: " << data->GetScalarType() << std::endl;

  
  vtkImageExport *exporter = vtkImageExport::New();
  exporter->SetInput(data);

  int numel = dims[0]*dims[1]*dims[2];
  void* void_ptr = malloc(numel);
  unsigned char* char_ptr = (unsigned char*) void_ptr;
  exporter->SetExportVoidPointer(void_ptr);
  exporter->Export();
  unsigned char* c_data = (unsigned char*)exporter->GetExportVoidPointer();
  os << "A point: " << c_data[3200] << std::endl;

  cv::Mat mat(dims[1],dims[0],CV_8U ,void_ptr);

  std::vector<cv::KeyPoint> keypoints;
  detector.detect(mat,keypoints);

  //-- Draw keypoints
  cv::Mat img_keypoints;

  cv::drawKeypoints( mat, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  cv::imshow("Keypoints 1", img_keypoints );

  cv::waitKey(0);

  free(void_ptr);
}
