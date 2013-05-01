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

// SurfFeatures Logic includes
#include "vtkSlicerSurfFeaturesLogic.h"

// MRML includes

// VTK includes
#include <vtkNew.h>

// STD includes
#include <cassert>
#include <ctime>
#include <fstream>

// MRML includes
#include <vtkMRMLVolumeNode.h>
#include <vtkMRMLScalarVolumeNode.h>
#include <vtkImageData.h>
#include <vtkImageExport.h>

// OpenCV includes
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/nonfree/features2d.hpp>

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerSurfFeaturesLogic);

//----------------------------------------------------------------------------
vtkSlicerSurfFeaturesLogic::vtkSlicerSurfFeaturesLogic()
{
  this->observedNode = NULL;
  ofs.open("C:\\DK\\hello.txt");
}

//----------------------------------------------------------------------------
vtkSlicerSurfFeaturesLogic::~vtkSlicerSurfFeaturesLogic()
{
}

//----------------------------------------------------------------------------
void vtkSlicerSurfFeaturesLogic::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

//---------------------------------------------------------------------------
void vtkSlicerSurfFeaturesLogic::SetMRMLSceneInternal(vtkMRMLScene * newScene)
{
  //cv::FileNodeIterator cf;
  vtkNew<vtkIntArray> events;
  events->InsertNextValue(vtkMRMLScene::NodeAddedEvent);
  events->InsertNextValue(vtkMRMLScene::NodeRemovedEvent);
  events->InsertNextValue(vtkMRMLScene::EndBatchProcessEvent);
  this->SetAndObserveMRMLSceneEventsInternal(newScene, events.GetPointer());
}

//-----------------------------------------------------------------------------
void vtkSlicerSurfFeaturesLogic::RegisterNodes()
{
  assert(this->GetMRMLScene() != 0);
}

//---------------------------------------------------------------------------
void vtkSlicerSurfFeaturesLogic::UpdateFromMRMLScene()
{
  assert(this->GetMRMLScene() != 0);
}

//---------------------------------------------------------------------------
void vtkSlicerSurfFeaturesLogic
::OnMRMLSceneNodeAdded(vtkMRMLNode* vtkNotUsed(node))
{
}

//---------------------------------------------------------------------------
void vtkSlicerSurfFeaturesLogic
::OnMRMLSceneNodeRemoved(vtkMRMLNode* vtkNotUsed(node))
{
}

void vtkSlicerSurfFeaturesLogic::displayFeatures(vtkMRMLNode* node)
{
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
  /*
  ofs << "Node class name: " << node->GetClassName() << std::endl;
  ofs << "Data dimensions " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
  ofs << "Data scalar type: " << data->GetScalarTypeAsString() << std::endl << "Data type int: " << data->GetScalarType() << std::endl;
  //*/

  
  vtkImageExport *exporter = vtkImageExport::New();
  exporter->SetInput(data);

  int numel = dims[0]*dims[1]*dims[2];
  void* void_ptr = malloc(numel);
  unsigned char* char_ptr = (unsigned char*) void_ptr;
  exporter->SetExportVoidPointer(void_ptr);
  exporter->Export();
  unsigned char* c_data = (unsigned char*)exporter->GetExportVoidPointer();
  ofs << "A point: " << c_data[3200] << std::endl;

  cv::Mat mat(dims[1],dims[0],CV_8U ,void_ptr);

  std::vector<cv::KeyPoint> keypoints;


  clock_t startTime = clock();
  detector.detect(mat,keypoints);
  clock_t endTime = clock();
  clock_t clockTicksTaken = endTime - startTime;
  double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
  ofs << "time taken for feature detector: " << timeInSeconds << " seconds." << std::endl;
  
  

 

  /*
  //-- Draw keypoints
  cv::Mat img_keypoints;

  cv::drawKeypoints( mat, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  cv::imshow("Keypoints 1", img_keypoints );

  cv::waitKey(0);
  //*/

  free(void_ptr);
}

void vtkSlicerSurfFeaturesLogic
::ProcessMRMLNodesEvents( vtkObject* caller, unsigned long event, void * callData )
{
  if ( caller == NULL )
    {
    return;
    }
  
  if ( event != vtkCommand::ModifiedEvent
       && event != vtkMRMLVolumeNode::ImageDataModifiedEvent )
    {
    this->Superclass::ProcessMRMLNodesEvents( caller, event, callData );
    }
  else
  {
    vtkMRMLScalarVolumeNode* scalarNode = vtkMRMLScalarVolumeNode::SafeDownCast( caller );
    this->displayFeatures(scalarNode);
  }
}

void vtkSlicerSurfFeaturesLogic::setObservedNode(vtkMRMLScalarVolumeNode *snode)
{
  if(snode==this->observedNode)
    return;
  if(!snode)
    return;

  int wasModifying = this->StartModify();
  if(this->observedNode)
    vtkSetAndObserveMRMLNodeMacro( this->observedNode, 0 );

  vtkMRMLScalarVolumeNode* newNode = NULL;
  vtkSmartPointer< vtkIntArray > events = vtkSmartPointer< vtkIntArray >::New();
  events->InsertNextValue( vtkCommand::ModifiedEvent );
  events->InsertNextValue( vtkMRMLVolumeNode::ImageDataModifiedEvent );
  vtkSetAndObserveMRMLNodeEventsMacro( newNode, snode, events );
  this->observedNode = newNode;
  
  this->EndModify( wasModifying );
}

