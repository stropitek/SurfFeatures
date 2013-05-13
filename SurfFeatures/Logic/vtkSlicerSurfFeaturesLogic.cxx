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
#include <vtkExtractVOI.h>

// STD includes
#include <cassert>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <iostream>

// MRML includes
#include <vtkMRMLVolumeNode.h>
#include <vtkMRMLScalarVolumeNode.h>
#include <vtkImageData.h>
#include <vtkImageExport.h>

// OpenCV includes
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerSurfFeaturesLogic);

//----------------------------------------------------------------------------
vtkSlicerSurfFeaturesLogic::vtkSlicerSurfFeaturesLogic()
{
  this->observedVolume = NULL;
  this->observedTransform = NULL;
  this->lastImageModified = clock();
  this->lastStopWatch = clock();
  this->initTime = clock();
  this->recording = false;
  this->matchNext = false;
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
  // Only if recording
  if(!this->recording && !this->matchNext)
    return;
  // Verify validity of node
  if(!node)
    return;
  // Verify we have scalar volume node
  if(strcmp(node->GetClassName(),"vtkMRMLScalarVolumeNode"))
    return;

  if(this->matchNext)
    this->recording = false;

  if(this->recording)
    this->recordData(node);
  else if(this->matchNext)
  {
    this->matchImageToDatabase(node);
    this->matchNext = false;
  }
}

void vtkSlicerSurfFeaturesLogic::recordData(vtkMRMLNode* node)
{
  vtkMRMLScalarVolumeNode* sv_node = vtkMRMLScalarVolumeNode::SafeDownCast(node);

  // Surf detector initialization
  int minHessian = 400;
  cv::SurfFeatureDetector detector( minHessian );
  cv::SurfDescriptorExtractor extractor;
  
  // Get and verify image data
  vtkImageData* data = sv_node->GetImageData();
  if(!data)
    return;

  // Get image dimensions
  int dims[3];
  data->GetDimensions(dims);

  // Crop image
  vtkSmartPointer<vtkExtractVOI> extractVOI = vtkExtractVOI::New();
  extractVOI->SetInput(data);
  extractVOI->SetVOI(dims[0]/5.,3.*dims[0]/3.9,dims[1]/6,3.*dims[1]/4, 0, 0);
  extractVOI->SetSampleRate(1,1,1);
  extractVOI->Update();
  vtkImageData* croppedData = extractVOI->GetOutput();

  /*
  ofs << "Node class name: " << node->GetClassName() << std::endl;
  ofs << "Data dimensions " << dims[0] << " " << dims[1] << " " << dims[2] << std::endl;
  ofs << "Data scalar type: " << data->GetScalarTypeAsString() << std::endl << "Data type int: " << data->GetScalarType() << std::endl;
  //*/ 

  // Export data to a c pointer
  vtkImageExport *exporter = vtkImageExport::New();
  exporter->SetInput(croppedData);
  croppedData->GetDimensions(dims);
  int numel = dims[0]*dims[1]*dims[2];
  void* void_ptr = malloc(numel);
  unsigned char* char_ptr = (unsigned char*) void_ptr;
  exporter->SetExportVoidPointer(void_ptr);
  exporter->Export();

  // Data as opencv matrix
  cv::Mat mat(dims[1],dims[0],CV_8U ,void_ptr);

  std::vector<cv::KeyPoint> keypoints;
  clock_t startTime = clock();
  detector.detect(mat,keypoints);
  //for(int i = 0; i<keypoints.size(); i++){
  //  cv::Point2f p = keypoints[i].pt;
  //  int loc[3] = { (int)p.x, (int)(p.y), 1};
  //  double point[3];
  //  vtkIdType id = croppedData->ComputePointId(loc);
  //  croppedData->GetPoint(id,point);
  //  std::ostringstream oss;
  //  oss << "Keypoint (" << loc[0] << "," << loc[1] << "," << loc[2] << ") ";
  //  oss << "--> data (" << point[0] << "," << point[1] << "," << point[2] << ") " << this->stopWatchWrite() << "<br>";
  //  this->console->insertPlainText(oss.str().c_str());
  //}
  cv::Mat descriptors;
  extractor.compute(mat, keypoints, descriptors);
  this->descriptorDatabase.push_back(descriptors);
  clock_t endTime = clock();
  clock_t clockTicksTaken = endTime - startTime;
  double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
  std::ostringstream oss;
  oss << "time taken for feature detector: " << timeInSeconds << " seconds.";
  //this->stopWatchWrite(oss);
  //this->console->insertPlainText(oss.str().c_str());
  
  

 

  /*//
  //-- Draw keypoints
  cv::Mat img_keypoints;

  cv::drawKeypoints( mat, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
  cv::imshow("Keypoints 1", img_keypoints );

  cv::waitKey(0);
  //*/

  free(void_ptr);
}

void vtkSlicerSurfFeaturesLogic::matchImageToDatabase(vtkMRMLNode* node)
{
  vtkMRMLScalarVolumeNode* sv_node = vtkMRMLScalarVolumeNode::SafeDownCast(node);

  this->flannMatcher.clear();
  this->flannMatcher.add(this->descriptorDatabase);
  std::vector<std::vector<cv::DMatch> > matches;
  //this->flannMatcher.knnMatch(descriptors,matches,3);

}

void vtkSlicerSurfFeaturesLogic
::ProcessMRMLNodesEvents( vtkObject* caller, unsigned long event, void * callData )
{
  if ( caller == NULL )
    {
    return;
    }
  
  if(event == vtkMRMLVolumeNode::ImageDataModifiedEvent)
  {
    if(this->recording){
      std::ostringstream oss;
      oss << "New ImageDataModifiedEvent";
      this->stopWatchWrite(oss);
      this->console->insertPlainText(oss.str().c_str());
    }
    //clock_t now = clock();
    //clock_t clockTicksTaken = now - this->lastImageModified;
    //this->lastImageModified = now;
    //double timeInSeconds = clockTicksTaken / (double) CLOCKS_PER_SEC;
    //vtkMRMLScalarVolumeNode* scalarNode = vtkMRMLScalarVolumeNode::SafeDownCast( caller );
    //this->displayFeatures(scalarNode);
  }
  else if(event == vtkMRMLLinearTransformNode::TransformModifiedEvent)
  {
    if(this->recording){
      std::ostringstream oss;
      oss << "New TransformModifiedEvent ";
      this->stopWatchWrite(oss);
      this->console->insertPlainText(oss.str().c_str());
    }
  }
  else
    this->Superclass::ProcessMRMLNodesEvents( caller, event, callData );
}

void vtkSlicerSurfFeaturesLogic::setObservedNode(vtkMRMLScalarVolumeNode *snode)
{
  if(snode==this->observedVolume)
    return;
  if(!snode){
    this->removeObservedVolume();
    return;
  }

  int wasModifying = this->StartModify();
  if(this->observedVolume)
    vtkSetAndObserveMRMLNodeMacro( this->observedVolume, 0 );

  vtkMRMLScalarVolumeNode* newNode = NULL;
  vtkSmartPointer< vtkIntArray > events = vtkSmartPointer< vtkIntArray >::New();
  //events->InsertNextValue( vtkCommand::ModifiedEvent );
  events->InsertNextValue( vtkMRMLVolumeNode::ImageDataModifiedEvent );
  vtkSetAndObserveMRMLNodeEventsMacro( newNode, snode, events );
  this->observedVolume = newNode;
  
  this->EndModify( wasModifying );
}

void vtkSlicerSurfFeaturesLogic::setObservedNode(vtkMRMLLinearTransformNode *tnode)
{
  if(tnode==this->observedTransform)
    return;
  if(!tnode){
    this->removeObservedTransform();
    return;
  }

  int wasModifying = this->StartModify();
  if(this->observedTransform)
    vtkSetAndObserveMRMLNodeMacro(this->observedTransform, 0);

  vtkMRMLLinearTransformNode* newNode = NULL;
  vtkSmartPointer< vtkIntArray > events = vtkSmartPointer< vtkIntArray >::New();
  events->InsertNextValue(vtkMRMLTransformableNode::TransformModifiedEvent);
  vtkSetAndObserveMRMLNodeEventsMacro(newNode,tnode,events);
  this->observedTransform = newNode;

  this->EndModify(wasModifying);
}

void vtkSlicerSurfFeaturesLogic::removeObservedVolume()
{
  if(this->observedVolume)
    vtkSetAndObserveMRMLNodeMacro( this->observedVolume, 0 );
  this->observedVolume = NULL;
}

void vtkSlicerSurfFeaturesLogic::removeObservedTransform()
{
  if(this->observedTransform)
    vtkSetAndObserveMRMLNodeMacro( this->observedTransform, 0 );
  this->observedTransform = NULL;
}

void vtkSlicerSurfFeaturesLogic::toggleRecord()
{
  this->resetConsoleFont();
  this->recording = !this->recording;
  std::ostringstream oss;
  oss << "Toggling recording to " << this->recording;
  this->stopWatchWrite(oss);
  this->console->insertPlainText(oss.str().c_str());
}

void vtkSlicerSurfFeaturesLogic::match()
{
  this->matchNext = true;
}

void vtkSlicerSurfFeaturesLogic::setConsole(QTextEdit* console)
{
  this->console = console;
  this->resetConsoleFont();
}

void vtkSlicerSurfFeaturesLogic::stopWatchWrite(std::ostringstream& oss)
{
  clock_t endTime = clock();
  double timeInSeconds = (endTime - this->initTime) / (double) CLOCKS_PER_SEC;
  double intervalInMiliSeconds = (endTime - this->lastStopWatch)/(double) CLOCKS_PER_SEC * 1000;
  std::string output = oss.str();
  int iSpaces = 50 - output.size();
  std::string spaces("");
  if(iSpaces > 0)
  {
    spaces = std::string(iSpaces,' ');
  }
  oss << spaces << "-- " << timeInSeconds  << " s -- + " << intervalInMiliSeconds << " ms --" << std::endl;
  this->lastStopWatch = endTime;
}

void vtkSlicerSurfFeaturesLogic::resetConsoleFont()
{
  QFont font("Courier",8);
  font.setStretch(QFont::UltraCondensed);
  this->console->setCurrentFont(font);
}