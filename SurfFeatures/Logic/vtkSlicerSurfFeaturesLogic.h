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

// .NAME vtkSlicerSurfFeaturesLogic - slicer logic class for volumes manipulation
// .SECTION Description
// This class manages the logic associated with reading, saving,
// and changing propertied of the volumes


#ifndef __vtkSlicerSurfFeaturesLogic_h
#define __vtkSlicerSurfFeaturesLogic_h

// Slicer includes
#include "vtkSlicerModuleLogic.h"

// MRML includes
#include "vtkMRMLScalarVolumeNode.h"
#include "vtkMRMLLinearTransformNode.h"

// STD includes
#include <cstdlib>
#include <fstream>
#include <ctime>

// OpenCV includes
#include "opencv2/core/core.hpp"
#include <opencv2/nonfree/features2d.hpp>

// Qt includes
#include <QTextEdit>
#include "vtkSlicerSurfFeaturesModuleLogicExport.h"

// vtk includes
#include <vtkMatrix4x4.h>
#include <vtkImageData.h>

#include "util_macros.h"

// vnl includes
#include <vnl/vnl_double_3.h>
#include <vnl/vnl_float_3.h>
#include <vnl/vnl_matrix.h>

using namespace std;
using namespace cv;


/// \ingroup Slicer_QtModules_ExtensionTemplate
class VTK_SLICER_SURFFEATURES_MODULE_LOGIC_EXPORT vtkSlicerSurfFeaturesLogic :
  public vtkSlicerModuleLogic
{
public:

  static vtkSlicerSurfFeaturesLogic *New();
  vtkTypeMacro(vtkSlicerSurfFeaturesLogic, vtkSlicerModuleLogic);
  void PrintSelf(ostream& os, vtkIndent indent);


  

  

protected:
  vtkSlicerSurfFeaturesLogic();
  virtual ~vtkSlicerSurfFeaturesLogic();

  // Callbacks reimplemented from vtkMRMLAbstractLogic
  virtual void SetMRMLSceneInternal(vtkMRMLScene* newScene);
  virtual void RegisterNodes();
  virtual void UpdateFromMRMLScene();
  virtual void OnMRMLSceneNodeAdded(vtkMRMLNode* node);
  virtual void OnMRMLSceneNodeRemoved(vtkMRMLNode* node);
private:

  vtkSlicerSurfFeaturesLogic(const vtkSlicerSurfFeaturesLogic&); // Not implemented
  void operator=(const vtkSlicerSurfFeaturesLogic&);               // Not implemented


  // ============================================
  // Helper Functions
  // ============================================

  // Crop data (either opencv or vtk image)
  vtkImageData* cropData(vtkImageData* data);
  void cropData(cv::Mat& img);

  // Conversions from opencv to vtk
  cv::Mat convertImage(vtkImageData* data);
  
  // Check if tracking info for an image is invalid
  bool isTracked(vtkMRMLNode* node);

  // Draw features into the image
  cv::Mat drawFeatures(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints);
  void drawQueryMatches();
  void drawBestTrainMatches();

  // Other
  void computeCentroid(const cv::Mat& mask, int& x, int& y);
  bool cropRatiosValid();

  // =============================================
  // Console
  // =============================================
  void stopWatchWrite(std::ostringstream& oss);
  void resetConsoleFont();

  // =============================================
  // Visualization
  // =============================================

  // Public Interface to visualization
public:
  void showCurrentImage();
  void previousImage();
  void nextImage();
  void updateQueryNode();
  void updateMatchNode();
  void updateMatchNodeRansac();
  void showCropFirstImage();
  void saveCurrentImage();

private:
  void updateImage();
  void showImage(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints);
  void showImage(vtkMRMLNode* node);
  

  // =============================================
  // Opencv computations
  // =============================================
  
  // Public Interface
public:
  void play();
  void computeBogus();
  void computeTrain();
  void computeQuery();
  int computeNext(vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, vector<Mat>& descriptors, cv::Ptr<cv::DescriptorMatcher> descriptorMatcher);
  bool isCorrespondenceComputing();
  bool isQueryLoading();
  bool isTrainLoading();
  bool isBogusLoading();
  bool isLoading();
  void computeNextQuery();
  void computeNextTrain();
  void computeNextBogus();
  void computeInterSliceCorrespondence();
  void computeNextInterSliceCorrespondence();

  // Private helpers
private:
  void computeKeypointsAndDescriptors(const cv::Mat& data,\
                                      std::vector<cv::KeyPoint>& keypoints,\
                                      cv::Mat& descriptors);
  void readAndComputeFeaturesOnMhaFile(const std::string& file,\
                                       std::vector<cv::Mat>& images,\
                                       std::vector<std::string>& imagesNames,\
                                       std::vector<std::vector<float> >& imagesTransform,\
                                       std::vector<bool>& transformsValidity,\
                                       std::vector<std::vector<cv::KeyPoint> >& keypoints,\
                                       std::vector<cv::Mat>& descriptors,\
                                       cv::Ptr<cv::DescriptorMatcher> descriptorMatcher,\
                                       int& startFrame, int& stopFrame, std::string who);

  // ================================================
  // Other
  // ================================================
  public:
    // log matches
    void writeMatches();

  private:
    int ransac(const std::vector<vnl_double_3>& points, std::vector<int>& inliersIdx, std::vector<int>& planePointsIdx);

  
private:
  // ============================================
  // Attributes
  // ============================================
  // Node
  vtkMRMLScalarVolumeNode* node;
  vtkMRMLScalarVolumeNode* queryNode;
  vtkMRMLScalarVolumeNode* matchNode;

  // Clock
  clock_t initTime;
  clock_t lastStopWatch;

  // State
  int currentImgIndex;

  // train images, keypoints, descriptors and matcher
  std::vector<cv::Mat> trainImages;
  std::vector<std::string> trainImagesNames;
  std::vector< std::vector<float> > trainImagesTransform;
  std::vector<std::vector<float> > queryTransformEstimate;
  std::vector<std::vector<cv::KeyPoint> > trainKeypoints;
  std::vector<cv::Mat> trainDescriptors;
  std::vector<bool> trainTransformsValidity;
  cv::Ptr<cv::DescriptorMatcher> trainDescriptorMatcher;
  

  // bogus images, keypoints, descriptors and matcher
  std::vector<cv::Mat> bogusImages;
  std::vector<std::string> bogusImagesNames;
  std::vector< std::vector<float> > bogusImagesTransform;
  std::vector<std::vector<cv::KeyPoint> > bogusKeypoints;
  std::vector<cv::Mat> bogusDescriptors;
  std::vector<bool> bogusTransformsValidity;
  cv::Ptr<cv::DescriptorMatcher> bogusDescriptorMatcher;

  // query images, keypoints, descriptors and matcher
  std::vector<cv::Mat> queryImages;
  std::vector<std::string> queryImagesNames;
  std::vector< std::vector<float> > queryImagesTransform;
  std::vector<std::vector<cv::KeyPoint> > queryKeypoints;
  std::vector<cv::Mat> queryDescriptors;
  std::vector<bool> queryTransformsValidity;
  cv::Ptr<cv::DescriptorMatcher> queryDescriptorMatcher;

  cv::Mat queryImageWithFeatures;
  cv::Mat trainImageWithFeatures;
  cv::Mat firstImage;
  cv::Mat firstImageCropped;

  vtkSmartPointer<vtkImageData> queryImageData;
  vtkSmartPointer<vtkImageData> matchImageData;

  cv::Mat mask;
  cv::Mat croppedMask;

  

  // Best matches are recorded (when calculating correspondences
  std::vector<int> bestMatches;
  std::vector<int> bestMatchesCount;

  // Keep matches between selected train image and query images (when calculating correspondences)
  // [# query images]*[# kept matches]
  std::vector<std::vector<DMatch> > matchesWithBestTrainImage;
  // [# query images] * [# kept matches]
  std::vector<std::vector<DMatch> > afterHoughMatches;
  

  // Console
  QTextEdit* console;

  // Progress
  int queryProgress;
  int trainProgress;
  int bogusProgress;
  int correspondenceProgress;
  
  // ???
  int playDirection;
  int xcentroid;
  int ycentroid;

  // Parameters
  int minHessian;
  std::string matcherType;
  std::string trainFile;
  std::string bogusFile;
  std::string queryFile;
  int trainStartFrame;
  int trainStopFrame;
  int bogusStartFrame;
  int bogusStopFrame;
  int queryStartFrame;
  int queryStopFrame;
  float cropRatios[4];
  vtkSmartPointer<vtkMatrix4x4> ImageToProbeTransform;
  

public:
  // ======================================
  // Attribute setters and getters
  // ======================================
  void setMinHessian(int);
  int getMinHessian();

  // parameter setters and getters
  GET(QTextEdit*,console,Console);
  void setConsole(QTextEdit* console);

  void setBogusFile(std::string);
  void setTrainFile(std::string);
  void setQueryFile(std::string);
  GET(std::string, queryFile, QueryFile);
  GET(std::string, trainFile, TrainFile);
  GET(std::string, bogusFile, BogusFile);
  
  GETSET(int, trainStartFrame, TrainStartFrame);
  GETSET(int, trainStopFrame, TrainStopFrame);
  GETSET(int, bogusStartFrame, BogusStartFrame);
  GETSET(int, bogusStopFrame, BogusStopFrame);
  GETSET(int,queryStartFrame,QueryStartFrame);
  GETSET(int,queryStopFrame,QueryStopFrame);

  GET(int, queryProgress, QueryProgress);
  GET(int, trainProgress, TrainProgress);
  GET(int, bogusProgress, BogusProgress);
  GET(int, correspondenceProgress, CorrespondenceProgress);
  void setQueryProgress(int p);
  void setTrainProgress(int p);
  void setBogusProgress(int p);
  void setCorrespondenceProgress(int p);

  GETSET(double, cropRatios[0], LeftCrop);
  GETSET(double, cropRatios[1], RightCrop);
  GETSET(double, cropRatios[2], TopCrop);
  GETSET(double, cropRatios[3], BottomCrop);
};



#endif
