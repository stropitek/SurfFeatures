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
#include <vtkImageData.h>
#include <vtkImageExport.h>
#include <vtkImageImport.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>

// STD includes
#include <cassert>
#include <ctime>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <math.h>

// MRML includes
#include <vtkMRMLVolumeNode.h>
#include <vtkMRMLScalarVolumeNode.h>
#include <vtkMRMLAnnotationFiducialNode.h>


// OpenCV includes
#include <cv.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/imgproc/imgproc_c.h>
#include "opencv2/nonfree/gpu.hpp"

// vnl includes
#include <vnl/vnl_double_3.h>

using namespace std;
using namespace cv;


#ifdef WIN32
  #include <time.h>
  #include <sys\timeb.h>
#endif

// ===========================================
// Hough Transform
// ===========================================

class HoughCodeSimilarity
{
public:

  HoughCodeSimilarity(
    )
  {
    m_fAngleToBooth = -1;
    m_fDistanceToBooth = -1;
    m_fAngleDiff = -1;
    m_fScaleDiff = -1;
  }

  ~HoughCodeSimilarity(
    )
  {
  }


  int
    SetHoughCode(
    const float &fPollRow,	// Poll booth row
    const float &fPollCol,	// Poll booth col
    const float &fPollOri,	// Poll booth orientation
    const float &fPollScl	// Poll booth scale
    )
  {
    // Set distance to poll booth
    float fDiffRow = fPollRow - m_Key_fRow;
    float fDiffCol = fPollCol - m_Key_fCol;
    float fDistSqr = fDiffCol*fDiffCol + fDiffRow*fDiffRow;
    m_fDistanceToBooth = (float)sqrt( fDistSqr );

    // Set angle between feature orientation and poll booth
    float fAngleToBooth = (float)atan2( fDiffRow, fDiffCol );
    m_fAngleToBooth = fAngleToBooth;

    // Set angular and orientation differences
    m_fAngleDiff = fPollOri - m_Key_fOri;
    m_fScaleDiff = fPollScl / m_Key_fScale;

    return 0;
  }

  int
    FindPollBooth(
    const float &fRow,
    const float &fCol,
    const float &fOri,
    const float &fScale,
    float &fPollRow,		// Return row value
    float &fPollCol,		// Return col value,
    float &fPollOri,		// Return orientation value
    float &fPollScl,		// Return scale value
    int   bMirror	= 0// Match is a mirror of this feature
    ) const
  {
    if( !bMirror )
    {
      float fOriDiff = fOri - m_Key_fOri;
      float fScaleDiff = fScale / m_Key_fScale;

      float fDist = m_fDistanceToBooth*fScaleDiff;
      float fAngle = m_fAngleToBooth + fOriDiff;

      fPollRow = (float)(sin( fAngle )*fDist + fRow);
      fPollCol = (float)(cos( fAngle )*fDist + fCol);

      fPollOri = fOri + m_fAngleDiff;
      fPollScl = m_fScaleDiff*fScale;
    }
    else
    {
      // Here, we consider a mirrored poll both
      float fOriDiff = fOri - (-m_Key_fOri);
      float fScaleDiff = fScale / m_Key_fScale;

      float fDist = m_fDistanceToBooth*fScaleDiff;
      float fAngle = (-m_fAngleToBooth) + fOriDiff;

      fPollRow = (float)(sin( fAngle )*fDist + fRow);
      fPollCol = (float)(cos( fAngle )*fDist + fCol);

      fPollOri = fOri - m_fAngleDiff;
      fPollScl = m_fScaleDiff*fScale;
    }

    return 0;
  }



  float m_fAngleToBooth;	// With just this, we can draw a line through to the booth
  float m_fDistanceToBooth;	// With just this, we can draw a circle through booth

  float m_fAngleDiff;		// Poll booth angle - key angle
  float m_fScaleDiff;		// Poll booth scale / key scale

  float m_Key_fRow;
  float m_Key_fCol;
  float m_Key_fOri;
  float m_Key_fScale;

};

class KeypointGeometry
{
public:
  KeypointGeometry(){};
  ~KeypointGeometry(){};

  int iIndex0;
  int iIndex1;

  float m_Key_fRow;
  float m_Key_fCol;
  float m_Key_fOri;
  float m_Key_fScale;	
};



#define PI 3.141592653589793
float
  angle_radians(
  float fAngle )
{
  return (2.0f*PI*fAngle) / 180.0f;
}

#define SCALE_DIFF log((float)1.5f)
#define ORIENTATION_DIFF (20.0f*PI/180.0F)
#define TRANSLATION_ERROR (30.0f/43.0f)

int
  compatible_poll_booths_line_segment(
  float fTrainingCol,
  float fTrainingRow,
  float fTrainingOri,
  float fTrainingScl,

  float fPollCol,
  float fPollRow,
  float fPollOri,
  float fPollScl,

  float fTranslationErrorThres = TRANSLATION_ERROR,
  float fScaleErrorThres = SCALE_DIFF,
  float fOrientationErrorThres = ORIENTATION_DIFF
  )
{
  float fLogTrainingPollScl = log( fTrainingScl );
  float fLogPollScl = log( fPollScl );
  float fLogSclDiff = fabs( fLogTrainingPollScl - fLogPollScl );

  float fDiffCol = fPollCol - fTrainingCol;
  float fDiffRow = fPollRow - fTrainingRow;
  float fDistDiff = sqrt( fDiffCol*fDiffCol + fDiffRow*fDiffRow );

  float fOriDiff = fPollOri > fTrainingOri ?
    fPollOri - fTrainingOri : fTrainingOri - fPollOri;
  assert( fOriDiff >= 0 && fOriDiff < 2*PI );
  if( fOriDiff > (2*PI - fOriDiff) )
  {
    fOriDiff = (2*PI - fOriDiff);
  }
  assert( fOriDiff >= 0 && fOriDiff < 2*PI );

  // If the best match and this feature are within the match
  // hypothesis

  //if(		fDistDiff	< TRANSLATION_ERROR*fTrainingScl
  //	&&	fLogSclDiff	< SCALE_DIFF
  //	&&	fOriDiff	< ORIENTATION_DIFF
  //	)
  if(		fDistDiff	< fTranslationErrorThres*fTrainingScl
    &&	fLogSclDiff	< fScaleErrorThres
    &&	fOriDiff	< fOrientationErrorThres
    )
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

int
  houghTransform(
  const vector<KeyPoint>& queryKeypoints,
  const vector< vector<KeyPoint> > & trainKeypoints,
  vector<DMatch> & matches,
  int refx, int refy
  )
{
  float fRefX;
  float fRefY;

  vector< KeypointGeometry > vecKG;
  //vecKG.resize( mmatches.size()*mmatches[0].size() );

  for( int i = 0; i < matches.size(); i++ )
  {
    if( matches[i].trainIdx < 0 || matches[i].imgIdx < 0 )
    {
      continue;
    }
    int iQIndex = matches[i].queryIdx;
    int iTIndex = matches[i].trainIdx;
    int iIIndex = matches[i].imgIdx;
    const KeyPoint &keyQ = queryKeypoints[iQIndex];
    const KeyPoint &keyT = trainKeypoints[iIIndex][iTIndex];

    HoughCodeSimilarity hc;

    hc.m_Key_fRow = keyQ.pt.y;
    hc.m_Key_fCol = keyQ.pt.x;
    hc.m_Key_fOri = angle_radians(keyQ.angle);
    hc.m_Key_fScale = keyQ.size;

    KeypointGeometry kg;

    hc.SetHoughCode( refy, refx, 0, 10 );
    hc.FindPollBooth( keyT.pt.y, keyT.pt.x, angle_radians(keyT.angle), keyT.size, kg.m_Key_fRow, kg.m_Key_fCol, kg.m_Key_fOri, kg.m_Key_fScale );
    kg.iIndex0 = i;
    vecKG.push_back( kg );
  }

  int iMaxIndex;
  int iMaxCount = 0;
  for( int i = 0; i < vecKG.size(); i++ )
  {
    KeypointGeometry &kg1 = vecKG[i];
    int iCount = 0;
    for( int j = 0; j < vecKG.size(); j++ )
    {
      KeypointGeometry &kg2 = vecKG[j];
      if( compatible_poll_booths_line_segment(
        kg1.m_Key_fCol,
        kg1.m_Key_fRow,
        angle_radians(kg1.m_Key_fOri),
        kg1.m_Key_fScale,
        kg2.m_Key_fCol,
        kg2.m_Key_fRow,
        angle_radians(kg2.m_Key_fOri),
        kg2.m_Key_fScale ) )
      {
        iCount++;
      }
    }
    if( iCount > iMaxCount )
    {
      iMaxIndex = i;
      iMaxCount = iCount;
    }
  }

  for( int i = 0; i < matches.size(); i++ )
  {
    KeypointGeometry &kg1 = vecKG[iMaxIndex];
    for( int j = 0; j < vecKG.size(); j++ )
    {
      KeypointGeometry &kg2 = vecKG[j];
      if( compatible_poll_booths_line_segment(
        kg1.m_Key_fCol,
        kg1.m_Key_fRow,
        angle_radians(kg1.m_Key_fOri),
        kg1.m_Key_fScale,
        kg2.m_Key_fCol,
        kg2.m_Key_fRow,
        angle_radians(kg2.m_Key_fOri),
        kg2.m_Key_fScale ) )
      {
      }
      else
      {
        matches[j].imgIdx = -1;
        matches[j].queryIdx = -1;
        matches[j].trainIdx = -1;
      }
    }
  }

  return iMaxCount;
}

//
// houghTransform()
//
// Call with one vector of keypoints.
//
int
  houghTransform(
  const vector<KeyPoint>& queryKeypoints,
  const vector<KeyPoint> & trainKeypoints,
  vector<DMatch> & matches,
  int refx, int refy
  )
{
  vector< vector<KeyPoint> > vvtrainKeypoints;
  vvtrainKeypoints.push_back( trainKeypoints );
  return houghTransform( queryKeypoints, vvtrainKeypoints, matches, refx, refy );
}

//
// houghTransform()
//
// Call with a double vector of matches.
//
int
  houghTransform(
  const vector<KeyPoint>& queryKeypoints,
  const vector< vector<KeyPoint> > & trainKeypoints,
  vector< vector<DMatch> > & matches,
  int refx, int refy
  )
{
  vector<DMatch> vvMatches;

  for( int i = 0; i < matches.size(); i++ )
  {
    for( int j = 0; j < matches[i].size(); j++ )
    {
      if( matches[i][j].trainIdx >= 0 )
        vvMatches.push_back( matches[i][j] );
    }
  }

  return houghTransform( queryKeypoints, trainKeypoints, vvMatches, refx, refy );
}


// ===========================================
// Open cv matching
// ===========================================

std::string getDir(const std::string& filename)
{
  #ifdef WIN32
  const char dlmtr = '\\';
  #else
  const char dlmtr = '/';
  #endif

  std::string dirName;
  size_t pos = filename.rfind(dlmtr);
  dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;
  return dirName;
}


void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{

  trainFilenames.clear();

  ifstream file( filename.c_str() );
  if ( !file.is_open() )
    return;

  dirName = getDir(filename);
  while( !file.eof() )
  {
    string str; getline( file, str );
    if( str.empty() ) break;
    trainFilenames.push_back(str);
  }
  file.close();
}

int readImageDimensions_mha(const std::string& filename, int& cols, int& rows, int& count)
{
  ifstream file( filename.c_str() );
  if ( !file.is_open() )
    return 1;
    
  // Read until get dimensions
  while( !file.eof() )
  {
    string str; getline( file, str );
    if( str.empty() ) break;
    char *pch = &(str[0]);
    if( !pch )
    {
      return 1;
      file.close();
    }

    if( strstr( pch, "DimSize =" ) )
    {
      if( sscanf( pch, "DimSize = %d %d %d", &cols, &rows, &count ) != 3 )
      {
        printf( "Error: could not read dimensions\n" );
        file.close();
        return 1;
      }
      file.close();
      return 0;
    }
  }
  file.close();
  return 1;
}

void readImageTransforms_mha(const std::string& filename, std::vector<std::vector<float> >& transforms, std::vector<bool>& transformsValidity, std::vector<std::string>& filenames)
{
  std::string dirName = getDir(filename);
  filenames.clear();
  transforms.clear();
  
  // Vector for reading in transforms
  vector< float > vfTrans;
  vfTrans.resize(12);
  std::string pngFilename;

  ifstream file( filename.c_str() );
  if ( !file.is_open() )
    return;

  while( !file.eof() )
  {
    string str; getline( file, str );
    if( str.empty() ) break;
    char *pch = &(str[0]);
    if( !pch )
      return;

    if( strstr( pch, "ProbeToTrackerTransform =" )
      || strstr( pch, "UltrasoundToTrackerTransform =" ) )
    {
       // Parse name and transform
       // Seq_Frame0000_ProbeToTrackerTransform = -0.224009 -0.529064 0.818481 212.75 0.52031 0.6452 0.559459 -14.0417 -0.824074 0.551188 0.130746 -26.1193 0 0 0 1 

      char *pcName = pch;
      char *pcTrans = strstr( pch, "=" );
      pcTrans[-1] = 0; // End file name string pcName
       //pcTrans++; // Increment to just after equal sign

      pngFilename = dirName + pcName + ".png";// + pcTrans;

      char *pch = pcTrans;

      for( int j =0; j < 12; j++ )
      {
        pch = strchr( pch + 1, ' ' );
        if( !pch )
          return;
        vfTrans[j] = atof( pch );
        pch++;
      }
      transforms.push_back( vfTrans );
      filenames.push_back(pngFilename);
    }
    else if(strstr(pch, "UltrasoundToTrackerTransformStatus") || strstr(pch, "ProbeToTrackerTransformStatus")) {
      if(strstr(pch, "OK")){
        transformsValidity.push_back(true);
      }
      else if(strstr(pch, "INVALID"))
        transformsValidity.push_back(false);
    }
    if( strstr( pch, "ElementDataFile = LOCAL" ) )
    {
       // Done reading
      break;
    }
  }
}

void readImages_mha(const std::string& filename, std::vector<cv::Mat>& images, int& firstFrame, int& lastFrame, const std::vector<bool>& transformsValidity)
{
  int iImgCols = -1;
  int iImgRows = -1;
  int iImgCount = -1;
  if(readImageDimensions_mha(filename, iImgCols, iImgRows, iImgCount))
    return;
  
  FILE *infile = fopen( filename.c_str(), "rb" );
   char buffer[400];
   while( fgets( buffer, 400, infile ) )
   {
     if( strstr( buffer, "ElementDataFile = LOCAL" ) )
     {
       // Done reading
       break;
     }
   }

   unsigned char *pucImgData = new unsigned char[iImgRows*iImgCols];
   Mat mtImg(iImgRows, iImgCols, CV_8UC1, pucImgData);

   if(firstFrame < 0)
     firstFrame = 0;
   if(firstFrame >= iImgCount)
     firstFrame = iImgCount-1;
   if(lastFrame < 0 || lastFrame >= iImgCount)
     lastFrame = iImgCount-1;
   if(firstFrame > lastFrame)
     firstFrame = lastFrame;
   // Read & write images
   void *ptr = NULL;
   for( int i = 0; i < iImgCount; i++ )
   {
     if(i<firstFrame)
       fseek(infile, iImgRows*iImgCols, SEEK_CUR);
     else if(i>=firstFrame && i <= lastFrame && transformsValidity[i]) {
       Mat mtImgNew = mtImg.clone();
       fread( mtImgNew.data, 1, iImgRows*iImgCols, infile );
       images.push_back( mtImgNew );
     }
     else if(i>lastFrame)
       break;
   }
   delete [] pucImgData;
   
   fclose( infile );
}

// Read in relevant lines from mha file, used with PLUS ultrasound data
int read_mha(
  const string& filename,
  vector <Mat>& images,
  vector<string>& filenames,
  vector< vector<float> >& transforms,
  vector<bool>& transformsValidity,
  int& firstFrame,
  int& lastFrame
  )
{

  readImageTransforms_mha(filename, transforms, transformsValidity, filenames);
  readImages_mha(filename, images, firstFrame, lastFrame, transformsValidity);
    
  // Keep the transforms that are in the first-last range
  std::vector<std::vector<float> > allTransforms = transforms;
  std::vector<std::string> allFilenames = filenames;
  transforms.clear();
  filenames.clear();
  
  for(int i=0; i<transformsValidity.size(); i++) {
    if(i<firstFrame || i>lastFrame  || !transformsValidity[i])
      continue;
    transforms.push_back(allTransforms[i]);
    filenames.push_back(allFilenames[i]);
  }
  transformsValidity = std::vector<bool>(transformsValidity.begin()+firstFrame, transformsValidity.begin()+lastFrame+1);

  std::cout << "Read images: " << images.size() << std::endl;
  std::cout << "Read transforms: " << transforms.size() << std::endl;
  std::cout << "Read transforms validity: " << transformsValidity.size() << std::endl;
  std::cout << "Read names: " << filenames.size() << std::endl;
  
  return 0;

}


void matchDescriptorsKNN( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, Ptr<DescriptorMatcher>& descriptorMatcher, int k)
{
  // Assumes training descriptors have already been added to descriptorMatcher
  cout << "< Set train descriptors collection in the matcher and match query descriptors to them..." << endl;
    //descriptorMatcher->add( trainDescriptors );
  descriptorMatcher->knnMatch( queryDescriptors, matches, k );
  CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );
  cout << ">" << endl;
}


// TODO: make functions called often inline


// ==============================================
// Geometry Functions
// ==============================================
vnl_matrix<double> getRotationMatrix(vnl_double_3 & axis, double theta)
{
  vnl_matrix<double> result(3,3);
  result(0,0) = cos(theta)+axis[0]*axis[0]*(1-cos(theta));
  result(0,1) = axis[0]*axis[1]*(1-cos(theta))-axis[2]*sin(theta);
  result(0,2) = axis[0]*axis[2]*(1-cos(theta))+axis[1]*sin(theta);
  result(1,0) = axis[1]*axis[0]*(1-cos(theta))+axis[2]*sin(theta);
  result(1,1) = cos(theta)+axis[1]*axis[1]*(1-cos(theta));
  result(1,2) = axis[1]*axis[2]*(1-cos(theta))-axis[0]*sin(theta);
  result(2,0) = axis[2]*axis[0]*(1-cos(theta))-axis[1]*sin(theta);
  result(2,1) = axis[2]*axis[1]*(1-cos(theta))+axis[0]*sin(theta);
  result(2,2) = cos(theta)+axis[2]*axis[2]*(1-cos(theta));
  result.normalize_columns();
  return result;
}

vnl_matrix<double> convertVnlVectorToMatrix(const vnl_double_3& v)
{
  vnl_matrix<double> result(3,1);
  result(0,0) = v[0];
  result(1,0) = v[1];
  result(2,0) = v[2];
  return result;
}

vnl_double_3 convertVnlMatrixToVector(const vnl_matrix<double>& m)
{
  vnl_double_3 result;
  if(m.rows()==1 && m.cols()==3) {
    for(int i=0; i<3; i++)
      result[i] = m(0,i);
  }
  else if(m.rows()==3 && m.cols()==1) {
    for(int i=0; i<3; i++)
      result[i] = m(i,0);
  }
  return result;
}

vnl_double_3 projectPoint(vnl_double_3 point, vnl_double_3 normalToPlane, double offset)
{
  normalToPlane.normalize();
  double dist = dot_product(point, normalToPlane) + offset;
  vnl_double_3 vec = dist*normalToPlane;
  return point - vec;
}

double getAngle(const vnl_double_3& u, const vnl_double_3& v)
{
  return acos(dot_product(u,v)/(u.two_norm()*v.two_norm()));
}

void getAxisAndRotationAngle(vnl_double_3 v, vnl_double_3 v_new, vnl_double_3& axis, double& angle)
{
  axis = vnl_cross_3d(v, v_new);
  angle = getAngle(v,v_new);
}

double computeMeanDistance(const std::vector<vnl_double_3>& x1, const std::vector<vnl_double_3>& x2)
{
  if(x1.size() != x2.size())
    return DBL_MAX;
  double result = 0.;
  for(int i=0; i<x1.size(); i++)
  {
    vnl_double_3 diff = x1[i]-x2[i];
    result += diff.two_norm();
  }
  result /= x1.size();
  return result;
}

std::vector<float> vtkToStdMatrix(vtkMatrix4x4* matrix)
{
  std::vector<float> result;
  for(int i=0; i<3; i++)
  {
    for(int j=0; j<4; j++)
      result.push_back(matrix->GetElement(i,j));
  }

  return result;
}

void vnlToVtkMatrix(const vnl_matrix<double> vnlMatrix , vtkMatrix4x4* vtkMatrix)
{
  vtkMatrix->Identity();
  int rows = vnlMatrix.rows();
  int cols = vnlMatrix.cols();
  if(rows > 4)
    rows = 4;
  if(cols > 4)
    cols = 4;
  for(int i=0; i<rows; i++)
  {
    for(int j=0; j<cols; j++)
      vtkMatrix->SetElement(i,j,vnlMatrix(i,j));
  }

}

//----------------------------------------------------------------------------
vtkStandardNewMacro(vtkSlicerSurfFeaturesLogic);

//----------------------------------------------------------------------------
vtkSlicerSurfFeaturesLogic::vtkSlicerSurfFeaturesLogic()
{
  this->node = NULL;
  this->lastStopWatch = clock();
  this->initTime = clock();
  this->ransacMargin = 1.0;
  this->minHessian = 400;
  this->matcherType = "FlannBased";
  this->trainDescriptorMatcher = DescriptorMatcher::create(matcherType);
  this->bogusDescriptorMatcher = DescriptorMatcher::create(matcherType);
  this->queryDescriptorMatcher = DescriptorMatcher::create(matcherType);
  this->cropRatios[0] = 1/5.; this->cropRatios[1]= 3./3.9;
  this->cropRatios[2] = 1/6.; this->cropRatios[3] = 3./4.;

  #ifdef WIN32
  this->setBogusFile("C:\\Users\\DanK\\MProject\\data\\US\\Plus_test\\TrackedImageSequence_20121210_162606.mha");
  this->setTrainFile("C:\\Users\\DanK\\MProject\\data\\US\\decemberUS\\TrackedImageSequence_20121211_095535.mha");
  this->setQueryFile("C:\\Users\\DanK\\MProject\\data\\US\\decemberUS\\TrackedImageSequence_20121211_095535.mha");
  #else
  this->setBogusFile("/Users/dkostro/Projects/MProject/data/US/Plus_test/TrackedImageSequence_20121210_162606.mha");
  this->setTrainFile("/Users/dkostro/Projects/MProject/data/US/decemberUS/TrackedImageSequence_20121211_095535.mha");
  this->setQueryFile("/Users/dkostro/Projects/MProject/data/US/decemberUS/TrackedImageSequence_20121211_095535.mha");
  #endif

  this->bogusStartFrame = 0;
  this->bogusStopFrame = 2;
  this->trainStartFrame = 0;
  this->trainStopFrame = 2;
  this->queryStartFrame = 3;
  this->queryStopFrame = 5;
  this->currentImgIndex = 0;


  // Progress is 0...
  this->queryProgress = 0;
  this->trainProgress = 0;
  this->bogusProgress = 0;
  this->correspondenceProgress = 0;
  
  this->playDirection = 1;

  // Load mask
  #ifdef WIN32
  this->mask = cv::imread("C:\\Users\\DanK\\MProject\\data\\US\\SlicerSaved\\mask.bmp",CV_LOAD_IMAGE_GRAYSCALE);
  #else
  this->mask = cv::imread("/Users/dkostro/Projects/MProject/data/US/masks/mask.bmp", CV_LOAD_IMAGE_GRAYSCALE);
  #endif

  // Initialize Image to Probe transform
  this->ImageToProbeTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  this->ImageToProbeTransform->Identity();
  this->ImageToProbeTransform->SetElement(0,0,0.107535);
  this->ImageToProbeTransform->SetElement(0,1,0.00094824);
  this->ImageToProbeTransform->SetElement(0,2,0.0044213);
  this->ImageToProbeTransform->SetElement(0,3,-65.9013);
  this->ImageToProbeTransform->SetElement(1,0,0.0044901);
  this->ImageToProbeTransform->SetElement(1,1,-0.00238041);
  this->ImageToProbeTransform->SetElement(1,2,-0.106347);
  this->ImageToProbeTransform->SetElement(1,3,-3.05698);
  this->ImageToProbeTransform->SetElement(2,0,-0.000844189);
  this->ImageToProbeTransform->SetElement(2,1,0.105271);
  this->ImageToProbeTransform->SetElement(2,2,-0.00244457);
  this->ImageToProbeTransform->SetElement(2,3,-17.1613);


  this->queryNode = vtkMRMLScalarVolumeNode::New();
  this->queryNode->SetName("query node");
  this->matchNode = vtkMRMLScalarVolumeNode::New();
  this->matchNode->SetName("match node");

  this->Modified();
}

void vtkSlicerSurfFeaturesLogic::setQueryProgress(int p){
  if(p != this->queryProgress)
  {
    this->queryProgress = p;
    this->Modified();
  }
}

void vtkSlicerSurfFeaturesLogic::setTrainProgress(int p){
  if(p != this->trainProgress)
  {
    this->trainProgress = p;
    this->Modified();
  }
}

void vtkSlicerSurfFeaturesLogic::setBogusProgress(int p){
  if(p != this->bogusProgress)
  {
    this->bogusProgress = p;
    this->Modified();
  }
}

void vtkSlicerSurfFeaturesLogic::setCorrespondenceProgress(int p){
  if(p != this->correspondenceProgress)
  {
    this->correspondenceProgress = p;
    this->Modified();
  }
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


vtkImageData* vtkSlicerSurfFeaturesLogic::cropData(vtkImageData* data)
{
  int dims[3];
  data->GetDimensions(dims);
  vtkSmartPointer<vtkExtractVOI> extractVOI = vtkSmartPointer<vtkExtractVOI>::New();
  extractVOI->SetInput(data);
  extractVOI->SetVOI(dims[0]*this->cropRatios[0],dims[0]*this->cropRatios[1],dims[1]*this->cropRatios[2],dims[1]*this->cropRatios[3], 0, 0);
  extractVOI->SetSampleRate(1,1,1);
  extractVOI->Update();
  return extractVOI->GetOutput();
}

void vtkSlicerSurfFeaturesLogic::cropData(cv::Mat& img)
{
  cv::Size size = img.size();
  cv::Rect roi(size.width*this->cropRatios[0], size.height*this->cropRatios[2],\
    size.width*(this->cropRatios[1]-this->cropRatios[0]), size.height*(this->cropRatios[3]-this->cropRatios[2]));
  // Clone the image to physically crop the image (not just jumping on same pointer)
  img = img(roi).clone();
}

cv::Mat vtkSlicerSurfFeaturesLogic::convertImage(vtkImageData* image)
{
  int dims[3];
  image->GetDimensions(dims);
  /*vtkSmartPointer<vtkImageExport> exporter = vtkSmartPointer<vtkImageExport>::New();
  exporter->SetInput(data);
  int numel = dims[0]*dims[1]*dims[2];
  void* void_ptr = malloc(numel);
  exporter->SetExportVoidPointer(void_ptr);
  exporter->Export();*/

  // TODO: Make sure the pointer is unsigned char... or use the proper opencv type...
    cv::Mat ocvImage(dims[1],dims[0],CV_8U ,image->GetScalarPointer());
  // Clone the image so you don't share the pointer with vtkImageData
  return ocvImage.clone();
}



void vtkSlicerSurfFeaturesLogic::updateQueryNode()
{
  // Stop if don't have loaded any query image yet
  if(this->queryProgress != 100)
    return;
  vtkSmartPointer<vtkMatrix4x4> transform = vtkSmartPointer<vtkMatrix4x4>::New();
  transform->Identity();
  for(int i=0; i<3; i++)
    for(int j=0; j<4; j++)
    transform->SetElement(i,j,this->queryImagesTransform[this->currentImgIndex][i*4+j]);
  vtkSmartPointer<vtkTransform> combinedTransform = vtkSmartPointer<vtkTransform>::New();
  combinedTransform->Concatenate(transform);
  combinedTransform->Concatenate(this->ImageToProbeTransform);
  vtkSmartPointer<vtkMatrix4x4> matrix = combinedTransform->GetMatrix();


  const cv::Mat& image = this->queryImageWithFeatures;
  int width = image.cols;
  int height = image.rows;
  vtkSmartPointer<vtkImageImport> importer = vtkSmartPointer<vtkImageImport>::New();
  // TODO: check type of opencv image data first...
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetImportVoidPointer(image.data,1); // Save argument to 1 won't destroy the pointer when importer destroyed
  importer->SetWholeExtent(0,width-1,0, height-1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  importer->Update();
  // You've got to keep a reference of that vtkimage data somewhere, because the node does not make a deep copy in SetAndObserveImageData
  this->queryImageData = importer->GetOutput();

  // vtkImage shares pointer with opencv image
  this->queryNode->SetAndObserveImageData(this->queryImageData);
  this->queryNode->SetIJKToRASMatrix(matrix);

  if(!this->GetMRMLScene()->IsNodePresent(this->queryNode))
    this->GetMRMLScene()->AddNode(this->queryNode);

}

void vtkSlicerSurfFeaturesLogic::updateMatchNode()
{
  // Do nothing if we did not calculate correspondences yet
  if(this->correspondenceProgress != 100)
    return;
  int trainIndex = this->bestMatches[this->currentImgIndex];
  vtkSmartPointer<vtkMatrix4x4> transform = vtkSmartPointer<vtkMatrix4x4>::New();
  transform->Identity();
  for(int i=0; i<3; i++)
    for(int j=0; j<4; j++)
    transform->SetElement(i,j,this->trainImagesTransform[trainIndex][i*4+j]);
  vtkSmartPointer<vtkTransform> combinedTransform = vtkSmartPointer<vtkTransform>::New();
  combinedTransform->Concatenate(transform);
  combinedTransform->Concatenate(this->ImageToProbeTransform);
  vtkSmartPointer<vtkMatrix4x4> matrix = combinedTransform->GetMatrix();


  const cv::Mat& image = this->trainImageWithFeatures;
  int width = image.cols;
  int height = image.rows;
  vtkSmartPointer<vtkImageImport> importer = vtkSmartPointer<vtkImageImport>::New();
  // TODO: check type of opencv image data first...
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetImportVoidPointer(image.data);
  importer->SetWholeExtent(0,width-1,0, height-1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  importer->Update();
  this->matchImageData = importer->GetOutput();


  this->queryImageData = importer->GetOutput();
  this->matchNode->SetAndObserveImageData(this->matchImageData);
  this->matchNode->SetIJKToRASMatrix(matrix);

  if(!this->GetMRMLScene()->IsNodePresent(this->matchNode))
    this->GetMRMLScene()->AddNode(this->matchNode);
}

vnl_double_3 arrayToVnlDouble(double arr[4])
{
  vnl_double_3 result;
  result[0]=arr[0];
  result[1]=arr[1];
  result[2]=arr[2];
  return result;
}

void vnlToArrayDouble(vnl_double_3 v, double arr[4])
{
  arr[0]=v[0];
  arr[1]=v[1];
  arr[2]=v[2];
  arr[3]=1.0;
}

vnl_double_3 transformPoint(const vnl_double_3 in, vtkMatrix4x4* transform)
{
  double point[4];
  vnlToArrayDouble(in, point);
  double point_t[4];
  transform->MultiplyPoint(point, point_t);
  return arrayToVnlDouble(point_t);
}

void getPlaneFromPoints(vnl_double_3 p1, vnl_double_3 p2, vnl_double_3 p3, vnl_double_3& normal, double& d)
{
  vnl_double_3 v1,v2;
    v1 = p1-p2;
    v2 = p1-p3;
    v1.normalize();
    v2.normalize();

    normal = vnl_cross_3d(v1,v2);       // Because (a,b,c) is given by normal vector
    normal.normalize();
    d = -dot_product(p1,normal);      // Because a.x1+b.x2+c.x3+d=0
}

void writeMatlabFile(const std::vector<vnl_double_3>& trainPoints, const std::vector<int>& inliersIdx,  vtkMatrix4x4* groundTruth, vtkMatrix4x4* estimate, int maxX, int maxY)
{
  #ifdef WIN32
    std::ofstream matlabof("C:\\Users\\DanK\\MProject\\data\\Results\\kpdata.m");
  #else
    std::ofstream matlabof("/Users/dkostro/Projects/MProject/data/US/Results/kpdata.m");
  #endif
  std::ostringstream oss1, oss2, oss3;

  for(int j=0; j<trainPoints.size(); j++)
  {
    if(j>0) {
      oss1 << ","; oss2 << ","; oss3 << ",";
    }
    oss1 << trainPoints[j][0];
    oss2 << trainPoints[j][1];
    oss3 << trainPoints[j][2];
  }

  matlabof << "X=["+oss1.str()+"];\nY=["+oss2.str()+"];\nZ=["+oss3.str()+"];\n";

  oss1.clear(); oss1.str("");
  oss2.clear(); oss2.str("");
  oss3.clear(); oss3.str("");

  for(int j=0; j<inliersIdx.size(); j++)
  {
    if(j>0) {
      oss1 << ","; oss2 << ","; oss3 << ",";
    }
    oss1 << trainPoints[inliersIdx[j]][0];
    oss2 << trainPoints[inliersIdx[j]][1];
    oss3 << trainPoints[inliersIdx[j]][2];
  }

  matlabof << "Xi=["+oss1.str()+"];\nYi=["+oss2.str()+"];\nZi=["+oss3.str()+"];\n";
  matlabof << "X=setdiff(X,Xi); Y=setdiff(Y,Yi); Z=setdiff(Z,Zi);\n";


  vnl_double_3 p1, p2, p3, p4;
  p1[0]=0; p1[1]=0; p1[2]=0;
  p2[0]=0; p2[1]=maxY; p3[2]=0;
  p3[0]=maxX; p3[1]=0; p3[2]=0;
  p4[0]=maxX; p4[1]=maxY; p4[2]=0;

  vnl_double_3 vpe1 = transformPoint(p1, estimate);
  vnl_double_3 vpe2 = transformPoint(p2, estimate);
  vnl_double_3 vpe3 = transformPoint(p3, estimate);
  vnl_double_3 vpe4 = transformPoint(p4, estimate);
  vnl_double_3 vpg1 = transformPoint(p1, groundTruth);
  vnl_double_3 vpg2 = transformPoint(p2, groundTruth);
  vnl_double_3 vpg3 = transformPoint(p3, groundTruth);
  vnl_double_3 vpg4 = transformPoint(p4, groundTruth);


  vnl_double_3 normal_e;
  double d_e;
  vnl_double_3 normal_g;
  double d_g;

  getPlaneFromPoints(vpe1, vpe2, vpe3, normal_e, d_e);
  getPlaneFromPoints(vpg1, vpg2, vpg3, normal_g, d_g);

  double a = -normal_e[0]/normal_e[2];
  double b = -normal_e[1]/normal_e[2];
  double c = -d_e/normal_e[2];
  matlabof << "efunc=@(x,y)(" << a << "*x+" << b << "*y+" << c << ");\n";

  a = -normal_g[0]/normal_g[2];
  b = -normal_g[1]/normal_g[2];
  c = -d_g/normal_g[2];
  matlabof << "gfunc=@(x,y)(" << a << "*x+" << b << "*y+" << c << ");\n";

  matlabof << "scatter3(X,Y,Z,20,'red'); hold on; scatter3(Xi,Yi,Zi,20,'green');\n";
 
  matlabof << "cornerXe = [" << vpe1[0] << " " << vpe2[0] << " " << vpe3[0] << " " << vpe4[0] << "];\n";
  matlabof << "cornerYe = [" << vpe1[1] << " " << vpe2[1] << " " << vpe3[1] << " " << vpe4[1] << "];\n";
  matlabof << "cornerXg = [" << vpg1[0] << " " << vpg2[0] << " " << vpg3[0] << " " << vpg4[0] << "];\n";
  matlabof << "cornerYg = [" << vpg1[1] << " " << vpg2[1] << " " << vpg3[1] << " " << vpg4[1] << "];\n";

  matlabof << "h(1)=ezmesh(gfunc, [min(cornerXg) max(cornerXg) min(cornerYg) max(cornerYg)],20);\n";
  matlabof << "h(2)=ezmesh(efunc, [min(cornerXe) max(cornerXe) min(cornerYe) max(cornerYe)],20);\n";
  matlabof << "colormap([0,1,0;1,0,0]);\n";

  matlabof << "set(h(1),'CData',ones(20,20));\n";
  matlabof << "set(h(2),'CData',2*ones(20,20));\n";
  
}

void vtkSlicerSurfFeaturesLogic::updateMatchNodeRansac()
{
  if(this->correspondenceProgress != 100)
    return;
  std::ostringstream oss;
  
  // Find coordinates of all keypoints matching with the current query image
  // Train points in the x,y,z patient coordinates
  std::vector<vnl_double_3> trainPoints;
  // Query points in the u,v image coordinates
  std::vector<vnl_double_3> queryPoints;
  for(int j=0; j<this->afterHoughMatches[this->currentImgIndex].size(); j++)
  {
    int imgIdx = this->afterHoughMatches[this->currentImgIndex][j].imgIdx;
    if(imgIdx == -1)
      continue;
    int tIdx = this->afterHoughMatches[this->currentImgIndex][j].trainIdx;
    int qIdx = this->afterHoughMatches[this->currentImgIndex][j].queryIdx;

    vnl_double_3 p;
    vtkSmartPointer<vtkMatrix4x4> transform = vtkSmartPointer<vtkMatrix4x4>::New();
    transform->Identity();
    for(int i=0; i<3; i++)
      for(int j=0; j<4; j++)
      transform->SetElement(i,j,this->trainImagesTransform[imgIdx][i*4+j]);
    vtkSmartPointer<vtkTransform> combinedTransform = vtkSmartPointer<vtkTransform>::New();
    combinedTransform->Concatenate(transform);
    combinedTransform->Concatenate(this->ImageToProbeTransform);
    vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
    matrix = combinedTransform->GetMatrix();
    float point[4];
    point[0] = this->trainKeypoints[imgIdx][tIdx].pt.x;
    point[1] = this->trainKeypoints[imgIdx][tIdx].pt.y;
    point[2] = 0.0;
    point[3] = 1.0;
    float tPoint[4];
    matrix->MultiplyPoint(point,tPoint);
    trainPoints.push_back(vnl_double_3(tPoint[0],tPoint[1],tPoint[2]));
    queryPoints.push_back(vnl_double_3(this->queryKeypoints[this->currentImgIndex][qIdx].pt.x, this->queryKeypoints[this->currentImgIndex][qIdx].pt.y, 0.0));
  }


  vnl_double_3 old_plane(0.0,0.0,1.0);
  vnl_double_3 rotationAxis;
  std::vector<int> inliersIdx;
  std::vector<int> planePointsIdx;
  double angle;

  this->ransac(trainPoints, inliersIdx, planePointsIdx);
  if(planePointsIdx.empty())
  {
    this->console->insertPlainText("Not enough matches to construct plane\n");
    return;
  }

  vnl_double_3 plane = vnl_cross_3d(trainPoints[planePointsIdx[0]]-trainPoints[planePointsIdx[1]], trainPoints[planePointsIdx[0]]-trainPoints[planePointsIdx[2]]);
  plane.normalize();
  double d = -dot_product(trainPoints[planePointsIdx[0]],plane);


  // Project the train keypoints to the computed plane
  std::vector<vnl_double_3> projTrainPoints;
  for(int i=0; i<trainPoints.size(); i++)
  {
    projTrainPoints.push_back(projectPoint(trainPoints[i], plane, d));
  }

  // In this plane normal vector we got wx,wy,wz figured out. Now we want ux,uy,uz
  int max_iter = 10000;
  int mult = 20;
  int iter = projTrainPoints.size()*mult<max_iter ? projTrainPoints.size()*mult : max_iter;
  vnl_double_3 u(0,0,0);
  vnl_double_3 v(0,0,0);
  for(int i=0; i<iter; i++)
  {
    // randomly get two inliers
    int idx1=0, idx2=0;
    while(idx1==idx2){
      idx1 = rand()%inliersIdx.size();
      idx2 = rand()%inliersIdx.size();
    }
    // compute the diff vector of the query keypoint locations
    vnl_double_3 queryDiff = queryPoints[inliersIdx[idx1]] - queryPoints[inliersIdx[idx2]];
    // Angle between the diff vector and the u unit vector
    double theta = getAngle(vnl_double_3(1.0,0.0,0.0), queryDiff);
    double theta_v = getAngle(vnl_double_3(0.0,1.0,0.0), queryDiff);
  
    // compute the diff vector of the projected train keypoint locations
    vnl_double_3 trainDiff = projTrainPoints[inliersIdx[idx1]] - projTrainPoints[inliersIdx[idx2]];
  
    // rotate this around normal vector by theta
    vnl_matrix<double> rotationMatrix = getRotationMatrix(plane, theta);
    vnl_matrix<double> rotationMatrix_v = getRotationMatrix(plane, theta_v);
    vnl_double_3 u_i = convertVnlMatrixToVector(rotationMatrix*convertVnlVectorToMatrix(trainDiff));
    vnl_double_3 v_i = convertVnlMatrixToVector(rotationMatrix_v*convertVnlVectorToMatrix(trainDiff));
    u_i.normalize();
    v_i.normalize();
    v += v_i;
    u += u_i;
  }
  

  // We figure out v computing the cross product between u and w
  u /= iter;
  v /= iter;
  vnl_double_3 w = vnl_cross_3d(u,v);
  if(dot_product(plane,w)>0)
    w=-plane;
  else
    w=plane;
  w.normalize();
  u.normalize();
  v.normalize();
  v = vnl_cross_3d(u,w);
  v.normalize();

  // Scale
  u *= 0.10763;
  v *= 0.10530;
  w *= 0.10646;


  vnl_double_3 trainCentroid(0,0,0);
  vnl_double_3 queryCentroid(0,0,0);
  for(int i=0; i<inliersIdx.size(); i++) {
    trainCentroid += projTrainPoints[inliersIdx[i]];
    queryCentroid += queryPoints[inliersIdx[i]];
  }
  trainCentroid /= queryPoints.size();
  queryCentroid /= queryPoints.size();

  // Now we still have to figure out a translation component. Calculate the centroids of query and train keypoints and make them match
  vnl_double_3 t;
  t[0] = trainCentroid[0] - (u[0]*queryCentroid[0] + v[0]*queryCentroid[1] + w[0]*queryCentroid[2]);
  t[1] = trainCentroid[1] - (u[1]*queryCentroid[0] + v[1]*queryCentroid[1] + w[1]*queryCentroid[2]);
  t[2] = trainCentroid[2] - (u[2]*queryCentroid[0] + v[2]*queryCentroid[1] + w[2]*queryCentroid[2]);

  // vtkSmartPointer<vtkMatrix4x4> rotation = vtkSmartPointer<vtkMatrix4x4>::New();
  // vtkSmartPointer<vtkMatrix4x4> scaling = vtkSmartPointer<vtkMatrix4x4>::New();
  // vtkSmartPointer<vtkMatrix4x4> translation = vtkSmartPointer<vtkMatrix4x4>::New();
  vtkSmartPointer<vtkMatrix4x4> estimate = vtkSmartPointer<vtkMatrix4x4>::New();

  // getAxisAndRotationAngle(old_plane, plane, rotationAxis, angle);
  //   vnl_matrix<double> vnlRotMat = getRotationMatrix(rotationAxis, angle);
  //   vnlToVtkMatrix(vnlRotMat, rotation);
  estimate->Identity();
  estimate->SetElement(0,0,u[0]);
  estimate->SetElement(1,0,u[1]);
  estimate->SetElement(2,0,u[2]);
  estimate->SetElement(0,1,v[0]);
  estimate->SetElement(1,1,v[1]);
  estimate->SetElement(2,1,v[2]);
  estimate->SetElement(0,2,w[0]);
  estimate->SetElement(1,2,w[1]);
  estimate->SetElement(2,2,w[2]);
  estimate->SetElement(0,3,t[0]);
  estimate->SetElement(1,3,t[1]);
  estimate->SetElement(2,3,t[2]);
  
  // Compute squared distance error of matching keypoints
  std::vector<vnl_double_3> trainInlierPoints;
  std::vector<vnl_double_3> queryInlierPoints;
  for(int i=0; i<inliersIdx.size(); i++) {
    trainInlierPoints.push_back(trainPoints[inliersIdx[i]]);
    float qpoint[4];
    qpoint[0] = queryPoints[inliersIdx[i]][0];
    qpoint[1] = queryPoints[inliersIdx[i]][1];
    qpoint[2] = 0.0; qpoint[3] = 1.0;
    float tqpoint[4];
    estimate->MultiplyPoint(qpoint, tqpoint);
    vnl_double_3  vnl_tqpoint(tqpoint[0], tqpoint[1], tqpoint[2]);
    queryInlierPoints.push_back(vnl_tqpoint);
  }
  double meanMatchDistance = computeMeanDistance(trainInlierPoints, queryInlierPoints);
  
  oss << "Mean Match Distance: " << meanMatchDistance << std::endl;
    

  // Find the ground truth transform
  vtkSmartPointer<vtkMatrix4x4> transform = vtkSmartPointer<vtkMatrix4x4>::New();
  transform->Identity();
  for(int i=0; i<3; i++)
    for(int j=0; j<4; j++)
    transform->SetElement(i,j,this->queryImagesTransform[this->currentImgIndex][i*4+j]);
  vtkSmartPointer<vtkTransform> combinedTransform = vtkSmartPointer<vtkTransform>::New();
  combinedTransform->Concatenate(transform);
  combinedTransform->Concatenate(this->ImageToProbeTransform);
  vtkSmartPointer<vtkMatrix4x4> groundTruth = vtkSmartPointer<vtkMatrix4x4>::New();
  groundTruth = combinedTransform->GetMatrix();

  // Compute squared distance plane to plane in roi
  std::vector<vnl_double_3> queryPlanePoints;
  std::vector<vnl_double_3> estimatePlanePoints;
  int step = 4;
  for(int i=0; i<this->croppedMask.rows; i+=step) {
    for(int j=0; j<this->croppedMask.cols; j+=step) {
      if(this->croppedMask.at<unsigned char>(i,j) == 0)
        continue;
      
      // Apply transforms to current point
      float point[4];
      point[0] = i; point[1] = j;
      point[2] = 0.0; point[3] = 1.0;
      float qpoint[4];
      float epoint[4];
      estimate->MultiplyPoint(point, epoint);
      groundTruth->MultiplyPoint(point, qpoint);
      
      // Compute distance between points
      queryPlanePoints.push_back(vnl_double_3(qpoint[0], qpoint[1], qpoint[2]));
      estimatePlanePoints.push_back(vnl_double_3(epoint[0], epoint[1], epoint[2]));
    }
  }
  
  double meanPlaneDistance = computeMeanDistance(queryPlanePoints, estimatePlanePoints);
  oss << "Mean Plane Distance: " << meanPlaneDistance << std::endl;
  this->console->insertPlainText(oss.str().c_str());

  writeMatlabFile(trainPoints, inliersIdx, groundTruth, estimate, this->queryImages[this->currentImgIndex].cols, this->queryImages[this->currentImgIndex].rows);

  // scaling->Identity();
  // scaling->SetElement(0,0,0.10763);
  // scaling->SetElement(1,1,0.10530);
  // scaling->SetElement(2,2,0.10646);


  // vtkSmartPointer<vtkTransform> transform = vtkSmartPointer<vtkTransform>::New();
  // transform->Concatenate(rotation);
  // transform->Concatenate(scaling);
  // vtkSmartPointer<vtkMatrix4x4> estimate = transform->GetMatrix();

  // Translation ...

  // Store the estimate
  //this->queryTransformEstimate[this->currentImgIndex] = vtkToStdMatrix(estimate);

  cv::Mat& image = this->trainImageWithFeatures;
  image = this->queryImages[this->currentImgIndex].clone();
  //image = cv::Mat(this->queryImages[this->currentImgIndex].rows, this->queryImages[this->currentImgIndex].cols, CV_8U);
  //image.setTo(cv::Scalar::all(255));

  // TODO draw circles where inliners are...

  int width = image.cols;
  int height = image.rows;
  vtkSmartPointer<vtkImageImport> importer = vtkSmartPointer<vtkImageImport>::New();
  // TODO: check type of opencv image data first...
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetImportVoidPointer(image.data);
  importer->SetWholeExtent(0,width-1,0, height-1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  importer->Update();
  this->matchImageData = importer->GetOutput();


  this->matchNode->SetAndObserveImageData(this->matchImageData);
  this->matchNode->SetIJKToRASMatrix(estimate);

  if(!this->GetMRMLScene()->IsNodePresent(this->matchNode))
    this->GetMRMLScene()->AddNode(this->matchNode);
}


void vtkSlicerSurfFeaturesLogic::setConsole(QTextEdit* console)
{
  this->console = console;
  this->resetConsoleFont();
}

void vtkSlicerSurfFeaturesLogic::setMinHessian(int minHessian)
{
  if(minHessian != this->minHessian){
    this->minHessian = minHessian;
    this->correspondenceProgress = 0;
    this->Modified();
  }
}

int vtkSlicerSurfFeaturesLogic::getMinHessian()
{
  return this->minHessian;
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

void vtkSlicerSurfFeaturesLogic::showImage(vtkMRMLNode *node)
{
  // Get Image Data
  vtkMRMLScalarVolumeNode* sv_node = vtkMRMLScalarVolumeNode::SafeDownCast(node);
  vtkImageData* data = sv_node->GetImageData();
  if(!data)
    return;

  // Crop, then convert to opencv matrix
  vtkImageData* cdata = this->cropData(data);
  cv::Mat ocvImg = this->convertImage(cdata);

  // Compute keypoints and descriptors
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  this->computeKeypointsAndDescriptors(ocvImg, keypoints, descriptors);

  this->showImage(ocvImg,keypoints);
}



void vtkSlicerSurfFeaturesLogic::showImage(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints)
{
  if(img.empty())
    return;
  cv::Mat img_keypoints;
  cv::drawKeypoints( img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
  int x,y;
  this->computeCentroid(this->croppedMask,x,y);
  cv::Point centroid(x,y);
  cv::circle(img_keypoints, centroid, 20, cv::Scalar(255,255,255), 3);

  cvStartWindowThread();
  cv::imshow("Keypoints", img_keypoints );
  cv::waitKey(0);
  cvDestroyWindow("Keypoints");
}


void vtkSlicerSurfFeaturesLogic::setBogusFile(std::string file)
{
  char* pch = &file[0];
  if( strstr( pch, ".mha" ) ){
    this->bogusFile = file;
  }
}

void vtkSlicerSurfFeaturesLogic::setTrainFile(std::string file)
{
  char* pch = &file[0];
  if( strstr( pch, ".mha" ) ){
    this->trainFile = file;
  }
}

void vtkSlicerSurfFeaturesLogic::setQueryFile(std::string file)
{
  char* pch = &file[0];
  if( strstr( pch, ".mha" ) ){
    this->queryFile = file;
  }
}

void vtkSlicerSurfFeaturesLogic::computeKeypointsAndDescriptors(const cv::Mat& data, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
  // Get the descriptor of the current image
  cv::SurfFeatureDetector detector(this->minHessian);
  detector.detect(data,keypoints,this->croppedMask);

  // Get keypoints' surf descriptors
  cv::SurfDescriptorExtractor extractor;
  extractor.compute(data, keypoints, descriptors);
}

bool vtkSlicerSurfFeaturesLogic::isTracked(vtkMRMLNode* node)
{
  vtkMRMLVolumeNode* vnode = vtkMRMLVolumeNode::SafeDownCast(node);
  if(!vnode)
    return false;
  double* origin = vnode->GetSpacing();
  if(origin[0]==1.0 && origin[1]==1.0 && origin[2]==1.0)
    return false;
  return true;
}

void vtkSlicerSurfFeaturesLogic::nextImage()
{
  this->currentImgIndex += 1;
  if(this->currentImgIndex >= this->queryImages.size())
    this->currentImgIndex = 0;
  this->playDirection = 1;
  this->updateImage();
}

void vtkSlicerSurfFeaturesLogic::previousImage()
{
  this->currentImgIndex -= 1;
  if(this->currentImgIndex < 0)
    this->currentImgIndex = this->queryImages.size()-1;
  this->playDirection = -1;
  this->updateImage();
}

void vtkSlicerSurfFeaturesLogic::play()
{
  if(this->playDirection > 0)
    this->nextImage();
  else
    this->previousImage();
}

void vtkSlicerSurfFeaturesLogic::updateImage()
{
  std::ostringstream oss;
  oss << "The current image is " << this->queryImagesNames[this->currentImgIndex] << std::endl;
  this->console->insertPlainText(oss.str().c_str());
  this->drawQueryMatches();
  //this->drawBestTrainMatches();
  this->updateQueryNode();
  //this->updateMatchNode();
  this->updateMatchNodeRansac();
}


void vtkSlicerSurfFeaturesLogic::showCurrentImage()
{
  this->showImage(this->queryImages[this->currentImgIndex], this->queryKeypoints[this->currentImgIndex]);
}

void vtkSlicerSurfFeaturesLogic::computeBogus()
{
  this->setCorrespondenceProgress(0);
  this->setBogusProgress(0);
  this->readAndComputeFeaturesOnMhaFile(this->bogusFile, this->bogusImages, this->bogusImagesNames, \
    this->bogusImagesTransform, this->bogusTransformsValidity, this->bogusKeypoints, this->bogusDescriptors, this->bogusDescriptorMatcher, this->bogusStartFrame, this->bogusStopFrame, "bogus");
}

void vtkSlicerSurfFeaturesLogic::computeTrain()
{
  this->setCorrespondenceProgress(0);
  this->setTrainProgress(0);
  this->readAndComputeFeaturesOnMhaFile(this->trainFile, this->trainImages, this->trainImagesNames, \
    this->trainImagesTransform, this->trainTransformsValidity, this->trainKeypoints, this->trainDescriptors, this->trainDescriptorMatcher, this->trainStartFrame, this->trainStopFrame, "train");
}

void vtkSlicerSurfFeaturesLogic::computeQuery()
{
  this->setQueryProgress(0);
  this->setCorrespondenceProgress(0);
  this->readAndComputeFeaturesOnMhaFile(this->queryFile, this->queryImages, this->queryImagesNames, \
    this->queryImagesTransform, this->queryTransformsValidity, this->queryKeypoints, this->queryDescriptors, this->queryDescriptorMatcher, this->queryStartFrame, this->queryStopFrame, "query");
}

void vtkSlicerSurfFeaturesLogic::readAndComputeFeaturesOnMhaFile(const std::string& file, std::vector<cv::Mat>& images,\
  std::vector<std::string>& imagesNames,\
  std::vector<std::vector<float> >& imagesTransform,\
  std::vector<bool>& transformsValidity,\
  std::vector<std::vector<cv::KeyPoint> >& keypoints,\
  std::vector<cv::Mat>& descriptors,\
  cv::Ptr<cv::DescriptorMatcher> descriptorMatcher,\
  int& startFrame, int& stopFrame, std::string who)
{
  if(!this->cropRatiosValid())
    return;
  const char* pch = &(file)[0];
  // Read images
  if( strstr( pch, ".mha" ) ){
    images.clear();
    imagesNames.clear();
    imagesTransform.clear();
    transformsValidity.clear();
    if( read_mha( file, images, imagesNames, imagesTransform, transformsValidity, startFrame, stopFrame) )
      return;
  }
  else
    return;
    
  // Compute keypoints and descriptors
  keypoints.clear();
  descriptors.clear();

  // Start and stop frames may have been modified
  this->Modified();
  

  // Crop the mask
  this->croppedMask = this->mask.clone();
  cropData(this->croppedMask);
  
  for(int i=0; i<images.size(); i++)
  {
    cropData(images[i]);
  }
}

bool vtkSlicerSurfFeaturesLogic::isCorrespondenceComputing()
{
  return (this->queryImages.size() != this->afterHoughMatches.size());
}

bool vtkSlicerSurfFeaturesLogic::isQueryLoading()
{
  return (this->queryImages.size() != this->queryKeypoints.size());
}

bool vtkSlicerSurfFeaturesLogic::isTrainLoading()
{
  return (this->trainImages.size() != this->trainKeypoints.size());
}

bool vtkSlicerSurfFeaturesLogic::isBogusLoading()
{
  return (this->bogusImages.size() != this->bogusKeypoints.size());
}

bool vtkSlicerSurfFeaturesLogic::isLoading()
{
  return (this->isQueryLoading() || this->isBogusLoading() || this->isTrainLoading());
}

int vtkSlicerSurfFeaturesLogic::computeNext(vector<Mat>& images, vector<vector<KeyPoint> >& keypoints, vector<Mat>& descriptors, cv::Ptr<cv::DescriptorMatcher> descriptorMatcher)
{
  int index = keypoints.size();
  if(index == images.size())
    return 100;
  keypoints.push_back(vector<KeyPoint>());
  descriptors.push_back(Mat());
  this->computeKeypointsAndDescriptors(images[index], keypoints[index], descriptors[index]);
  int progress = ((index+1)*100)/images.size();
  if(progress == 100) {
    descriptorMatcher->clear();
    descriptorMatcher->add(descriptors);
  }
  return progress;
}

void vtkSlicerSurfFeaturesLogic::computeNextQuery()
{
  this->setQueryProgress(this->computeNext(this->queryImages, this->queryKeypoints, this->queryDescriptors, this->queryDescriptorMatcher));
}

void vtkSlicerSurfFeaturesLogic::computeNextTrain()
{
  this->setTrainProgress(this->computeNext(this->trainImages, this->trainKeypoints, this->trainDescriptors, this->trainDescriptorMatcher));
}

void vtkSlicerSurfFeaturesLogic::computeNextBogus()
{
 this->setBogusProgress(this->computeNext(this->bogusImages, this->bogusKeypoints, this->bogusDescriptors, this->bogusDescriptorMatcher)); 
}

void vtkSlicerSurfFeaturesLogic::computeInterSliceCorrespondence()
{
  if(this->queryImages.empty() || this->bogusImages.empty() || this->trainImages.empty())
    return;
    
  if(this->isLoading())
    return;

  this->bestMatches.clear();
  this->bestMatchesCount.clear();
  this->matchesWithBestTrainImage.clear();
  this->afterHoughMatches.clear();
  
  // Compute mask centroid
  this->computeCentroid(this->croppedMask, this->xcentroid, this->ycentroid);
  
  this->Modified();
}

void vtkSlicerSurfFeaturesLogic::computeNextInterSliceCorrespondence()
{
  if(this->queryImages.empty() || this->bogusImages.empty() || this->trainImages.empty())
    return;
    
  if(this->isQueryLoading() || this->isBogusLoading() || this->isTrainLoading())
    return;

  
  // Go to next query image
  int i = this->afterHoughMatches.size();
  if(i == this->queryImages.size())
    return;
  
  
  vector<DMatch> matches;
  vector<vector<DMatch> > mmatches;      // #query keypoints x #depth
  vector<vector<DMatch> > mmatchesBogus; // #query keypoints x #depth
  vector<int> vecImgMatches;
  vecImgMatches.resize( this->trainKeypoints.size() );

  for( int j = 0; j < vecImgMatches.size(); j++ )
    vecImgMatches[j] = 0;

    // mmatches will contain the 3 closest matches to each keypoint found in the query image
  matchDescriptorsKNN( this->queryDescriptors[i], mmatches, this->trainDescriptorMatcher, 3 );
  matchDescriptorsKNN( this->queryDescriptors[i], mmatchesBogus, this->bogusDescriptorMatcher, 1 );

    // Vector that contains the matches remaining after filtering
  vector< DMatch > vmMatches; // #query keypoints - #filtered out

  float fRatioThreshold = 0.95;
    // Disable best matches if they also have a good match with bogus data
  for( int j = 0; j < mmatches.size(); j++ )
  {
    float fRatio = mmatches[j][0].distance / mmatchesBogus[j][0].distance;
      //float fRatio = mmatches[j][0].distance / mmatches[j][1].distance;
    if( fRatio > fRatioThreshold )
    {
      mmatches[j][0].queryIdx = -1;
      mmatches[j][0].trainIdx = -1;
      mmatches[j][0].imgIdx = -1;
    }
    else
    {
        // Keep the filtered matches in a 1D vector
      vmMatches.push_back( mmatches[j][0] );
    }
  }
    // Do the hough transform. Filters out some more matches from vmMatches.
  int iMatchCount = houghTransform( this->queryKeypoints[i], this->trainKeypoints, vmMatches, this->xcentroid, this->ycentroid );
  this->afterHoughMatches.push_back(vmMatches);

    // Count the number of votes for each train image
  for( int j = 0; j < vmMatches.size(); j++ )
  {
    int iImg =  vmMatches[j].imgIdx;
    if( iImg >= 0 )
    {
      vecImgMatches[iImg]++;
    }
  }

  vector< int > vecImgMatchesSmooth;
  vecImgMatchesSmooth.resize( vecImgMatches.size(), 0 );

  int iMaxIndex = -1;
  int iMaxCount = -1;
  for( int j = 0; j < vecImgMatches.size(); j++ )
  {
      // Store the index and count of the training image with the most counts.
    int iCount = vecImgMatches[j];
    if( iCount > iMaxCount )
    {
      iMaxCount = iCount;
      iMaxIndex = j;
    }
      // Smooth the # of matches curve by convultion with [1 1 1].
    vecImgMatchesSmooth[j] = vecImgMatches[j];
    if( j > 0 ) vecImgMatchesSmooth[j] += vecImgMatches[j-1];
    if( j < vecImgMatches.size()-1 ) vecImgMatchesSmooth[j] += vecImgMatches[j+1];
  }

    // vecImgMatches is reinitilised. It will now store flags.
  for( int j = 0; j < vecImgMatchesSmooth.size(); j++ )
  {
    vecImgMatches[j] = 0;
  }
  for( int j = 0; j < vecImgMatchesSmooth.size(); j++ )
  {
      // If training image has more than 2 matches (after smoothing), flag the training image and its immediate neighborhood.
    if( vecImgMatchesSmooth[j] >= 2 )
    {
        // flag neighborhood
      vecImgMatches[j] = 1;
      if( j > 0 ) vecImgMatches[j-1]=1;
      if( j < vecImgMatches.size()-1 ) vecImgMatches[j+1]=1;
    }
  }

    // Save all matches
  vector< DMatch > vmMatchesSmooth;
  vmMatchesSmooth.clear();
  for( int j = 0; j < vecImgMatchesSmooth.size(); j++ )
  {
      // Discard unflagged train images
    if( vecImgMatches[j] > 0 )
    {
      for( int k = 0; k < vmMatches.size(); k++ )
      {
        int iImg =  vmMatches[k].imgIdx;
        if( iImg == j )
        {
          vmMatchesSmooth.push_back( vmMatches[k] );
        }
      }
    }
  }

    // This commented part here would be I think a more efficient way for the block just above this
    //for(int k=0; k < vmMatches.size(); k++)
    //{
    //  iImg = vmMatches[k].imgIdx;
    //  if( vecImgMatches[imgIdx] > 0 )
    //    continue;
    //  vmMatchesSmooth.push_back( vmMatches[k] );
    //}

  vector< DMatch > matchesWithTrain;
  for(int k=0; k<vmMatches.size(); k++)
  {
    if(vmMatches[k].imgIdx == iMaxIndex)
      matchesWithTrain.push_back(vmMatches[k]);
  }
  this->matchesWithBestTrainImage.push_back(matchesWithTrain);


    // Record closest match
  this->bestMatches.push_back(iMaxIndex);
  this->bestMatchesCount.push_back(iMaxCount);

  int progress = ((i+1)*100)/this->queryImages.size();
  this->setCorrespondenceProgress(progress);
}

cv::Mat vtkSlicerSurfFeaturesLogic::drawFeatures(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints)
{
  cv::Mat result = img.clone();
  int radius = 5;
  int thickness = 2;
  int brightness=0;

  for(int i=0; i<keypoints.size(); i++) {
    cv::Point center = keypoints[i].pt;
    cv::circle(result,center,radius,brightness,thickness);
  }

  return result;
}


void vtkSlicerSurfFeaturesLogic::drawQueryMatches()
{
  if(this->queryProgress != 100)
    return;
  int qidx = this->currentImgIndex;
  this->queryImageWithFeatures = this->queryImages[qidx].clone();
  if(this->correspondenceProgress != 100)
    return;
  int radius = 5;
  int thickness = 1;
  int brightness = 0;

  for(int i=0; i<this->matchesWithBestTrainImage[qidx].size(); i++)
  {
    cv::Point center = this->queryKeypoints[qidx][this->matchesWithBestTrainImage[qidx][i].queryIdx].pt;
    cv::circle(this->queryImageWithFeatures,center,radius,brightness,thickness);
  }
}
void vtkSlicerSurfFeaturesLogic::drawBestTrainMatches()
{
  // We don't know best train image if correspondence not computed
  if(this->correspondenceProgress != 100)
    return;
  int qidx = this->currentImgIndex;
  int tidx = this->bestMatches[this->currentImgIndex];
  this->trainImageWithFeatures = this->trainImages[tidx].clone();
  int radius = 5;
  int thickness = 1;
  int brightness = 0;

  for(int i=0; i<this->matchesWithBestTrainImage[qidx].size(); i++)
  {
    cv::Point center = this->trainKeypoints[tidx][this->matchesWithBestTrainImage[qidx][i].trainIdx].pt;
    cv::circle(this->trainImageWithFeatures,center,radius,brightness,thickness);
  }
}

void vtkSlicerSurfFeaturesLogic::showCropFirstImage()
{
  if(this->firstImage.empty())
    return;
  if(!this->cropRatiosValid())
    return;
  this->firstImageCropped = this->firstImage.clone();
  this->cropData(this->firstImageCropped);

  cv::Mat& image = this->firstImageCropped;
  int width = image.cols;
  int height = image.rows;
  vtkSmartPointer<vtkImageImport> importer = vtkSmartPointer<vtkImageImport>::New();
  importer->SetDataScalarTypeToUnsignedChar();
  importer->SetImportVoidPointer(image.data);
  importer->SetWholeExtent(0,width-1,0, height-1, 0, 0);
  importer->SetDataExtentToWholeExtent();
  importer->Update();
  vtkImageData* vtkImage = importer->GetOutput();

  vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  matrix->Identity();

  // Pointer is shared with the opencv image
  this->queryNode->SetAndObserveImageData(vtkImage);
  this->queryNode->SetIJKToRASMatrix(matrix);

  if(!this->GetMRMLScene()->IsNodePresent(this->queryNode))
    this->GetMRMLScene()->AddNode(this->queryNode);


}

void vtkSlicerSurfFeaturesLogic::saveCurrentImage()
{
  if(this->queryImages.empty())
    return;
  std::string fn = this->queryImagesNames[this->currentImgIndex];
  fn.replace(fn.end()-4,fn.end(),".png");
  cv::imwrite(std::string("C:\\Users\\DanK\\MProject\\data\\US\\SlicerSaved\\")+fn,this->queryImages[this->currentImgIndex]);
}


int vtkSlicerSurfFeaturesLogic::ransac(const std::vector<vnl_double_3>& points, std::vector<int>& inliersIdx, std::vector<int>& planePointsIdx)
{
  inliersIdx.clear();
  planePointsIdx.clear();

  if(points.size()==3)
  {
    planePointsIdx.push_back(0);
    planePointsIdx.push_back(1);
    planePointsIdx.push_back(2);
    inliersIdx.push_back(0);
    inliersIdx.push_back(1);
    inliersIdx.push_back(2);
    return 0;
  }
  else if(points.size()<3)
    return 1;

  int iIterations = 1000;
  float threshold = this->ransacMargin; // (mm)
  int iClose = 10; // Number of close values needed to assert model fits data well

  // current and best model parameters
  vnl_double_3 currentN;
  double currentD;
  vnl_double_3 bestN;

  // Error
  double numOutliersBest = points.size();
  double bestError = DBL_MAX;

  for(int i=0; i<iIterations; i++)
  {
    // Select three points randomly
    int idx[3] = {0,0,0};
    while(idx[0]==idx[1] || idx[1]==idx[2] || idx[0]==idx[2])
    {
      idx[0] = rand()%points.size();
      idx[1] = rand()%points.size();
      idx[2] = rand()%points.size();
    }
    if((idx[0]==47 || idx[1]==47 || idx[2]==47) && (idx[0]==48 || idx[1]==48 || idx[2]==48))
      std::cout << "hello" ;
    // Compute the plane based on these three point
    vnl_double_3 p1,p2,p3;
    p1 = points[idx[0]];
    p2 = points[idx[1]];
    p3 = points[idx[2]]; 

    vnl_double_3 v1,v2;
    v1 = p1-p2;
    v2 = p1-p3;
    v1.normalize();
    v2.normalize();

    if(v1.two_norm() < 0.99 || v2.two_norm() < 0.99) // can happen when v is (0,0,0)
      continue;
    // Make sure the vectors are not colinear to avoid numerical problems
    if(abs(dot_product(v1,v2)) > 0.99)
      continue;

    currentN = vnl_cross_3d(v1,v2);       // Because (a,b,c) is given by normal vector
    currentN.normalize();
    currentD = -dot_product(p1,currentN); // Because a.x1+b.x2+c.x3+d=0


    // Compute error of current model
    double numOutliers = 0;
    double currentError = 0;
    std::vector<int> tmpInliers;
    for(int j=0; j<points.size(); j++)
    {
      // Ignore points selected for current model
      if(j==idx[0] || j==idx[1] || j==idx[2])
        continue;
      vnl_double_3 point;
      point[0] = points[j][0];
      point[1] = points[j][1];
      point[2] = points[j][2];
      // Calculate distance between plane and point
      float dist = abs(dot_product(currentN,point)+currentD);
      if(dist>threshold)
      {
        numOutliers++;
      }
      else
      {
        currentError += dist;
        tmpInliers.push_back(j);
      }
    }
    if(numOutliers < numOutliersBest || (numOutliers==numOutliersBest && currentError<bestError))
    {
      numOutliersBest = numOutliers;
      bestN = currentN;
      bestError = currentError;
      planePointsIdx.clear();
      planePointsIdx.push_back(idx[0]);
      planePointsIdx.push_back(idx[1]);
      planePointsIdx.push_back(idx[2]);
      inliersIdx = tmpInliers;
    }
  }
  inliersIdx.push_back(planePointsIdx[0]);
  inliersIdx.push_back(planePointsIdx[1]);
  inliersIdx.push_back(planePointsIdx[2]);
  
  std::ostringstream oss;
  oss << bestError/inliersIdx.size() << " mm error, " << inliersIdx.size() << " inliers, " << numOutliersBest << " outliers."  << std::endl;
  this->console->insertPlainText(oss.str().c_str());
  return 0;
}

void vtkSlicerSurfFeaturesLogic::computeCentroid(const cv::Mat& mask, int& x, int& y)
{
  float rowcentroid=0;
  float colcentroid=0;
  int count = 0;
  for(int i=0; i<mask.rows; i++)
  {
    for(int j=0; j<mask.cols; j++)
    {
      if(mask.at<unsigned char>(i,j) > 0)
      {
        rowcentroid += i;
        colcentroid += j;
        count += 1;
      }
    }
  }
  rowcentroid /= count;
  colcentroid /= count;
  y = (int)rowcentroid;
  x = (int)colcentroid;
}

bool vtkSlicerSurfFeaturesLogic::cropRatiosValid()
{
  return (this->cropRatios[0] < this->cropRatios[1] || this->cropRatios[2] < this->cropRatios[3]);
}

void vtkSlicerSurfFeaturesLogic::writeMatches()
{
  const std::vector<std::vector<DMatch> >& m = this->afterHoughMatches;
  #ifdef WIN32
    std::ofstream ofs("C:\\Users\\DanK\\MProject\\data\\matches.txt");
  #else
    std::ofstream ofs("/Users/dkostro/Projects/MProject/data/US/log/matches.txt");
  #endif
  for(int i=0; i < m.size(); i++)
  {
    int iMatches = 0;
    for(int j=0; j<m[i].size(); j++)
    {
      if(m[i][j].imgIdx != -1)
        iMatches++;
    }
    for(int j=0; j<m[i].size(); j++)
    {
      if(m[i][j].imgIdx == -1)
        continue;
      ofs << iMatches << "|" << this->queryImagesNames[i] << "|" << this->trainImagesNames[m[i][j].imgIdx] << std::endl;
    }
  }
  ofs.close();
}

