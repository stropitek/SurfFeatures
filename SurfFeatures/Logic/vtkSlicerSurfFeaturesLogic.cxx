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

// Matt functions ==================================================================

#ifdef WIN32
  #include <time.h>
  #include <sys\timeb.h>
#endif

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



#define PI 3.1415926535897
float
  angle_radians(
  float fAngle )
{
  return (2.0f*PI*fAngle) / 180.0f;
}

#define PI 3.141592653589793
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



const string defaultDetectorType = "SURF";
const string defaultDescriptorType = "SURF";
const string defaultMatcherType = "FlannBased";
const string defaultQueryImageName = "../../opencv/samples/cpp/matching_to_many_images/query.png";
const string defaultFileWithTrainImages = "../../opencv/samples/cpp/matching_to_many_images/train/trainImages.txt";
const string defaultDirToSaveResImages = "../../opencv/samples/cpp/matching_to_many_images/results";

int
  maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask )
{
  int iCount = 0;
  mask.resize( matches.size() );
  fill( mask.begin(), mask.end(), 0 );
  for( size_t i = 0; i < matches.size(); i++ )
  {
    if( matches[i].imgIdx == trainImgIdx )
    {
      mask[i] = 1;
      iCount++;
    }
  }
  return iCount;
}

int
  maskMatchesByMatchPresent( const vector<DMatch>& matches, vector<char>& mask )
{
  int iCount = 0;
  for( size_t i = 0; i < matches.size(); i++ )
  {
    if( matches[i].queryIdx <= 0 )
      mask[i] = 0;
    if( mask[i] > 0 )
    {
      iCount++;
    }
  }
  return iCount;
}

void readTrainFilenames( const string& filename, string& dirName, vector<string>& trainFilenames )
{
  const char dlmtr = '/';

  trainFilenames.clear();

  ifstream file( filename.c_str() );
  if ( !file.is_open() )
    return;

  size_t pos = filename.rfind(dlmtr);
  dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;
  while( !file.eof() )
  {
    string str; getline( file, str );
    if( str.empty() ) break;
    trainFilenames.push_back(str);
  }
  file.close();
}


// Read in relevant lines from mha file, used with PLUS ultrasound data
void readTrainTransforms_mha(
  const string& filename,
  string& dirName,
  vector<string>& trainFilenames,
  vector <Mat>& trainImages,
  vector< vector<float> >& trainImageData,
  int& firstFrame,
  int& lastFrame
  )
{
#ifdef WIN32
  const char dlmtr = '\\';
#else
  const char dlmtr = '/';
#endif

  trainFilenames.clear();

  ifstream file( filename.c_str() );
  if ( !file.is_open() )
    return;

  size_t pos = filename.rfind(dlmtr);
  dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

  int iImgCols = -1;
  int iImgRows = -1;
  int iImgCount = -1;

  int iCurrImage = 0;

  // Vector for reading in transforms
  vector< float > vfTrans;
  vfTrans.resize(12);

  // Read until get dimensions
  while( !file.eof() )
  {
    string str; getline( file, str );
    if( str.empty() ) break;
    char *pch = &(str[0]);
    if( !pch )
      return;

    if( strstr( pch, "DimSize =" ) )
    {
      if( sscanf( pch, "DimSize = %d %d %d", &iImgCols, &iImgRows, &iImgCount ) != 3 )
      {
        printf( "Error: could not read dimensions\n" );
        return;
      }
    }
    if( strstr( pch, "ProbeToTrackerTransform =" )
      || strstr( pch, "UltrasoundToTrackerTransform =" ) )
    {
       // Parse name and transform
       // Seq_Frame0000_ProbeToTrackerTransform = -0.224009 -0.529064 0.818481 212.75 0.52031 0.6452 0.559459 -14.0417 -0.824074 0.551188 0.130746 -26.1193 0 0 0 1 

      char *pcName = pch;
      char *pcTrans = strstr( pch, "=" );
      pcTrans[-1] = 0; // End file name string pcName
       //pcTrans++; // Increment to just after equal sign

      string filename = dirName + pcName + ".png";// + pcTrans;
      trainFilenames.push_back( filename );

      char *pch = pcTrans;

      for( int j =0; j < 12; j++ )
      {
        pch = strchr( pch + 1, ' ' );
        if( !pch )
          return;
        vfTrans[j] = atof( pch );
        pch++;
      }
      trainImageData.push_back( vfTrans );
    }
    if( strstr( pch, "ElementDataFile = LOCAL" ) )
    {
       // Done reading
      break;
    }
  }

  int iPosition = file.tellg();
  file.close();

  FILE *infile = fopen( filename.c_str(), "rb" );
  //fseek( infile, iPosition, SEEK_SET );
  char buffer[400];
  while( fgets( buffer, 400, infile ) )
  {
    if( strstr( buffer, "ElementDataFile = LOCAL" ) )
    {
      // Done reading
      break;
    }
  }
  int iPos2 = ftell( infile );

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
    else if(i>=firstFrame && i <= lastFrame) {
      Mat mtImgNew = mtImg.clone();
      fread( mtImgNew.data, 1, iImgRows*iImgCols, infile );
      trainImages.push_back( mtImgNew );
      //imwrite( trainFilenames[i], trainImages[i] );
      //imwrite( trainFilenames[i], mtImgNew );
    }
    else if(i>lastFrame)
      break;
  }
  
  // Keep the transforms that are in the first-last range
  trainFilenames = std::vector<std::string>(trainFilenames.begin()+firstFrame, trainFilenames.begin()+lastFrame+1);
  trainImageData = std::vector<std::vector<float> >(trainImageData.begin()+firstFrame, trainImageData.begin()+lastFrame+1);

  for( int i = 0; i < iImgCount; i++ )
  {
    //imwrite( trainFilenames[i], trainImages[i] );
  }


  delete [] pucImgData;

  fclose( infile );

  printf( "Read images: %d\n", trainImages.size() );
  printf( "Read data  : %d\n", trainImageData.size() );
  printf( "Read names : %d\n", trainFilenames.size() );

}

//
// splitFile_mha()
//
// Split big mha file into several parts
//
void splitFile_mha(
  const string& filename,
  int iParts = 2
  )
{
#ifdef WIN32
  const char dlmtr = '\\';
#else
  const char dlmtr = '/';
#endif

  ifstream file( filename.c_str() );
  if ( !file.is_open() )
    return;

  string dirName;

  size_t pos = filename.rfind(dlmtr);
  dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

  int iImgCols = -1;
  int iImgRows = -1;
  int iImgCount = -1;

  int iCurrImage = 0;

  // Vector for reading in transforms
  vector< float > vfTrans;
  vfTrans.resize(12);

  string fname1 = filename + "1.mha";
  string fname2 = filename + "2.mha";
  FILE *file1 = fopen( &fname1[0], "wb" );
  FILE *file2 = fopen( &fname2[0], "wb" );
  int iCount1;
  int iCount2;
  int bDone = 0;

  // For December AMIGO data (neuro-case/anonymized)
  // No intra-op, used for calibration
  int iStartRow = 124;
  int iStartCol = 265;
  int iRows = 640;
  int iCols = 690;

  // For January AMIGO data (20130108-us-anon\TrackedImageSequence_20130108_121528.mha)
  // 
  // For some reason there is an anoying shift ... 
  iStartRow = 140;
  iStartCol = 520;
  iRows = 750;
  iCols = 820;

  // For January AMIGO data (20130108-us-anon\TrackedImageSequence_20130108_132324.mha)
  // 
  // For some reason there is an anoying shift ... 
  iStartRow = 140;
  iStartCol = 520;

  // Read until get dimensions
  while( !file.eof() && !bDone )
  {
    string str;
    getline( file, str );
    if( str.empty() ) break;
    char *pch = &(str[0]);
    if( !pch )
      return;

    int iCurrFrame = -1;

    if( iImgCount > -1 )
    {
      if( sscanf( pch, "Seq_Frame%d", &iCurrFrame ) != 1 )
      {
        iCurrFrame = -1;
      }
      else
      {
        printf( "Frame: %d\n", iCurrFrame );
      }
    }

    if( strstr( pch, "DimSize =" ) )
    {
      if( sscanf( pch, "DimSize = %d %d %d", &iImgCols, &iImgRows, &iImgCount ) != 3 )
      {
        printf( "Error: could not read dimensions\n" );
        return;
      }
      iCount1 = iImgCount / iParts;
      iCount2 = iImgCount - iCount1;

       // Special write operation
       // Consider cropped images
      fprintf( file1, "DimSize = %d %d %d",iCols,iRows,iCount1);
      fprintf( file2, "DimSize = %d %d %d",iCols,iRows,iCount2);
      continue;
    }
     //if( strstr( pch, "ProbeToTrackerTransform =" )
      // || strstr( pch, "UltrasoundToTrackerTransform =" ) )
     //{
      // // Parse name and transform
      // // Seq_Frame0000_ProbeToTrackerTransform = -0.224009 -0.529064 0.818481 212.75 0.52031 0.6452 0.559459 -14.0417 -0.824074 0.551188 0.130746 -26.1193 0 0 0 1 
     //}
    if( strstr( pch, "ElementDataFile = LOCAL" ) )
    {
       // Done reading
      bDone = 1;
    }

     // Output
    if( iCurrFrame < 0 )
    {
      fwrite( &str[0], 1, str.size(), file1 );fprintf( file1, "\n" );
      fwrite( &str[0], 1, str.size(), file2 );fprintf( file2, "\n" );
    }
    else if( iCurrFrame < iCount1 )
    {
      fwrite( &str[0], 1, str.size(), file1 );fprintf( file1, "\n" );
    }
    else
    {
      fwrite( &str[0], 1, str.size(), file2 );fprintf( file2, "\n" );
    }
  }

  int iPosition = file.tellg();
  file.close();

  //fclose( file1 );
  //fclose( file2 );

  FILE *infile = fopen( filename.c_str(), "rb" );
  fseek( infile, iPosition, SEEK_SET );

  unsigned char *pucImgData = new unsigned char[iImgRows*iImgCols];
  Mat mtImg(iImgRows, iImgCols, CV_8UC1, pucImgData);

  unsigned char *pucImgData2 = new unsigned char[iRows*iCols];
  Mat mtImg2(iRows, iCols, CV_8UC1, pucImgData2);

  // Read & write images
  for( int i = 0; i < iImgCount; i++ )
  {
    fread( mtImg.data, 1, iImgRows*iImgCols, infile );
    //imwrite( "image_big.png", mtImg );

    for( int r = 0; r < iRows; r++ )
    {
      memcpy( mtImg2.data+r*iCols, mtImg.data + (iStartRow + r)*iImgCols + iStartCol, iCols );
    }
    //imwrite( "image_small.png", mtImg2 );

    if( i < iCount1 )
    {
      for( int r = 0; r < iRows; r++ )
      {
        //fwrite( mtImg.data + (iStartRow + r)*iImgCols + iStartCol, 1, iCols, file1 );
        fwrite( mtImg2.data+r*iCols, 1, iCols, file1 );
      }
    }
    else
    {
      for( int r = 0; r < iRows; r++ )
      {
        //fwrite( mtImg.data + (iStartRow + r)*iImgCols + iStartCol, 1, iCols, file2 );
        fwrite( mtImg2.data+r*iCols, 1, iCols, file2 );
      }
    }
    //imwrite( trainFilenames[i], mtImg );
  }

  fclose( file1 );
  fclose( file2 );

  delete [] pucImgData;

  fclose( infile );

}

//
// convertToPng_mha()
//
// Convert mha file into png images
//
void
  convertToPng_mha(
  const string& filename,
  const string& outputCode
  )
{
#ifdef WIN32
  const char dlmtr = '\\';
#else
  const char dlmtr = '/';
#endif

  vector<string> trainFilenames;
  vector< vector<float> > trainImageData;

  trainFilenames.clear();

  ifstream file( filename.c_str() );
  if ( !file.is_open() )
    return;

  size_t pos = filename.rfind(dlmtr);
  string dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

  int iImgCols = -1;
  int iImgRows = -1;
  int iImgCount = -1;

  int iCurrImage = 0;

  // Vector for reading in transforms
  vector< float > vfTrans;
  vfTrans.resize(12);

  // Read until get dimensions
  while( !file.eof() )
  {
    string str; getline( file, str );
    if( str.empty() ) break;
    char *pch = &(str[0]);
    if( !pch )
      return;

    if( strstr( pch, "DimSize =" ) )
    {
      if( sscanf( pch, "DimSize = %d %d %d", &iImgCols, &iImgRows, &iImgCount ) != 3 )
      {
        printf( "Error: could not read dimensions\n" );
        return;
      }
    }
    if( strstr( pch, "ProbeToTrackerTransform =" )
      || strstr( pch, "UltrasoundToTrackerTransform =" ) )
    {
       // Parse name and transform
       // Seq_Frame0000_ProbeToTrackerTransform = -0.224009 -0.529064 0.818481 212.75 0.52031 0.6452 0.559459 -14.0417 -0.824074 0.551188 0.130746 -26.1193 0 0 0 1 

      char *pcName = pch;
      char *pcTrans = strstr( pch, "=" );
      pcTrans[-1] = 0; // End file name string pcName
       //pcTrans++; // Increment to just after equal sign

      string filename = dirName + outputCode + pcName + ".png";// + pcTrans;
      trainFilenames.push_back( filename );

      char *pch = pcTrans;

      for( int j =0; j < 12; j++ )
      {
        pch = strchr( pch + 1, ' ' );
        if( !pch )
          return;
        vfTrans[j] = atof( pch );
        pch++;
      }
      trainImageData.push_back( vfTrans );
    }
    if( strstr( pch, "ElementDataFile = LOCAL" ) )
    {
       // Done reading
      break;
    }
  }

  int iPosition = file.tellg();
  file.close();

  FILE *infile = fopen( filename.c_str(), "rb" );
  fseek( infile, iPosition, SEEK_SET );

  unsigned char *pucImgData = new unsigned char[iImgRows*iImgCols];
  Mat mtImg(iImgRows, iImgCols, CV_8UC1, pucImgData);

  // Read & write images
  for( int i = 0; i < iImgCount; i++ )
  {
    fread( mtImg.data, 1, iImgRows*iImgCols, infile );
    //imwrite( trainFilenames[i], trainImages[i] );
    imwrite( trainFilenames[i], mtImg );
  }

  delete [] pucImgData;

  fclose( infile );

}

//
// convertToPngToMHA_mha()
//
// Store pngs into an mha file.
//
void
  convertToPngToMHA_mha(
  const string& filename,
  const string& outputCode
  )
{
#ifdef WIN32
  const char dlmtr = '\\';
#else
  const char dlmtr = '/';
#endif

  vector<string> trainFilenames;
  vector< vector<float> > trainImageData;

  trainFilenames.clear();

  ifstream file( filename.c_str() );
  if ( !file.is_open() )
    return;

  size_t pos = filename.rfind(dlmtr);
  string dirName = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;

  int iImgCols = -1;
  int iImgRows = -1;
  int iImgCount = -1;

  int iCurrImage = 0;

  // Vector for reading in transforms
  vector< float > vfTrans;
  vfTrans.resize(12);

  // Read until get dimensions
  while( !file.eof() )
  {
    string str; getline( file, str );
    if( str.empty() ) break;
    char *pch = &(str[0]);
    if( !pch )
      return;

    if( strstr( pch, "DimSize =" ) )
    {
      if( sscanf( pch, "DimSize = %d %d %d", &iImgCols, &iImgRows, &iImgCount ) != 3 )
      {
        printf( "Error: could not read dimensions\n" );
        return;
      }
    }
    if( strstr( pch, "ProbeToTrackerTransform =" )
      || strstr( pch, "UltrasoundToTrackerTransform =" ) )
    {
       // Parse name and transform
       // Seq_Frame0000_ProbeToTrackerTransform = -0.224009 -0.529064 0.818481 212.75 0.52031 0.6452 0.559459 -14.0417 -0.824074 0.551188 0.130746 -26.1193 0 0 0 1 

      char *pcName = pch;
      char *pcTrans = strstr( pch, "=" );
      pcTrans[-1] = 0; // End file name string pcName
       //pcTrans++; // Increment to just after equal sign

      string filename = dirName + outputCode + pcName + ".png";// + pcTrans;
      trainFilenames.push_back( filename );

      char *pch = pcTrans;

      for( int j =0; j < 12; j++ )
      {
        pch = strchr( pch + 1, ' ' );
        if( !pch )
          return;
        vfTrans[j] = atof( pch );
        pch++;
      }
      trainImageData.push_back( vfTrans );
    }
    if( strstr( pch, "ElementDataFile = LOCAL" ) )
    {
       // Done reading
      break;
    }
  }

  int iPosition = file.tellg();
  file.close();

  FILE *infile = fopen( filename.c_str(), "rb" );
  fseek( infile, iPosition, SEEK_SET );

  unsigned char *pucImgData = new unsigned char[iImgRows*iImgCols];
  Mat mtImg(iImgRows, iImgCols, CV_8UC1, pucImgData);

  // Read & write images
  for( int i = 0; i < iImgCount; i++ )
  {
    fread( mtImg.data, 1, iImgRows*iImgCols, infile );
    //imwrite( trainFilenames[i], trainImages[i] );
    imwrite( trainFilenames[i], mtImg );
  }

  delete [] pucImgData;

  fclose( infile );

}



bool createDetectorDescriptorMatcher( const string& detectorType, const string& descriptorType, const string& matcherType,
  Ptr<FeatureDetector>& featureDetector,
  Ptr<DescriptorExtractor>& descriptorExtractor,
  Ptr<DescriptorMatcher>& descriptorMatcher )
{
  cout << "< Creating feature detector..." << endl;
  //cout << " " << detectorType <<" " << featureDetector << endl; 
  featureDetector = FeatureDetector::create( detectorType );
  cout << "descriptor extractor..." << endl;
  descriptorExtractor = DescriptorExtractor::create( descriptorType );
  cout << "descriptor matcher ..." << endl;
  descriptorMatcher = DescriptorMatcher::create( matcherType );
  cout << "done >" << endl;

  bool isCreated = !( featureDetector.empty() || descriptorExtractor.empty() || descriptorMatcher.empty() );
  if( !isCreated )
    cout << "Can not create feature detector or descriptor extractor or descriptor matcher of given types." << endl << ">" << endl;

  return isCreated;
}

bool readImages( const string& queryImageName, const string& trainFilename,
  Mat& queryImage, vector <Mat>& trainImages, vector<string>& trainImageNames )
{
  cout << "< Reading the images..." << endl;
  queryImage = imread( queryImageName, CV_LOAD_IMAGE_GRAYSCALE);
  if( queryImage.empty() )
  {
    cout << "Query image can not be read." << endl << ">" << endl;
    return false;
  }
  string trainDirName;
  readTrainFilenames( trainFilename, trainDirName, trainImageNames );
  if( trainImageNames.empty() )
  {
    cout << "Train image filenames can not be read." << endl << ">" << endl;
    return false;
  }
  int readImageCount = 0;
  for( size_t i = 0; i < trainImageNames.size(); i++ )
  {
    string filename = trainDirName + trainImageNames[i];
    Mat img = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
    if( img.empty() )
      cout << "Train image " << filename << " can not be read." << endl;
    else
      readImageCount++;
    trainImages.push_back( img );
  }
  if( !readImageCount )
  {
    cout << "All train images can not be read." << endl << ">" << endl;
    return false;
  }
  else
    cout << readImageCount << " train images were read." << endl;
  cout << ">" << endl;

  return true;
}

bool
  readImageList(
  const string& trainFilename,
  vector <Mat>& trainImages,
  vector<string>& trainImageNames,
  vector< vector<float> >& trainImageData
  )
{
  string trainDirName;
  readTrainFilenames( trainFilename, trainDirName, trainImageNames );
  if( trainImageNames.empty() )
  {
    cout << "Train image filenames can not be read." << endl << ">" << endl;
    return false;
  }
  int readImageCount = 0;
  trainImageData.resize( trainImageNames.size() );
  for( size_t i = 0; i < trainImageNames.size(); i++ )
  {
    // Read in affine transform parameters
    trainImageData[i].resize( 12 );
    // Look for first tab after file name
    char *pch = &(trainImageNames[i][0]);
    if( !pch ) return false;

    // Read past name
    pch = strchr( pch, '\t' );
    if( !pch ) return false;
    pch++;

    // Read past index
    pch = strchr( pch, '\t' );
    if( !pch ) return false;
    pch++;
    for( int j =0 ;j < 12; j++ )
    {
      while( pch[0] == ' ' || pch[0] == '\t' )
        pch++;
      //pch = strchr( pch, '\t' );
      //if( !pch ) return false;
      //pch++;
      trainImageData[i][j] = atof( pch );		
      while( pch[0] != ' ' && pch[0] != '\t' )
        pch++;

    }
    pch = &(trainImageNames[i][0]);
    pch = strchr( pch, '\t' );
    if( pch ) *pch = 0;
    pch = &(trainImageNames[i][0]);
    pch = strstr( pch, ".crop.png" );
    if( pch ) *pch = 0;

    string filename = trainDirName + trainImageNames[i];


    Mat img = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
    if( img.empty() )
      cout << "Train image " << filename << " can not be read." << endl;
    else
      readImageCount++;
    trainImages.push_back( img );
  }
  if( !readImageCount )
  {
    cout << "All train images can not be read." << endl << ">" << endl;
    return false;
  }
  else
    cout << readImageCount << " train images were read." << endl;
  cout << ">" << endl;

  return true;
}

bool
  readImageList_mha(
  const string& trainFilename,
  vector <Mat>& trainImages,
  vector<string>& trainImageNames,
  vector< vector<float> >& trainImageData,
  int& firstFrame,
  int& lastFrame
  )
{
  string trainDirName;
  readTrainTransforms_mha( trainFilename, trainDirName, trainImageNames, trainImages, trainImageData, firstFrame, lastFrame);
  int readImageCount = trainImages.size();

  if( !readImageCount )
  {
    cout << "All train images can not be read." << endl << ">" << endl;
    return false;
  }
  else
    cout << readImageCount << " train images were read." << endl;
  cout << ">" << endl;

  return true;
}



bool
  
  DetectCompute(
  const string& trainFilename,
  vector <Mat>& trainImages,
  vector<string>& trainImageNames,
  Ptr<FeatureDetector>& featureDetector,
  Ptr<DescriptorExtractor>& descriptorExtractor,
  vector<vector<KeyPoint> > &trainKeypoints,
  vector<Mat> &trainDescriptors,
  Mat &maskImage
  )
{
  string trainDirName;
  readTrainFilenames( trainFilename, trainDirName, trainImageNames );
  if( trainImageNames.empty() )
  {
    cout << "Train image filenames can not be read." << endl << ">" << endl;
    return false;
  }
  int readImageCount = 0;

  trainKeypoints.resize( trainImageNames.size() );
  for( size_t i = 0; i < trainImageNames.size(); i++ )
  {
    string filename = trainDirName + trainImageNames[i];
    Mat img = imread( filename, CV_LOAD_IMAGE_GRAYSCALE );
    if( img.empty() )
      cout << "Train image " << filename << " can not be read." << endl;
    else
      readImageCount++;
    trainImages.push_back( img );

    char *pch = strchr( &filename[0], '.' );
    if( !pch )
    {
      printf( "Error: strange file name, no image extension\n" );
      return false;
    }

    featureDetector->detect( img, trainKeypoints[i], maskImage );
  }
  descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );

  if( !readImageCount )
  {
    cout << "All train images can not be read." << endl << ">" << endl;
    return false;
  }
  else
    cout << readImageCount << " train images were read." << endl;
  cout << ">" << endl;

  return true;
}

void detectKeypoints( const Mat& queryImage, vector<KeyPoint>& queryKeypoints,
  const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints,
  Ptr<FeatureDetector>& featureDetector )
{
  cout << endl << "< Extracting keypoints from images..." << endl;
  featureDetector->detect( queryImage, queryKeypoints );
  featureDetector->detect( trainImages, trainKeypoints );
  cout << ">" << endl;
}

void computeDescriptors( const Mat& queryImage, vector<KeyPoint>& queryKeypoints, Mat& queryDescriptors,
  const vector<Mat>& trainImages, vector<vector<KeyPoint> >& trainKeypoints, vector<Mat>& trainDescriptors,
  Ptr<DescriptorExtractor>& descriptorExtractor )
{
  cout << "< Computing descriptors for keypoints..." << endl;
  descriptorExtractor->compute( queryImage, queryKeypoints, queryDescriptors );
  descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );
  cout << ">" << endl;
}

void matchDescriptors( const Mat& queryDescriptors, vector<DMatch>& matches, Ptr<DescriptorMatcher>& descriptorMatcher )
{
  // Assumes training descriptors have already been added to descriptorMatcher
  cout << "< Set train descriptors collection in the matcher and match query descriptors to them..." << endl;
    //descriptorMatcher->add( trainDescriptors );
  descriptorMatcher->match( queryDescriptors, matches );
  CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );
  cout << ">" << endl;
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
//
//static void _prepareImgs( const Mat& img1, const vector<KeyPoint>& keypoints1,
//                                         const Mat& img2, const vector<KeyPoint>& keypoints2,
//                                         Mat& outImg, Mat& outImg1, Mat& outImg2,
//                                         const Scalar& singlePointColor, int flags )
//{
//    Size size( img1.cols + img2.cols, MAX(img1.rows, img2.rows) );
//    if( flags & DrawMatchesFlags::DRAW_OVER_OUTIMG )
//    {
//        if( size.width > outImg.cols || size.height > outImg.rows )
//            CV_Error( CV_StsBadSize, "outImg has size less than need to draw img1 and img2 together" );
//        outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
//        outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
//    }
//    else
//    {
//        outImg.create( size, CV_MAKETYPE(img1.depth(), 3) );
//        outImg1 = outImg( Rect(0, 0, img1.cols, img1.rows) );
//        outImg2 = outImg( Rect(img1.cols, 0, img2.cols, img2.rows) );
//
//        if( img1.type() == CV_8U )
//            cvtColor( img1, outImg1, CV_GRAY2BGR );
//        else
//            img1.copyTo( outImg1 );
//
//        if( img2.type() == CV_8U )
//            cvtColor( img2, outImg2, CV_GRAY2BGR );
//        else
//            img2.copyTo( outImg2 );
//    }
//}
//
//void drawMatches_here( const Mat& img1, const vector<KeyPoint>& keypoints1,
//                  const Mat& img2, const vector<KeyPoint>& keypoints2,
//                  const vector<DMatch>& matches1to2, Mat& outImg,
//                  const Scalar& matchColor, const Scalar& singlePointColor,
//                  const vector<char>& matchesMask, int flags )
//{
//    if( !matchesMask.empty() && matchesMask.size() != matches1to2.size() )
//        CV_Error( CV_StsBadSize, "matchesMask must have the same size as matches1to2" );
//
//    Mat outImg1, outImg2;
//    _prepareImgs( img1, keypoints1, img2, keypoints2,
//                                 outImg, outImg1, outImg2, singlePointColor, flags );
//
//    // draw matches
//    for( size_t m = 0; m < matches1to2.size(); m++ )
//    {
//        int i1 = matches1to2[m].queryIdx;
//        int i2 = matches1to2[m].trainIdx;
//        if( matchesMask.empty() || matchesMask[m] )
//        {
//            const KeyPoint &kp1 = keypoints1[i1], &kp2 = keypoints2[i2];
//            _drawMatch( outImg, outImg1, outImg2, kp1, kp2, matchColor, flags );
//        }
//    }
//}

void saveResultImage( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
  const Mat& trainImage, const vector<KeyPoint>& trainKeypoints,
  const vector<DMatch>& matches,
  int iTrainImageIndex,
  const string& outputImageName,
  const string& resultDir
  )
{
  cout << "< Save results..." << endl;
  Mat drawImg;
  vector<char> mask;
  mask.resize( matches.size() );
  fill( mask.begin(), mask.end(), 0 );
  for( size_t i = 0; i < matches.size(); i++ )
  {
    if( abs( matches[i].imgIdx - iTrainImageIndex ) <= 1 )
      mask[i] = 1;
  }

  maskMatchesByTrainImgIdx( matches, iTrainImageIndex, mask );
  maskMatchesByMatchPresent( matches, mask );
  drawMatches( queryImage, queryKeypoints, trainImage, trainKeypoints,
    matches, drawImg, Scalar::all(-1), Scalar::all(-1), mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
  int iLast = outputImageName.find_last_of( "\\" );
  int iSize  = outputImageName.size();
  string fname = outputImageName.substr( iLast + 1, iSize-iLast-1);
  string filename = resultDir + "\\single_res_" + fname;
  imwrite( filename, drawImg );
}

void
  saveResultData(
  FILE *outfile1,
  FILE *outfile2,
  vector<string>& queryImagesNames,
  vector<vector<KeyPoint> >& queryKeypoints,
  vector<string>& trainImagesNames,
  vector<vector<KeyPoint> >& trainKeypoints,
  vector<DMatch>& matches,
  int iQueryImageIndex
  )
{
  string& queryName = queryImagesNames[iQueryImageIndex];
  vector<KeyPoint>& queryKeys = queryKeypoints[iQueryImageIndex];

  for( int i = 0; i < matches.size(); i++ )
  {
    if( matches[i].imgIdx >= 0 )
    {
      KeyPoint &queryKey = queryKeys[matches[i].queryIdx];
      KeyPoint &trainKey = trainKeypoints[matches[i].imgIdx][matches[i].trainIdx];

      string& trainName = trainImagesNames[matches[i].imgIdx];

      fprintf( outfile1, "%d|%s 0 \%f \%f|%s 0 %f %f\n",
        matches.size(),
        queryName.c_str(), queryKey.pt.y, queryKey.pt.x,
        trainName.c_str(), trainKey.pt.y, trainKey.pt.x );
    }
  }
  return;
}

void saveResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
  const vector<Mat>& trainImages, const vector<vector<KeyPoint> >& trainKeypoints,
  const vector<DMatch>& matches, const vector<string>& trainImagesNames, const string& resultDir,
  int iIndex = 0 )
{
  cout << "< Save results..." << endl;
  Mat drawImg;
  vector<char> mask;
  int iMatches;
  for( size_t i = 0; i < trainImages.size(); i++ )
  {
    if( !trainImages[i].empty() )
    {
      iMatches = maskMatchesByTrainImgIdx( matches, i, mask );
      iMatches = maskMatchesByMatchPresent( matches, mask );
      if( iMatches <= 0 )
        continue;
      drawMatches( queryImage, queryKeypoints, trainImages[i], trainKeypoints[i],
        matches, drawImg, Scalar::all(-1), Scalar::all(-1), mask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
      int iLast = trainImagesNames[i].find_last_of( "\\" );
      int iSize  = trainImagesNames[i].size();
      string fname = trainImagesNames[i].substr( iLast + 1, iSize-iLast-1);
      char pcIndex[10];
      sprintf( pcIndex, "%3.3d_", iIndex );
      string filename = resultDir + "\\res_" + pcIndex + fname;
      if( !imwrite( filename, drawImg ) )
        cout << "Image " << filename << " can not be saved (may be because directory " << resultDir << " does not exist)." << endl;

//			Mat outImg1, outImg2;
//			outImg1 = drawImg( Rect(0, 0, queryImage.cols, queryImage.rows) );
//			outImg2 = drawImg( Rect(queryImage.cols, 0, trainImages[i].cols, trainImages[i].rows) );

   //         string filename1 = resultDir + "\\res_1_" + fname + "_1.png";
      //imwrite( filename1, outImg1 );
   //         string filename2 = resultDir + "\\res_2_" + fname + "_2.png";
      //imwrite( filename2, outImg2 );


    }
  }
  cout << ">" << endl;
}

//
// saveResultImages()
//
// Output multiple matches.
//
void saveResultImages( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,
  const vector<Mat>& trainImages, const vector<vector<KeyPoint> >& trainKeypoints,
  const vector< vector<DMatch> >& mmatches, const vector<string>& trainImagesNames, const string& resultDir )
{
  cout << "< Save results..." << endl;

  vector<DMatch> matches;
  matches.resize( mmatches.size() );
  for( int i = 0; i < mmatches.size(); i++ )
  {
    matches[i] = mmatches[i][0];
  }

  saveResultImages( queryImage, queryKeypoints, trainImages, trainKeypoints,
    matches, trainImagesNames, resultDir );

   // Mat drawImg;
   // vector<char> mask;
   // for( size_t i = 0; i < trainImages.size(); i++ )
   // {
   //     if( !trainImages[i].empty() )
   //     {
   //         //maskMatchesByTrainImgIdx( matches, i, mask );
   //         drawMatches( queryImage, queryKeypoints, trainImages[i], trainKeypoints[i],
   //                      matches, drawImg, Scalar::all(-1), Scalar::all(-1), mask );
      //int iLast = trainImagesNames[i].find_last_of( "\\" );
      //int iSize  = trainImagesNames[i].size();
      //string fname = trainImagesNames[i].substr( iLast + 1, iSize-iLast-1);
   //         string filename = resultDir + "\\res_" + fname;
   //         if( !imwrite( filename, drawImg ) )
   //             cout << "Image " << filename << " can not be saved (may be because directory " << resultDir << " does not exist)." << endl;
   //     }
   // }
  cout << ">" << endl;
}

//
// applyImageMask()
//
// Apply binary mask.
//
void
  applyImageMask(
  const Mat & maskImage,
  vector<Mat>& vecImages
  )
{
  for( int i = 0; i < vecImages.size(); i++ )
  {
    for( int y = 0; y < vecImages[i].rows; y++ )
    {
      for( int x = 0; x < vecImages[i].cols; x++ )
      {
        if( maskImage.data[ y*maskImage.step + x ] <= 0 )
        {
          vecImages[i].data[ y*vecImages[i].step + x ] = 0;
        }
      }
    }
  }
}

int
  _readDescriptorDatabase(
  char *pcFileWithImages,
  vector<Mat> &trainImages,
  vector<string> &trainImagesNames,
  vector< vector<float> > &trainImagesTransform,
  vector<vector<KeyPoint> > &trainKeypoints,
  vector<Mat> &trainDescriptors
  )
{
  string fileWithTrainImages = pcFileWithImages;

  char *pch = &(fileWithTrainImages[0]);
  if( !pch ) return false;
  if( strstr( pch, ".mha" ) )
  {
    int firstFrame=0, lastFrame=-1;
    if( !readImageList_mha( fileWithTrainImages, trainImages, trainImagesNames, trainImagesTransform, firstFrame, lastFrame ) )
    {
      printf( "Error: could not read training images\n" );
      return -1;
    }
  }
  else if( !readImageList( fileWithTrainImages, trainImages, trainImagesNames, trainImagesTransform ) )
  {
    printf( "Error: could not read training images\n" );
    return -1;
  }

  pch = strstr( &fileWithTrainImages[0], ".txt" );
  if( !pch )
  {
    pch = strstr( &fileWithTrainImages[0], ".mha" );

    if( !pch )
    {
      printf( "Error: strange file name, no extension .txt\n" );
      return false;
    }
  }

  fileWithTrainImages += "extra extra extra space";
  pch = strrchr( &fileWithTrainImages[0], '.' );

  FILE *outfile;
  int iValue;


  sprintf( pch, "%s", ".surf.key" );
  outfile = fopen( &fileWithTrainImages[0], "rb" );
  if( !outfile )
  {
    printf( "Error: no key file found\n" );
    return false;
  }
  fread( &iValue, sizeof(iValue), 1, outfile );	
  assert( iValue == trainImages.size() );
  trainKeypoints.resize( iValue ); // Number of vector<Keypoint>/Images
  for( int i = 0; i < trainKeypoints.size(); i++ )
  {
    fread( &iValue, sizeof(iValue), 1, outfile );
    trainKeypoints[i].resize(iValue); // Size of vector<Keypoint>
    for( int j = 0; j < trainKeypoints[i].size(); j++ )
    {
      fread( &(trainKeypoints[i][j]), sizeof(trainKeypoints[i][j]), 1, outfile );
    }
  }
  fclose( outfile );

  int iReturn;
  sprintf( pch, "%s", ".surf.dsc" );
  outfile = fopen( &fileWithTrainImages[0], "rb" );
  fread( &iValue, sizeof(iValue), 1, outfile );
  assert( iValue == trainImages.size() );
  trainDescriptors.resize(iValue); // Number of Mat/Images
  for( int i = 0; i < trainDescriptors.size(); i++ )
  {
    int iRows;
    int iCols;
    int iType;
    fread( &iRows, sizeof(iRows), 1, outfile );
    fread( &iCols, sizeof(iCols), 1, outfile );
    fread( &iType, sizeof(iType), 1, outfile );
    if( iRows > 0 && iCols > 0 )
    {
      assert( iType == CV_32FC1 );
      trainDescriptors[i].create( iRows, iCols, iType );
      iReturn = fread( trainDescriptors[i].data, sizeof(float), iRows*iCols, outfile );
    }
    else
    {
      printf( "Zero feature image: %d\n", i );
    }
  }
  fclose( outfile );

  return 0;
}


//
// main_prepareDescriptorDatabase()
//
// Produce & save descriptor database.
//
int
  main_prepareDescriptorDatabase(
  char *pcDetectorType,
  char *pcDescriptorType,
  char *pcFileWithImages,
  char *pcMaskImage
  )
{
  string detectorType = defaultDetectorType;
  string descriptorType = defaultDescriptorType;
  string matcherType = defaultMatcherType;
  string fileWithQueryImages = defaultQueryImageName;
  string fileWithTrainImages = defaultFileWithTrainImages;
  string fileWithBogusImages = "";
  string fileMaskImage = "";
  string dirToSaveResImages = defaultDirToSaveResImages;

  detectorType = pcDetectorType;
  descriptorType = pcDescriptorType;
  fileWithTrainImages = pcFileWithImages;
  fileMaskImage = pcMaskImage;

  Ptr<FeatureDetector> featureDetector;
  Ptr<DescriptorExtractor> descriptorExtractor;
  Ptr<DescriptorMatcher> descriptorMatcher;
  Ptr<DescriptorMatcher> descriptorMatcherBogus;

  printf( "Creating detector/descriptor/matcher: %s %s %s\n", detectorType.data(), descriptorType.data(), matcherType.data() );
  if( !createDetectorDescriptorMatcher( detectorType, descriptorType, matcherType, featureDetector, descriptorExtractor, descriptorMatcher ) )
  {
    printf( "Error: could not create descriptor extractor matcher\n" );
    return false;
  }

  printf( "Reading mask: %s\n", fileMaskImage.data() );
  Mat maskImage = imread( fileMaskImage, CV_LOAD_IMAGE_GRAYSCALE);
  if( maskImage.empty() )
  {
    cout << "Mask image can not be read:" << fileMaskImage << endl << ">" << endl;
    return false;
  }

  vector<Mat> trainImages;
  vector<string> trainImagesNames;
  vector<vector<float> > trainTransforms;
  vector<vector<KeyPoint> > trainKeypoints;
  vector<Mat> trainDescriptors;

  printf( "Reading image list: %s\n", fileWithTrainImages.data() );

  // Look for first tab after file name
  char *pch = &(fileWithTrainImages[0]);
  if( !pch ) return false;
  if( strstr( pch, ".mha" ) )
  {
    int firstFrame=0, lastFrame=-1;
    if( !readImageList_mha( fileWithTrainImages, trainImages, trainImagesNames, trainTransforms, firstFrame, lastFrame ) )
    {
      printf( "Error: could not read training images\n" );
      return -1;
    }
  }
  else if( !readImageList( fileWithTrainImages, trainImages, trainImagesNames, trainTransforms ) )
  {
    printf( "Error: could not read training images\n" );
    return -1;
  }

  trainKeypoints.resize( trainImages.size() );
  for( int i = 0; i < trainImages.size(); i++ )
  {
    printf( "%d\n", i );
    featureDetector->detect( trainImages[i], trainKeypoints[i], maskImage );
  }
  descriptorExtractor->compute( trainImages, trainKeypoints, trainDescriptors );

  // Original text file input (AMIGO)
  pch = strstr( &fileWithTrainImages[0], ".txt" );
  if( !pch )
  {
    // PLUS mha file input
    pch = strstr( &fileWithTrainImages[0], ".mha" );
  }
  if( !pch )
  {
    printf( "Error: strange file name, no extension .txt\n" );
    return false;
  }
  pch = strrchr( &fileWithTrainImages[0], '.' );

  FILE *outfile;
  int iValue;

  sprintf( pch, "%s", ".surf.key" );
  outfile = fopen( &fileWithTrainImages[0], "wb" );
  iValue = trainKeypoints.size(); // Number of vector<Keypoint>/Images
  fwrite( &iValue, sizeof(iValue), 1, outfile );
  for( int i = 0; i < trainKeypoints.size(); i++ )
  {
    int iValue = trainKeypoints[i].size(); // Size of vector<Keypoint>
    fwrite( &iValue, sizeof(iValue), 1, outfile );
    for( int j = 0; j < trainKeypoints[i].size(); j++ )
    {
      fwrite( &(trainKeypoints[i][j]), sizeof(trainKeypoints[i][j]), 1, outfile );
    }
  }
  fclose( outfile );

  int iReturn;
  sprintf( pch, "%s", ".surf.dsc" );
  outfile = fopen( &fileWithTrainImages[0], "wb" );
  iValue = trainDescriptors.size(); // Number of Mat/Images
  fwrite( &iValue, sizeof(iValue), 1, outfile );
  for( int i = 0; i < trainDescriptors.size(); i++ )
  {
    int iRows = trainDescriptors[i].rows; // Size of vector<Keypoint>
    int iCols = trainDescriptors[i].cols; // Size of vector<Keypoint>
    int iType = trainDescriptors[i].type(); // Size of vector<Keypoint>
    assert( iType == CV_32FC1 );
    fwrite( &iRows, sizeof(iRows), 1, outfile );
    fwrite( &iCols, sizeof(iCols), 1, outfile );
    fwrite( &iType, sizeof(iType), 1, outfile );
    iReturn = fwrite( trainDescriptors[i].data, sizeof(float), iRows*iCols, outfile );
  }
  fclose( outfile );


  vector<Mat> trainImages2;
  vector<string> trainImagesNames2;
  vector<vector<float> > trainTransforms2;
  vector<vector<KeyPoint> > trainKeypoints2;
  vector<Mat> trainDescriptors2;
  //main_readDescriptorDatabase( pcFileWithImages, trainImages2, trainImagesNames2, trainTransforms2, trainKeypoints2,trainDescriptors2 );

  return 0;
}

//
// addToOutput()
//
// Create linear coefficients files, to be solved via SVD, etc.
//
// Outputs are vecX and vecY
//
int
  addToOutput(
  const vector<DMatch> &matches,

  vector<float>  &queryImageTransform,
  vector<KeyPoint> &queryKeypoints,

  vector< vector<float> > &trainImagesTransforms,
  vector<vector<KeyPoint> > &trainKeypoints,

  vector< vector<float> > &vecX,
  vector< vector<float> > &vecY
  )
{
  float pfMatQuery[16];
  float pfMatQueryInv[16];

  float pfMatResult[16];

  float pfMatTrain[16];
  float pfMatTrainInv[16];

  cv::Mat matQuery( 4, 4, CV_32FC1, pfMatQuery );
  cv::Mat matQueryInv( 4, 4, CV_32FC1, pfMatQueryInv );

  cv::Mat matTrain( 4, 4, CV_32FC1, pfMatTrain );
  cv::Mat matTrainInv( 4, 4, CV_32FC1, pfMatTrainInv );

  memset( pfMatQuery, 0, sizeof(pfMatQuery) );	pfMatQuery[15] = 1;
  memset( pfMatTrain, 0, sizeof(pfMatTrain) );	pfMatTrain[15] = 1;

  // Copy and invert query image transform 
  memcpy( pfMatQuery, (&(queryImageTransform[0])), sizeof(float)*12 );
  cv::invert( matQuery, matQueryInv );

  //cvMatMul( (CvArr*)(&(CvMat)matQuery), (CvArr*)(&(CvMat)matQueryInv), (CvArr*)(&(CvMat)matResult) );

  vector< float > vec_x;
  vec_x.resize(9);
  vector< float > vec_y;
  vec_y.resize(1);

  for( int i = 0; i < matches.size(); i++ )
  {
    int iIdx = matches[i].trainIdx;
    int iImg = matches[i].imgIdx;
    if( iImg >= 0 && iIdx >= 0 )
    {
      KeyPoint kp1 = queryKeypoints[matches[i].imgIdx];
      KeyPoint kp2 = trainKeypoints[iImg][iIdx];

      memcpy( pfMatTrain, (&(trainImagesTransforms[iImg][0])), sizeof(float)*12 );
      cv::invert( matTrain, matTrainInv );

      for( int row = 0; row < 3; row++ )
      {
        vec_x[0] = pfMatQueryInv[4*row+0]*kp1.pt.x - pfMatTrainInv[4*row+0]*kp2.pt.x;
        vec_x[1] = pfMatQueryInv[4*row+0]*kp1.pt.y - pfMatTrainInv[4*row+0]*kp2.pt.y;

        vec_x[2] = pfMatQueryInv[4*row+1]*kp1.pt.x - pfMatTrainInv[4*row+1]*kp2.pt.x;
        vec_x[3] = pfMatQueryInv[4*row+1]*kp1.pt.y - pfMatTrainInv[4*row+1]*kp2.pt.y;

        vec_x[4] = pfMatQueryInv[4*row+2]*kp1.pt.x - pfMatTrainInv[4*row+2]*kp2.pt.x;
        vec_x[5] = pfMatQueryInv[4*row+2]*kp1.pt.y - pfMatTrainInv[4*row+2]*kp2.pt.y;

        vec_x[6] = pfMatQueryInv[4*row+0] - pfMatTrainInv[4*row+0];
        vec_x[7] = pfMatQueryInv[4*row+1] - pfMatTrainInv[4*row+1];
        vec_x[8] = pfMatQueryInv[4*row+2] - pfMatTrainInv[4*row+2];

        vec_y[0] = pfMatTrainInv[4*row+3] - pfMatQueryInv[4*row+3];

        vecX.push_back( vec_x );
        vecY.push_back( vec_y );
      }

        // Save
    }
  }

  //// Save resulting transform
  //outfileTransform = fopen( "match_transform.txt", "a+" );

  //fprintf( outfileTransform, "A:\t" );
  //for( int k = 0; k < queryImagesTransform[i].size(); k++ )
  //{
  //	fprintf( outfileTransform, "%f\t", queryImagesTransform[i][k] );
  //}
  //fprintf( outfileTransform, "B:\t" );
  //for( int k = 0; k < trainImagesTransform[iMaxIndex].size(); k++ )
  //{
  //	fprintf( outfileTransform, "%f\t", trainImagesTransform[iMaxIndex][k] );
  //}
  //fprintf( outfileTransform, "\n" );
  //fclose( outfileTransform );

  return 0;
}

//
//
//
// addToOutputNonLinear()
//
// Output matrices and corresponding points to non-linear solver.
//
//
int
  addToOutputNonLinear(
  const vector<DMatch> &matches,

  vector<float>  &queryImageTransform,
  vector<KeyPoint> &queryKeypoints,

  vector< vector<float> > &trainImagesTransforms,
  vector<vector<KeyPoint> > &trainKeypoints,

  vector< vector<float> > &vecX,
  vector< vector<float> > &vecY
  )
{
  float pfMatQuery[16];
  float pfMatQueryInv[16];

  float pfMatVec[4];
  float pfMatResult[16];

  float pfMatTrain[16];
  float pfMatTrainInv[16];

  cv::Mat matQuery( 4, 4, CV_32FC1, pfMatQuery );
  cv::Mat matTrain( 4, 4, CV_32FC1, pfMatTrain );

  cv::Mat matVec( 4, 1, CV_32FC1, pfMatVec );
  cv::Mat matRes1( 4, 1, CV_32FC1, pfMatResult+0 );
  cv::Mat matRes2( 4, 1, CV_32FC1, pfMatResult+4 );

  memset( pfMatQuery, 0, sizeof(pfMatQuery) );	pfMatQuery[15] = 1;
  memset( pfMatTrain, 0, sizeof(pfMatTrain) );	pfMatTrain[15] = 1;

  // Copy and invert query image transform 
  memcpy( pfMatQuery, (&(queryImageTransform[0])), sizeof(float)*12 );


  vector< float > vecXInner;// Store first 3 rows of matrix (12), then input point (3)
  vector< float > vecYInner;
  vecXInner.resize( 15 );
  vecYInner.resize( 15 );

  for( int i = 0; i < matches.size(); i++ )
  {
    int iIdxQuery = matches[i].queryIdx;
    int iIdxTrain = matches[i].trainIdx;
    int iImg = matches[i].imgIdx;
    if( iImg >= 0 && iIdxTrain >= 0 )
    {
      KeyPoint kp1 = queryKeypoints[iIdxQuery];
      KeyPoint kp2 = trainKeypoints[iImg][iIdxTrain];

      memcpy( pfMatTrain, (&(trainImagesTransforms[iImg][0])), sizeof(float)*12 );

      for( int row = 0; row < 3; row++ )
      {
        for( int col = 0; col < 4; col++ )
        {
          vecXInner[4*row+col] = pfMatQuery[4*row+col];
          vecYInner[4*row+col] = pfMatTrain[4*row+col];
        }
      }
      vecXInner[4*3+0] = kp1.pt.x;
      vecXInner[4*3+1] = kp1.pt.y;
      vecXInner[4*3+2] = 0;

      vecYInner[4*3+0] = kp2.pt.x;
      vecYInner[4*3+1] = kp2.pt.y;
      vecYInner[4*3+2] = 0;

      vecX.push_back( vecXInner );
      vecY.push_back( vecYInner );

      // For MR test data

      pfMatVec[0] = kp1.pt.x;
      pfMatVec[1] = kp1.pt.y;
      pfMatVec[2] = 0;
      pfMatVec[3] = 1;
      CvMat cvMatQuery = (CvMat)matQuery;
      CvMat cvMatVec = (CvMat)matVec;
      CvMat cvMatRes1 = (CvMat)matRes1;
      cvMatMul( (CvArr*)(&cvMatQuery), (CvArr*)(&cvMatVec), (CvArr*)(&cvMatRes1) );

      pfMatVec[0] = kp2.pt.x;
      pfMatVec[1] = kp2.pt.y;
      pfMatVec[2] = 0;
      pfMatVec[3] = 1;
      CvMat cvMatTrain = (CvMat)matTrain;
      cvMatVec = (CvMat)matVec;
      CvMat cvMatRes2 = (CvMat)matRes2;
      cvMatMul( (CvArr*)(&cvMatTrain), (CvArr*)(&cvMatVec), (CvArr*)(&cvMatRes2) );

    }
  }

  //// Save resulting transform
  //outfileTransform = fopen( "match_transform.txt", "a+" );

  //fprintf( outfileTransform, "A:\t" );
  //for( int k = 0; k < queryImagesTransform[i].size(); k++ )
  //{
  //	fprintf( outfileTransform, "%f\t", queryImagesTransform[i][k] );
  //}
  //fprintf( outfileTransform, "B:\t" );
  //for( int k = 0; k < trainImagesTransform[iMaxIndex].size(); k++ )
  //{
  //	fprintf( outfileTransform, "%f\t", trainImagesTransform[iMaxIndex][k] );
  //}
  //fprintf( outfileTransform, "\n" );
  //fclose( outfileTransform );

  return 0;
}


int
  test_output(

  const vector<DMatch> &matches,

  vector<float>  &queryImageTransform,
  vector<KeyPoint> &queryKeypoints,

  vector< vector<float> > &trainImagesTransforms,
  vector<vector<KeyPoint> > &trainKeypoints
  )
{
  // This matrix obtained via SVD
  float pfBMatrix[16] = //{-0.000383,-0.005676,0,0.000770,-0.003143,0.000443,0,0.004358,-9.016907,13.886870,0,-46.222257, 0, 0, 0, 1};
  {-0.000383, -0.005676, 0, -9.016907,
    0.000770, -0.003143, 0, 13.886870,
    0.000443, 0.004358, 0, 	-46.222257,
    0, 0, 0, 1 };


  float pfMatQuery[16];
  float pfMatQueryInv[16];

  float pfMatResult[16];

  float pfMatTrain[16];
  float pfMatTrainInv[16];

  cv::Mat matBMatrix( 4, 4, CV_32FC1, pfBMatrix );

  cv::Mat matQuery( 4, 4, CV_32FC1, pfMatQuery );
  cv::Mat matQueryInv( 4, 4, CV_32FC1, pfMatQueryInv );

  cv::Mat matTrain( 4, 4, CV_32FC1, pfMatTrain );
  cv::Mat matTrainInv( 4, 4, CV_32FC1, pfMatTrainInv );

  memset( pfMatQuery, 0, sizeof(pfMatQuery) );	pfMatQuery[15] = 1;
  memset( pfMatTrain, 0, sizeof(pfMatTrain) );	pfMatTrain[15] = 1;

  // Copy and multiply query image transform 
  memcpy( pfMatQuery, (&(queryImageTransform[0])), sizeof(float)*12 );
  CvMat cvMatQuery = (CvMat)matQuery;
  CvMat cvMatBMatrix = (CvMat)matBMatrix;
  CvMat cvMatQueryInv = (CvMat)matQueryInv;
  cvMatMul( (CvArr*)(&cvMatQuery), (CvArr*)(&cvMatBMatrix), (CvArr*)(&cvMatQueryInv) );

  float pfMatQueryIn[4];
  float pfMatQueryOt[4];
  float pfMatTrainIn[4];
  float pfMatTrainOt[4];
  cv::Mat matQueryIn( 4, 1, CV_32FC1, pfMatQueryIn );
  cv::Mat matQueryOt( 4, 1, CV_32FC1, pfMatQueryOt );
  cv::Mat matTrainIn( 4, 1, CV_32FC1, pfMatTrainIn );
  cv::Mat matTrainOt( 4, 1, CV_32FC1, pfMatTrainOt );

  for( int i = 0; i < matches.size(); i++ )
  {
    int iIdx = matches[i].trainIdx;
    int iImg = matches[i].imgIdx;
    if( iImg >= 0 && iIdx >= 0 )
    {
      KeyPoint kp1 = queryKeypoints[i];
      KeyPoint kp2 = trainKeypoints[iImg][iIdx];

        // Copy and multiply train image transform
      memcpy( pfMatTrain, (&(trainImagesTransforms[iImg][0])), sizeof(float)*12 );
      CvMat cvMatTrain = (CvMat)matTrain;
      cvMatBMatrix = (CvMat)matBMatrix;
      CvMat cvMatTrainInv = (CvMat)matTrainInv;
      cvMatMul( (CvArr*)(&cvMatTrain), (CvArr*)(&cvMatBMatrix), (CvArr*)(&cvMatTrainInv) );

      pfMatQueryIn[0] = kp1.pt.x;
      pfMatQueryIn[1] = kp1.pt.y;
      pfMatQueryIn[2] = 0;
      pfMatQueryIn[3] = 1;

      cvMatQueryInv = (CvMat)matQueryInv;
      CvMat cvMatQueryIn = (CvMat)matQueryIn;
      CvMat cvMatQueryOt = (CvMat)matQueryOt;
      cvMatMul( (CvArr*)(&cvMatQueryInv), (CvArr*)(&cvMatQueryIn), (CvArr*)(&cvMatQueryOt) );

      pfMatTrainIn[0] = kp2.pt.x;
      pfMatTrainIn[1] = kp2.pt.y;
      pfMatTrainIn[2] = 0;
      pfMatTrainIn[3] = 1;

      cvMatTrainInv = (CvMat)matTrainInv;
      CvMat cvMatTrainIn = (CvMat)matTrainIn;
      CvMat cvMatTrainOt = (CvMat)matTrainOt;
      cvMatMul( (CvArr*)(&cvMatTrainInv), (CvArr*)(&cvMatTrainIn), (CvArr*)(&cvMatTrainOt) );

    }
  }

  return 1;
}

int
  writeArray(
  char *pcFileNameBase,
  vector< vector< float > > &vvData,
  char *pcFormat
  )
{
  FILE *infile;
  char pcFileName[300];
  int iReturn;

  // Open header
  sprintf( pcFileName, "%s.txt", pcFileNameBase );
  infile = fopen( pcFileName, "wt" );
  if( !infile )
  {
    return -1;
  }
  int iRows = vvData.size();
  int iCols = vvData[0].size();
  iReturn = fprintf( infile, "cols:\t%d\n", vvData[0].size() );
  iReturn = fprintf( infile, "rows:\t%d\n", vvData.size() );
  iReturn = fprintf( infile, "format:\t%s\n", pcFormat );
  fclose( infile );

  // Open data
  sprintf( pcFileName, "%s.bin", pcFileNameBase );
  infile = fopen( pcFileName, "wb" );
  if( !infile )
  {
    return -1;
  }

  if( pcFormat[0] == 'i' )
  {
    for( int i = 0; i < iRows; i++ )
    {
      for( int j = 0; j < iCols; j++ )
      {
        int iData = vvData[i][j];
        fwrite( &iData, sizeof(int), 1, infile );
      }
    }
  }
  else if( pcFormat[0] == 'f' )
  {
    for( int i = 0; i < iRows; i++ )
    {
      for( int j = 0; j < iCols; j++ )
      {
        double dData = vvData[i][j];
        fwrite( &dData, sizeof(double), 1, infile );
      }
    }
  }
  fclose( infile );

  return 0;
}



int
  main_runMatching(
  char *pcFileQueryImages,
  char *pcFileBogusImages,
  char *pcDirSaveResult,
  char **pcFilesTrainingImages,
  int iTrainingImageCount
  )
{
  string matcherType = defaultMatcherType;
  string fileWithQueryImages = defaultQueryImageName;
  string fileWithTrainImages = defaultFileWithTrainImages;
  string fileWithBogusImages = "";
  string dirToSaveResImages = defaultDirToSaveResImages;

  //matcherType = argv[3];
  fileWithQueryImages = pcFileQueryImages;
  fileWithBogusImages = pcFileBogusImages;
  dirToSaveResImages = pcDirSaveResult;
  fileWithTrainImages = pcFilesTrainingImages[0];

  Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( matcherType );
  Ptr<DescriptorMatcher> descriptorMatcherBogus = DescriptorMatcher::create( matcherType );

 //   Mat maskImage = imread( fileMaskImage, CV_LOAD_IMAGE_GRAYSCALE);
  //if( maskImage.empty() )
  //{
  //	cout << "Mask image can not be read:" << fileMaskImage << endl << ">" << endl;
 //       return false;
  //}

  Mat queryImage;
  vector<Mat> queryImages;
  vector<string> queryImagesNames;
  vector< vector<float> > queryImagesTransform;
  vector<vector<KeyPoint> > queryKeypoints;
  vector<Mat> queryDescriptors;
  _readDescriptorDatabase( &fileWithQueryImages[0], queryImages, queryImagesNames,
    queryImagesTransform, queryKeypoints, queryDescriptors );

  vector<Mat> trainImages;
  vector<string> trainImagesNames;
  vector< vector<float> > trainImagesTransform;
  vector<vector<KeyPoint> > trainKeypoints;
  vector<Mat> trainDescriptors;
  _readDescriptorDatabase( &fileWithTrainImages[0], trainImages, trainImagesNames,
    trainImagesTransform, trainKeypoints, trainDescriptors );
  descriptorMatcher->add( trainDescriptors );


  vector<Mat> bogusImages;
  vector<string> bogusImagesNames;
  vector< vector<float> > bogusImagesTransform;
  vector<vector<KeyPoint> > bogusKeypoints;
  vector<Mat> bogusDescriptors;
  _readDescriptorDatabase( &fileWithBogusImages[0], bogusImages, bogusImagesNames,
    bogusImagesTransform, bogusKeypoints, bogusDescriptors );
  descriptorMatcherBogus->add( bogusDescriptors );

#ifdef WIN32
  _timeb t1, t2;
  _ftime( &t1 );
#endif

#ifdef WIN32
  _ftime( &t2 );
  int t_diff = (int) (1000.0 * (t2.time - t1.time) + (t2.millitm - t1.millitm));   
  printf( "Training time: %d\n", t_diff );
#endif
  vector< vector<float> > vecOutputX;
  vector< vector<float> > vecOutputY;

  // Open & flush output
  string resultName1 = dirToSaveResImages + "\\matches1.txt";
  string resultName2 = dirToSaveResImages + "\\matches2.txt";
  FILE *outfile1;
  FILE *outfile2;
  outfile1 = fopen( resultName1.c_str(), "wt" );
  outfile2 = fopen( resultName2.c_str(), "wt" );
  if( !outfile1 )
    return -1;
  if( !outfile2 )
    return -1;
  fclose( outfile1 );
  fclose( outfile2 );

  // Read each image, match to training
  //vector<KeyPoint> queryKeypoints;
  //Mat queryDescriptors;
  vector<DMatch> matches;
  vector<vector<DMatch> > mmatches;
  vector<vector<DMatch> > mmatchesBogus;
  vector<int> vecImgMatches;
  vecImgMatches.resize( trainImages.size() );
  FILE *outfileTransform = fopen( "match_transform.txt", "wt" );


  for( int i = 0; i < queryImages.size(); i++ )
  {
    for( int j = 0; j < vecImgMatches.size(); j++ )
      vecImgMatches[j] = 0;

    //queryKeypoints.clear();
    mmatches.clear();
    mmatchesBogus.clear();

#ifdef WIN32
    _ftime( &t1 );
#endif
    vector<KeyPoint> &queryKeypointsImg = queryKeypoints[i];
    Mat &queryDescriptorsImg = queryDescriptors[i];

    if( queryKeypointsImg.size() == 0 )
    {
      continue;
    }

    //featureDetector->detect( queryImages[i], queryKeypoints, maskImage );
    //descriptorExtractor->compute( queryImages[i], queryKeypoints, queryDescriptors );
    matchDescriptorsKNN( queryDescriptorsImg, mmatches, descriptorMatcher, 3 );
    matchDescriptorsKNN( queryDescriptorsImg, mmatchesBogus, descriptorMatcherBogus, 1 );

    // Remove matches to self, same image
    for( int j = 0; j < mmatches.size(); j++ )
    {
      for( int k = 1; k < mmatches[j].size(); k++ )
      //for( int k = 1; k < mmatches[k].size(); k++ )
      {
        if( mmatches[j][0].imgIdx == i )
        {
          if(  mmatches[j][k].imgIdx != i )
          {
            mmatches[j][0] =  mmatches[j][k];
            break;
          }
        }
      }
      // Disable for now
      if( 0 && mmatches[j][0].imgIdx == i )
      {
        mmatches[j][0].queryIdx = -1;
        mmatches[j][0].trainIdx = -1;
        mmatches[j][0].imgIdx = -1;
      }
    }

    vector< DMatch > vmMatches;
    //float fRatioThreshold = 0.90;
    float fRatioThreshold = 0.95;
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
        vmMatches.push_back( mmatches[j][0] );
      }
    }

#ifdef WIN32
    _ftime( &t2 );
    int t_diff = (int) (1000.0 * (t2.time - t1.time) + (t2.millitm - t1.millitm));   
    printf( "Testing Time: %d\n", t_diff );
#endif
    // Do hough transform to prune
    //int iMatchCount = houghTransform( queryKeypointsImg, trainKeypoints, vmMatches, 615, 400 );
    //int iMatchCount = houghTransform( queryKeypointsImg, trainKeypoints, vmMatches, 385, 153 ); // AMIGO December 2012 - MICCAI 2013 results
    int iMatchCount = houghTransform( queryKeypointsImg, trainKeypoints, vmMatches, 315, 300 ); // MNI
    //int iMatchCount = houghTransform( queryKeypointsImg, trainKeypoints, vmMatches, 410, 300 ); // AMIGO January 2013

    // Tally votes, find frame with the most matches
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
    // Convolution with [1 1 1] ==> Smoothing
    for( int j = 0; j < vecImgMatches.size(); j++ )
    {
      int iCount = vecImgMatches[j];
      if( iCount > iMaxCount )
      {
        iMaxCount = iCount;
        iMaxIndex = j;
      }
      vecImgMatchesSmooth[j] = vecImgMatches[j];
      if( j > 0 ) vecImgMatchesSmooth[j] += vecImgMatches[j-1];
      if( j < vecImgMatches.size()-1 ) vecImgMatchesSmooth[j] += vecImgMatches[j+1];
    }

    for( int j = 0; j < vecImgMatchesSmooth.size(); j++ )
    {
      vecImgMatches[j] = 0; // This vector will now be used as flag
    }
    // Flag all images and immediate neighborhood when count is above 2
    for( int j = 0; j < vecImgMatchesSmooth.size(); j++ )
    {
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
    // This loop reorders matches according to train image index and discards unflagged ones.
    for( int j = 0; j < vecImgMatchesSmooth.size(); j++ )
    {
      // if not flagged don't add the match
      if( vecImgMatches[j] > 0 ) // if flagged
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

    if( vmMatchesSmooth.size() < 2 )
    {
      continue;
    }

    saveResultImages( queryImages[i], queryKeypointsImg, trainImages, trainKeypoints, vmMatchesSmooth, trainImagesNames, dirToSaveResImages, i );

    // Open & append result
    outfile1 = fopen( resultName1.c_str(), "a+" );
    outfile2 = fopen( resultName2.c_str(), "a+" );
    if( !outfile1 || !outfile2 )
      return -1;
    saveResultData(
      outfile1, outfile2,
      queryImagesNames, queryKeypoints,
      trainImagesNames, trainKeypoints,
      vmMatchesSmooth, i );
    fclose( outfile1 );
    fclose( outfile2 );

    // Save resulting transform
    outfileTransform = fopen( "match_transform.txt", "a+" );

    fprintf( outfileTransform, "A:\t" );
    for( int k = 0; k < queryImagesTransform[i].size(); k++ )
    {
      fprintf( outfileTransform, "%f\t", queryImagesTransform[i][k] );
    }
    fprintf( outfileTransform, "B:\t" );
    for( int k = 0; k < trainImagesTransform[iMaxIndex].size(); k++ )
    {
      fprintf( outfileTransform, "%f\t", trainImagesTransform[iMaxIndex][k] );
    }
    fprintf( outfileTransform, "\n" );
    fclose( outfileTransform );

    //
    // Add to linear solver & test
    //addToOutput(
    //	vmMatchesSmooth,
    //	queryImagesTransform[i], queryKeypoints[i],
    //	trainImagesTransform, trainKeypoints,
    //	 vecOutputX, vecOutputY );
    // For testing result of linear solver (doesn't work well)
    //test_output( vmMatchesSmooth, queryImagesTransform[i], queryKeypoints[i], trainImagesTransform, trainKeypoints );

    addToOutputNonLinear(
      vmMatchesSmooth,
      queryImagesTransform[i], queryKeypoints[i],
      trainImagesTransform, trainKeypoints,
      vecOutputX, vecOutputY );

  }

  writeArray( "non-linear.matches.X", vecOutputX, "f" );
  writeArray( "non-linear.matches.Y", vecOutputY, "f" );

    //matchDescriptors( queryDescriptors, trainDescriptors, matches, descriptorMatcher );

    //descriptorMatcher->match( queryDescriptors, matches );
    //CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );

  //matchDescriptorsKNN( queryDescriptors, trainDescriptors, mmatches, descriptorMatcher, 3 );

    //saveResultImages( queryImage, queryKeypoints, trainImages, trainKeypoints,
      //                matches, trainImagesNames, dirToSaveResImages );
  return 0;
}

//
// SURF SURF FlannBased  C:\downloads\data\isaiah\amigo\anon\anon\20111207\images_query.txt C:\downloads\data\isaiah\amigo\anon\anon\20111207\out.626.txt C:\downloads\data\face\feret\out C:\downloads\data\isaiah\amigo\anon\anon\20111207\out\626-reference\mask.pgm
// SURF SURF FlannBased  C:\downloads\data\isaiah\amigo\anon\anon\20111207\images_query.txt C:\downloads\data\isaiah\amigo\anon\anon\20111207\out.626.txt C:\downloads\data\face\feret\out C:\downloads\data\isaiah\amigo\anon\anon\20111207\out\626-reference\mask.pgm
// SURF SURF FlannBased  C:\downloads\data\isaiah\amigo\anon\anon\20111207\images_query.txt C:\downloads\data\isaiah\amigo\anon\anon\20111207\images_train.txt C:\downloads\data\isaiah\amigo\anon\anon\20111207\matches C:\downloads\data\isaiah\amigo\anon\anon\20111207\out\626-reference\mask.pgm
// SURF SURF FlannBased  C:\downloads\data\isaiah\amigo\anon\anon\20111207\images_query_527.txt C:\downloads\data\isaiah\amigo\anon\anon\20111207\images_train_511.txt C:\downloads\data\isaiah\amigo\anon\anon\20111207\matches C:\downloads\data\isaiah\amigo\anon\anon\20111207\out\626-reference\mask.pgm
// SURF SURF FlannBased  C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\post-ressection.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\pre-ressection.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\matches C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\mask.pgm C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\bogus.txt
//
// SURF SURF FlannBased 
// C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\post-ressection.txt
// C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\pre-ressection.txt
// C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\matches
// C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\mask.pgm
// C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\bogus.txt
//
//
// To prepare data:
// <keypoint> <descriptor> <listing file> <mask>
// SURF SURF C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114625_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\mask.pgm
// SURF SURF C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_121528_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\mask.pgm
// SURF SURF C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_122031_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\mask.pgm
//
// SURF SURF C:\downloads\data\MNI\bite-us\group1\05\pre\sweep_5a\2d\sweep_5a.coords.txt C:\downloads\data\MNI\bite-us\mask.png
//
// SURF SURF C:\downloads\data\ultrasound\sweep-angular-X.coords.txt C:\downloads\data\ultrasound\mask.pgm
// SURF SURF C:\downloads\data\ultrasound\sweep-linear-Z.coords.txt C:\downloads\data\ultrasound\mask.pgm
//
// To match data:
// <new> <bogus> <outdir> <pre-op1> <pre-op2> ...
//
// C:\downloads\data\ultrasound\sweep-linear-Z.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114835_0.img.coords.txt C:\downloads\data\ultrasound\match C:\downloads\data\ultrasound\sweep-angular-X.coords.txt
//
// No ressection, rotation vs linear sweep:
//  C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114835_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_122031_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\matches C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114625_0.img.coords.txt
// No ressection, angular sweep vs linear sweep (used in training):
//  C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114113_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_122031_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\matches C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114625_0.img.coords.txt
// No ressection, linear sweep vs linear sweep:
//  C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114744_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_122031_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\matches C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114625_0.img.coords.txt
// Ressection, linear sweep vs. linear sweep.
//  C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_121528_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_122031_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\matches C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114625_0.img.coords.txt
//
// No ressection, rotation vs self, testing reconstruction:
//  C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114835_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_122031_0.img.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\test\matches C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114835_0.img.coords.txt 
//
// MRI test data
// C:\downloads\data\ultrasound\sweep-angular-X.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114835_0.img.coords.txt C:\downloads\data\ultrasound\match C:\downloads\data\ultrasound\sweep-linear-Z.coords.txt
// C:\downloads\data\ultrasound\sweep-angular-X.coords.txt C:\downloads\data\jay-ultrasoundd\Meat_Experiment2\video\frames\20120426_114835_0.img.coords.txt C:\downloads\data\ultrasound\match C:\downloads\data\ultrasound\sweep-linear-Z.coords.txt
//
// Tamas Data
//
// SURF SURF C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyPhantom_LAxis_Sliding.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\mask.png
// C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_SAxis_Turning.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\FemaleKidney C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_SAxis_Sliding.mha
//
// SURF SURF C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Sliding.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\MaskFemale.png
// SURF SURF C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\MaskFemale.png
// SURF SURF C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_SAxis_Sliding.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\MaskFemale.png
// SURF SURF C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_SAxis_Turning.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\MaskFemale.png
//
// SURF SURF C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\TrackedImageSequence_20121211_095535.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\MaskFemale.png
// SURF SURF C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\TrackedImageSequence_20121211_095535.mha1.mha C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\Mask.png
// SURF SURF C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\TrackedImageSequence_20121211_095535.mha2.mha C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\Mask.png
//
// ** MICCAI 13 results
//	C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\TrackedImageSequence_20121211_095535.mha2.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\matching C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\TrackedImageSequence_20121211_095535.mha1.mha
//
// MNI reconstruction/ground truth evaluation
// C:\downloads\data\MNI\bite-us\group1\02\pre\sweep_2a\2d\2a.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches C:\downloads\data\MNI\bite-us\group1\02\pre\sweep_2c\2d\2c.txt
// C:\downloads\data\MNI\bite-us\group1\02\post\sweep_2u\2d\2u.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches C:\downloads\data\MNI\bite-us\group1\02\pre\sweep_2a\2d\2a.txt
// C:\downloads\data\MNI\bite-us\group1\12\post\sweep_12w\2d\12w.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\12b-12w C:\downloads\data\MNI\bite-us\group1\12\pre\sweep_12b\2d\12b.txt
// C:\downloads\data\MNI\bite-us\group1\05\post\sweep_5v\2d\5v.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\05a-05v C:\downloads\data\MNI\bite-us\group1\05\pre\sweep_5a\2d\5a.txt
// C:\downloads\data\MNI\bite-us\group1\05\post\sweep_5u\2d\5u.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\05b-05u C:\downloads\data\MNI\bite-us\group1\05\pre\sweep_5b\2d\5b.txt
// C:\downloads\data\MNI\bite-us\group1\04\post\sweep_4u\2d\4u.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\04a-04u C:\downloads\data\MNI\bite-us\group1\04\pre\sweep_4a\2d\4a.txt
// C:\downloads\data\MNI\bite-us\group1\07\post\sweep_7x\2d\7x.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\7b-7x C:\downloads\data\MNI\bite-us\group1\07\pre\sweep_7b\2d\7b.txt
// C:\downloads\data\MNI\bite-us\group1\07\post\sweep_7v\2d\7v.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\7b-7v C:\downloads\data\MNI\bite-us\group1\07\pre\sweep_7b\2d\7b.txt
// C:\downloads\data\MNI\bite-us\group1\07\post\sweep_7u\2d\7u.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\7b-7u C:\downloads\data\MNI\bite-us\group1\07\pre\sweep_7b\2d\7b.txt
// C:\downloads\data\MNI\bite-us\group1\07\post\sweep_7w\2d\7w.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\7b-7w C:\downloads\data\MNI\bite-us\group1\07\pre\sweep_7b\2d\7b.txt
// C:\downloads\data\MNI\bite-us\group1\07\post\sweep_7z\2d\7z.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\7b-7z C:\downloads\data\MNI\bite-us\group1\07\pre\sweep_7b\2d\7b.txt
//
// C:\downloads\data\MNI\bite-us\group1\07\post\sweep_7x\2d\7x.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\7e-7x C:\downloads\data\MNI\bite-us\group1\07\pre\sweep_7e\2d\7e.txt
// C:\downloads\data\MNI\bite-us\group1\07\pre\sweep_7e\2d\7e.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\7x-7e C:\downloads\data\MNI\bite-us\group1\07\post\sweep_7x\2d\7x.txt
//
// C:\downloads\data\MNI\bite-us\group1\08\post\sweep_8v\2d\8v.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\8b-8v C:\downloads\data\MNI\bite-us\group1\08\pre\sweep_8b\2d\8b.txt
// C:\downloads\data\MNI\bite-us\group1\08\post\sweep_11v\2d\8v.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\8b-8v C:\downloads\data\MNI\bite-us\group1\08\pre\sweep_8b\2d\8b.txt
// C:\downloads\data\MNI\bite-us\group1\08\post\sweep_13v\2d\8v.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\8b-8v C:\downloads\data\MNI\bite-us\group1\08\pre\sweep_8b\2d\8b.txt
// C:\downloads\data\MNI\bite-us\group1\08\post\sweep_14v\2d\8v.txt  C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha  C:\downloads\data\MNI\bite-us\matches\8b-8v C:\downloads\data\MNI\bite-us\group1\08\pre\sweep_8b\2d\8b.txt
// 
//
// MNI Data
// SURF SURF C:\downloads\data\MNI\bite-us\group1\02\pre\sweep_2a\2d\2a.txt C:\downloads\data\MNI\bite-us\mask.png
// SURF SURF C:\downloads\data\MNI\bite-us\group1\02\pre\sweep_2c\2d\2c.txt C:\downloads\data\MNI\bite-us\mask.png
// SURF SURF C:\downloads\data\MNI\bite-us\group1\02\post\sweep_2u\2d\2u.txt C:\downloads\data\MNI\bite-us\mask.png
//
// AMIGO January 2013: pre/post ressection
// C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_121528.mha
// C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_123142.mha
// C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_132324.mha
// SURF SURF C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_121528.mha1.mha C:\downloads\data\tamas\20130108-us-anon\Mask.png
// SURF SURF C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_123142.mha1.mha C:\downloads\data\tamas\20130108-us-anon\Mask.png
// SURF SURF C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_132324.mha1.mha C:\downloads\data\tamas\20130108-us-anon\Mask.png
// C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_132324.mha1.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\KidneyFemale_LAxis_Turning.mha C:\downloads\data\tamas\20130108-us-anon\matching C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_121528.mha1.mha
//
// SURF SURF C:\downloads\data\tamas\20130108-us-anon\TrackedImageSequence_20130108_123142.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\MaskFemale.png
// SURF SURF C:\downloads\data\tamas\neuro-case\anonymized\ultrasound\TrackedImageSequence_20121211_095535.mha C:\downloads\data\tamas\2013-01-03_KidneyPhantomAndHumanUS\MaskFemale.png
// 
// 

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
  int max_iter = 1000;
  int iter = projTrainPoints.size()<max_iter/4 ? projTrainPoints.size()*4 : max_iter;
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
    
  // Compute squared distance plane to plane in roi
  std::vector<vnl_double_3> queryPlanePoints;
  std::vector<vnl_double_3> estimatePlanePoints;
  int step = 4;
  for(int i=0; i<this->croppedMask.rows; i+=step) {
    for(int j=0; j<this->croppedMask.cols; j+=step) {
      if(this->croppedMask.at<unsigned char>(i,j) == 0)
        continue;
        
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
  oss << "The current image index is " << this->currentImgIndex << std::endl;
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
    this->bogusImagesTransform, this->bogusKeypoints, this->bogusDescriptors, this->bogusDescriptorMatcher, this->bogusStartFrame, this->bogusStopFrame, "bogus");
}

void vtkSlicerSurfFeaturesLogic::computeTrain()
{
  this->setCorrespondenceProgress(0);
  this->setTrainProgress(0);
  this->readAndComputeFeaturesOnMhaFile(this->trainFile, this->trainImages, this->trainImagesNames, \
    this->trainImagesTransform, this->trainKeypoints, this->trainDescriptors, this->trainDescriptorMatcher, this->trainStartFrame, this->trainStopFrame, "train");
}

void vtkSlicerSurfFeaturesLogic::computeQuery()
{
  this->setQueryProgress(0);
  this->setCorrespondenceProgress(0);
  this->readAndComputeFeaturesOnMhaFile(this->queryFile, this->queryImages, this->queryImagesNames, \
    this->queryImagesTransform, this->queryKeypoints, this->queryDescriptors, this->queryDescriptorMatcher, this->queryStartFrame, this->queryStopFrame, "query");
}

void vtkSlicerSurfFeaturesLogic::readAndComputeFeaturesOnMhaFile(const std::string& file, std::vector<cv::Mat>& images,\
  std::vector<std::string>& imagesNames,\
  std::vector<std::vector<float> >& imagesTransform,\
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
    if( !readImageList_mha( file, images, imagesNames, imagesTransform, startFrame, stopFrame) )
      return;
  }
  else
    return;


  // Start and stop frames may have been modified
  this->Modified();
  

  // Compute keypoints and descriptors
  keypoints.clear();
  descriptors.clear();
  // Store an original image
  if(images.size() > 0)
    this->firstImage = images[0].clone();

  // Crop the mask
  this->croppedMask = this->mask.clone();
  cropData(this->croppedMask);

  for(int i=0; i<images.size(); i++)
  {
    cropData(images[i]);
    keypoints.push_back(vector<KeyPoint>());
    descriptors.push_back(Mat());
    this->computeKeypointsAndDescriptors(images[i], keypoints[i], descriptors[i]);
    // Update progress
    int progress = ((i+1)*100)/images.size();
    if(who=="bogus")
      this->setBogusProgress(progress);
    else if(who=="query")
      this->setQueryProgress(progress);
    else
      this->setTrainProgress(progress);
  }
  descriptorMatcher->clear();
  descriptorMatcher->add(descriptors);
}


void vtkSlicerSurfFeaturesLogic::computeInterSliceCorrespondence()
{
  if(this->queryImages.empty() || this->bogusImages.empty() || this->trainImages.empty())
    return;

  this->bestMatches.clear();
  this->bestMatchesCount.clear();
  this->matchesWithBestTrainImage.clear();
  this->afterHoughMatches.clear();

  // Compute mask centroid
  int xcentroid, ycentroid;
  this->computeCentroid(this->croppedMask, xcentroid, ycentroid);
  
  // Go through query images
  for(int i=0; i<this->queryImages.size(); i++)
  {
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
    int iMatchCount = houghTransform( this->queryKeypoints[i], this->trainKeypoints, vmMatches, xcentroid, ycentroid );
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
  fn.replace(fn.end()-4,fn.end(),".bmp");
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
  float threshold = 1; // (mm)
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

    // Make sure the vectors are not colinear to avoid numerical problems
    if(abs(dot_product(v1,v2)) > 0.999)
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
  oss << bestError/inliersIdx.size() << " mm error, " << inliersIdx.size() << " inliers."  << std::endl;
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

