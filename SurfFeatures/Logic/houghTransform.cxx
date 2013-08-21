#include "houghTransform.h"

#include <math.h>

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
    float &fPollRow,		// Return row value (expected position of train)
    float &fPollCol,		// Return col value, (expected position of train)
    float &fPollOri,		// Return orientation value, (expected position of train)
    float &fPollScl,		// Return scale value, (expected position of train)
    int   bMirror	= 0// Match is a mirror of this feature
    ) const
  {
    if( !bMirror )
    {
      float fOriDiff = fOri - m_Key_fOri; // Diff of ori between query and train keypoint
      float fScaleDiff = fScale / m_Key_fScale; // Diff of scale train/query

      float fDist = m_fDistanceToBooth*fScaleDiff;  // Distance between query and booth normalize by scale diff ==> expected distance of train to booth
      float fAngle = m_fAngleToBooth + fOriDiff;    // Expected orientation diff between train and booth

      fPollRow = (float)(sin( fAngle )*fDist + fRow); // expected y position of train
      fPollCol = (float)(cos( fAngle )*fDist + fCol); // Expected x position of train

      fPollOri = fOri + m_fAngleDiff;  // Expected orientation of train
      fPollScl = m_fScaleDiff*fScale;  // Expected scale of train... normalized by scale of booth?
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




float angle_radians(float fAngle )
{
  return (2.0f*PI*fAngle) / 180.0f;
}



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
  float fTranslationErrorThres,
  float fScaleErrorThres,
  float fOrientationErrorThres)
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

  // Determine the transform for keypoint correspondences
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

    hc.SetHoughCode( (float)refy, (float)refx, 0, 10 );
    hc.FindPollBooth( keyT.pt.y, keyT.pt.x, angle_radians(keyT.angle), keyT.size, kg.m_Key_fRow, kg.m_Key_fCol, kg.m_Key_fOri, kg.m_Key_fScale );
    kg.iIndex0 = i;
    vecKG.push_back( kg );
  }

  // Find the transform that has most votes
  int iMaxIndex;
  int iMaxCount = 0;
  int equals = 0;
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
      equals = 0;
    }
    else if( iCount == iMaxCount)
      equals++;
  }

  // Disable matches that don't agree on the chosen transform
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

  if(equals != 0)
    cout << "Hough Transform found " << equals+1 << " equivalent solutions." << endl;
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
