#include "mhaReader.h"

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
  transformsValidity.clear();
  
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

void readImages_mha(const std::string& filename, std::vector<cv::Mat>& images, std::vector<int> frames)
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

   // Read & write images
   // frames should be ordered ascendant
   void *ptr = NULL;
   int lastFrame = 0;
   for( int i = 0; i < frames.size(); i++ )
   {
     int skip = frames[i]-lastFrame;
     lastFrame = frames[i]+1;
     #ifdef WIN32
     _fseeki64(infile, (__int64)iImgRows*(__int64)iImgCols*(__int64)skip, SEEK_CUR);
     #else
     fseek(infile, (long int)iImgRows*(long int)iImgCols*(long int)skip, SEEK_CUR);
     #endif

     Mat mtImgNew = mtImg.clone();
     fread( mtImgNew.data, 1, iImgRows*iImgCols, infile );
     images.push_back( mtImgNew );

   }
   delete [] pucImgData;
   
   fclose( infile );
}


// Will read all the data form firstFrame to lastFrame, excluding all slices with invalid transform
int read_mha(
  const string& filename,
  vector <Mat>& images,
  vector<string>& filenames,
  vector< vector<float> >& transforms,
  vector<bool>& transformsValidity,
  int& firstFrame,
  int& lastFrame,
  std::vector<int>& frames
  )
{
  int iImgCols = -1;
  int iImgRows = -1;
  int iImgCount = -1;
  if(readImageDimensions_mha(filename, iImgCols, iImgRows, iImgCount))
    return 1;

    if(firstFrame < 0)
     firstFrame = 0;
   if(firstFrame >= iImgCount)
     firstFrame = iImgCount-1;
   if(lastFrame < 0 || lastFrame >= iImgCount)
     lastFrame = iImgCount-1;
   if(firstFrame > lastFrame)
     firstFrame = lastFrame;

  readImageTransforms_mha(filename, transforms, transformsValidity, filenames);
  
    
  // Keep the transforms that are in the first-last range
  std::vector<std::vector<float> > allTransforms = transforms;
  std::vector<std::string> allFilenames = filenames;
  std::vector<bool> allValidity = transformsValidity;
  transformsValidity.clear();
  transforms.clear();
  filenames.clear();
  frames.clear();
  
  // Discarding data with invalid transform
  for(int i=0; i<allValidity.size(); i++) {
    if(i<firstFrame || i>lastFrame  || !allValidity[i])
      continue;
    transforms.push_back(allTransforms[i]);
    filenames.push_back(allFilenames[i]);
    frames.push_back(firstFrame+i);
    transformsValidity.push_back(true);
  }

  readImages_mha(filename, images, frames);

  std::cout << "Read images: " << images.size() << std::endl;
  std::cout << "Read transforms: " << transforms.size() << std::endl;
  std::cout << "Read transforms validity: " << transformsValidity.size() << std::endl;
  std::cout << "Read names: " << filenames.size() << std::endl;
  
  return 0;

}

int read_mha(
  const string& filename,
  vector <Mat>& images,
  vector<string>& filenames,
  vector< vector<float> >& transforms,
  vector<bool>& transformsValidity,
  std::vector<int> frames
  )
{
  readImageTransforms_mha(filename, transforms, transformsValidity, filenames);
  
  std::vector<std::vector<float> > allTransforms = transforms;
  std::vector<std::string> allFilenames = filenames;
  std::vector<bool> allValidity = transformsValidity;
  transformsValidity.clear();
  transforms.clear();
  filenames.clear();
  
  // Discarding data with invalid transform
  for(int i=0; i<frames.size(); i++)
  {
    filenames.push_back(allFilenames[frames[i]]);
    transformsValidity.push_back(allValidity[frames[i]]);
    transforms.push_back(allTransforms[frames[i]]);
  }

  readImages_mha(filename, images, frames);

  std::cout << "Read images: " << images.size() << std::endl;
  std::cout << "Read transforms: " << transforms.size() << std::endl;
  std::cout << "Read transforms validity: " << transformsValidity.size() << std::endl;
  std::cout << "Read names: " << filenames.size() << std::endl;
  
  return 0;
}



// ==============================================
// Helpers - conversion functions
// ==============================================
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

std::vector<float> convertVtkMatrix4x4ToMhaTransform(vtkMatrix4x4* matrix)
{
  std::vector<float> result;
  for(int i=0; i<3; i++)
  {
    for(int j=0; j<4; j++)
      result.push_back(matrix->GetElement(i,j));
  }

  return result;
}

void convertVnlMatrixToVtkMatrix4x4(const vnl_matrix<double> vnlMatrix , vtkMatrix4x4* vtkMatrix)
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

void convertMhaTransformToVtkMatrix(const std::vector<float>& vec, vtkMatrix4x4* vtkMatrix)
{
  vtkMatrix->Identity();
  if(vec.size() != 12)
    return;
  for(int i=0; i<3; i++){
    for(int j=0; j<4; j++)
      vtkMatrix->SetElement(i,j,vec[i*4+j]);
  }
}

vnl_matrix<double> convertMhaTransformToVnlMatrix(const vector<float>& vec)
{
  vnl_matrix<double> result;
  if(vec.size() != 12)
    return result;
  result.set_size(4,4);
  result.set_identity();
  for(int i=0; i<3; i++){
    for(int j=0; j<4; j++)
      result(i,j)=vec[i*4+j];
  }
  return result;
}

// ==================================
// OpenCV
// ==================================
void computeKeypointsAndDescriptors(const cv::Mat& data,\
                                    const cv::Mat& mask,\
                                    std::vector<cv::KeyPoint> &keypoints,\
                                    cv::Mat &descriptors,\
                                    int minHessian)
{
  // Get the descriptor of the current image
  cv::SurfFeatureDetector detector(minHessian);

  // Find Keypoints
  detector.detect(data,keypoints,mask);

  // Get keypoints' surf descriptors
  cv::SurfDescriptorExtractor extractor;
  extractor.compute(data, keypoints, descriptors);
}

void matchDescriptors( const Mat& queryDescriptors, vector<DMatch>& matches, Ptr<DescriptorMatcher> descriptorMatcher )
{
	// Assumes training descriptors have already been added to descriptorMatcher
    descriptorMatcher->match( queryDescriptors, matches );
    CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );
}

void matchDescriptorsKNN( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, Ptr<DescriptorMatcher> descriptorMatcher, int k)
{
  // Assumes training descriptors have already been added to descriptorMatcher
  descriptorMatcher->knnMatch( queryDescriptors, matches, k );
  CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );
}

int maskMatchesByMatchPresent( const vector<DMatch>& matches, vector<char>& mask )
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

int maskMatchesByTrainImgIdx( const vector<DMatch>& matches, int trainImgIdx, vector<char>& mask )
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

cv::Mat getResultImage( const Mat& queryImage, const vector<KeyPoint>& queryKeypoints,\
                     const Mat& trainImage, const vector<KeyPoint>& trainKeypoints,\
                     const vector<DMatch>& matches,int iTrainImageIndex)
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
  return drawImg;
}

vector<int> countVotes(const vector<DMatch>& matches, int trainSize)
{
  vector<int> votes(trainSize, 0);
  for(int i=0; i<matches.size(); i++)
  {
    if (matches[i].imgIdx != -1)
      votes[matches[i].imgIdx]++;
  }
  return votes;
}

vector<DMatch> getValidMatches(const vector<vector<DMatch> >& matches)
{
  vector<DMatch> validMatches;
  for(int i=0; i<matches.size(); i++)
  {
    if(matches[i][0].imgIdx >= 0)
      validMatches.push_back(matches[i][0]);
  }
  return validMatches;
}

vector<DMatch> filterValidMatches(const vector<DMatch>& matches)
{
  vector<DMatch> validMatches;
  for(int i=0; i<matches.size(); i++)
  {
    if(matches[i].imgIdx >= 0)
      validMatches.push_back(matches[i]);
  }
  return validMatches;
}

vector<DMatch> filterBogus(const vector<DMatch>& matches, const vector<DMatch>& bmatches, float threshold)
{
  vector<DMatch> filtered;
  // Disable best matches if they also have a good match with bogus data
  for( int j = 0; j < matches.size(); j++ )
  {
    float fRatio = matches[j].distance / bmatches[j].distance;
    if( fRatio < threshold )
    {
      filtered.push_back(matches[j]);
    }
  }
  return filtered;
}


vector<DMatch> filterSmoothAndThreshold(const vector<int>& votes, const vector<DMatch> matches)
{
  vector<DMatch> smoothMatches;
  vector< int > votesSmooth(votes.size(), 0);

  for( int j = 0; j < votes.size(); j++ )
  {
    // Smooth the # of matches curve by convultion with [1 1 1].
    votesSmooth[j] = votes[j];
    if( j > 0 ) votesSmooth[j] += votes[j-1];
    if( j < votes.size()-1 ) votesSmooth[j] += votes[j+1];
  }

  vector<int> flags;
  flags.resize(votes.size());

    // votes is reinitilised. It will now store flags.
  for( int j = 0; j < votesSmooth.size(); j++ )
  {
    flags[j] = 0;
  }
  for( int j = 0; j < votesSmooth.size(); j++ )
  {
      // If training image has more than 2 matches (after smoothing), flag the training image and its immediate neighborhood.
    if( votesSmooth[j] >= 2 )
    {
        // flag neighborhood
      flags[j] = 1;
      if( j > 0 ) flags[j-1]=1;
      if( j < flags.size()-1 ) flags[j+1]=1;
    }
  }

  // Save matches
  smoothMatches.clear();
  for( int j = 0; j < votesSmooth.size(); j++ )
  {
      // Discard unflagged train images
    if( votes[j] > 0 )
    {
      for( int k = 0; k < matches.size(); k++ )
      {
        int iImg =  matches[k].imgIdx;
        if( iImg == j )
        {
          smoothMatches.push_back( matches[k] );
        }
      }
    }
  }

      // This commented part here would be I think a more efficient way for the block just above this
    //for(int k=0; k < matches.size(); k++)
    //{
    //  iImg = matches[k].imgIdx;
    //  if( votes[imgIdx] > 0 )
    //    continue;
    //  vmMatchesSmooth.push_back( matches[k] );
    //}
  return smoothMatches;
}

void computeCentroid(const cv::Mat& mask, int& x, int& y)
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

cv::Mat rotateImage(const Mat& source, double angle)
{
    Point2f src_center(source.cols/2.0F, source.rows/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
    Mat dst;
    warpAffine(source, dst, rot_mat, source.size());
    return dst;
}

// =======================================================
// Helpers - geometry functions
// =======================================================
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

vnl_double_3 projectPoint(vnl_double_3 point, vnl_double_3 normalToPlane, double offset)
{
  normalToPlane.normalize();
  double dist = dot_product(point, normalToPlane) + offset;
  vnl_double_3 vec = dist*normalToPlane;
  return point - vec;
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

double getAngle(const vnl_double_3& u, const vnl_double_3& v)
{
  // Get the angle (caution not signed!) use atan2 for signed angle in 2d plane
  return acos(dot_product(u,v)/(u.two_norm()*v.two_norm()));
}

void getAxisAndRotationAngle(vnl_double_3 v, vnl_double_3 v_new, vnl_double_3& axis, double& angle)
{
  axis = vnl_cross_3d(v, v_new);
  angle = getAngle(v,v_new);
}

vnl_double_3 transformPoint(const vnl_double_3 in, vtkMatrix4x4* transform)
{
  double point[4];
  vnlToArrayDouble(in, point);
  double point_t[4];
  transform->MultiplyPoint(point, point_t);
  return arrayToVnlDouble(point_t);
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

vnl_double_3 computeTranslation(vnl_double_3 u, vnl_double_3 v, vnl_double_3 w, vnl_double_3 trainCentroid, vnl_double_3 queryCentroid)
{
  vnl_double_3 t;
  t[0] = trainCentroid[0] - (u[0]*queryCentroid[0] + v[0]*queryCentroid[1] + w[0]*queryCentroid[2]);
  t[1] = trainCentroid[1] - (u[1]*queryCentroid[0] + v[1]*queryCentroid[1] + w[1]*queryCentroid[2]);
  t[2] = trainCentroid[2] - (u[2]*queryCentroid[0] + v[2]*queryCentroid[1] + w[2]*queryCentroid[2]);
  return t;
}

void setEstimate(vtkMatrix4x4* estimate, const vnl_double_3& u, const vnl_double_3& v, const vnl_double_3& w, const vnl_double_3& t)
{
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
}

// ===========================================
// Helpers - other
// ===========================================
void computeCorners(std::vector<vnl_double_3>& result, vtkMatrix4x4* transform, int maxX, int maxY) {
  vnl_double_3 p1, p2, p3, p4;
  p1[0]=0; p1[1]=0; p1[2]=0;
  p2[0]=0; p2[1]=maxY; p2[2]=0;
  p3[0]=maxX; p3[1]=maxY; p3[2]=0;
  p4[0]=maxX; p4[1]=0; p4[2]=0;

  result.push_back(transformPoint(p1, transform));
  result.push_back(transformPoint(p2, transform));
  result.push_back(transformPoint(p3, transform));
  result.push_back(transformPoint(p4, transform));
}

// =======================================================
// Image manipulation and conversion
// =======================================================
vtkImageData* cropData(vtkImageData* data, float cropRatios[4])
{
  int dims[3];
  data->GetDimensions(dims);
  vtkSmartPointer<vtkExtractVOI> extractVOI = vtkSmartPointer<vtkExtractVOI>::New();
  extractVOI->SetInput(data);
  extractVOI->SetVOI(dims[0]*cropRatios[0],dims[0]*cropRatios[1],dims[1]*cropRatios[2],dims[1]*cropRatios[3], 0, 0);
  extractVOI->SetSampleRate(1,1,1);
  extractVOI->Update();
  return extractVOI->GetOutput();
}

void cropData(cv::Mat& img, float cropRatios[4])
{
  cv::Size size = img.size();
  cv::Rect roi(size.width*cropRatios[0], size.height*cropRatios[2],\
    size.width*(cropRatios[1]-cropRatios[0]), size.height*(cropRatios[3]-cropRatios[2]));
  // Clone the image to physically crop the image (not just jumping on same pointer)
  img = img(roi).clone();
}

cv::Mat convertImage(vtkImageData* image)
{
  int dims[3];
  image->GetDimensions(dims);

  // TODO: Make sure the pointer is unsigned char... or use the proper opencv type...
    cv::Mat ocvImage(dims[1],dims[0],CV_8U ,image->GetScalarPointer());
  // Clone the image so you don't share the pointer with vtkImageData
  return ocvImage.clone();
}


void readTab(const string& filename, vector<vector<string> >& tab)
{
  tab.clear();
  ifstream file(filename.c_str());
  if(!file.is_open())
    return;
  vector<string> lines;
  while(!file.eof())
  {
    string str; getline(file,str);
  }

  for(int i=0; i<lines.size(); i++)
  {
    istringstream iss(lines[i]);
    vector<string> words;
    while(!iss.eof())
    {
      string word;
      iss >> word;
      if(!word.empty())
        words.push_back(word);
    }
    tab.push_back(words);
  }
}

// Normalize dir by adding the appropriate delimiter at the end, if not present
void normalizeDir(std::string& dir)
{
  #ifdef WIN32
  const char dlmtr = '\\';
  #else
  const char dlmtr = '/';
  #endif

  if(!dir.empty()){
    if(*(dir.end()-1) != dlmtr)
      dir = dir + dlmtr;
  }

}

// Get the directory given a filename.
void splitDir(const std::string& filename, std::string& dir, std::string& file, string& ext)
{
  #ifdef WIN32
  const char dlmtr = '\\';
  #else
  const char dlmtr = '/';
  #endif

  size_t pos = filename.rfind(dlmtr);
  dir = pos == string::npos ? "" : filename.substr(0, pos) + dlmtr;
  file = pos == string::npos ? "" : filename.substr(pos+1);

  if(file.empty()){
    ext=""; return;
  }

  pos = file.find_last_of(".");
  ext = pos == string::npos ? "" : file.substr(pos+1);
  file = pos == string::npos ? file : file.substr(0,pos);

}
