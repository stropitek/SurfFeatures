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


void matchDescriptorsKNN( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, Ptr<DescriptorMatcher>& descriptorMatcher, int k)
{
  // Assumes training descriptors have already been added to descriptorMatcher
  //cout << "< Set train descriptors collection in the matcher and match query descriptors to them..." << endl;
    //descriptorMatcher->add( trainDescriptors );
  descriptorMatcher->knnMatch( queryDescriptors, matches, k );
  CV_Assert( queryDescriptors.rows == (int)matches.size() || matches.empty() );
  //cout << ">" << endl;
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