#ifndef MATLAB_WRITER_H
#define MATLAB_WRITER_H

#include <fstream>
#include <string>
#include <vector>

#include <cv.h>

using namespace std;
using namespace cv;

void writeVarForMatlab(std::ofstream& ofs, std::string name, int var);
void writeVarForMatlib(std::ofstream& ofs, std::string name, float var);
void writeVarForMatlib(std::ofstream& ofs, std::string name, double var);
void writeVarForMatlab(std::ofstream& ofs, std::string name, std::string var);
void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<int> var);
void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<double> var);
void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<float> var);
void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<std::string> var);
void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<std::vector<DMatch> > var);

#endif