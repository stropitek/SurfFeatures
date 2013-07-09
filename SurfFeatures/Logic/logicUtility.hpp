void writeVarForMatlab(std::ofstream& ofs, std::string name, int var)
{
  ofs << name << " %d 1 " << var << std::endl;
}

void writeVarForMatlib(std::ofstream& ofs, std::string name, float var)
{
  ofs << name << " %f 1 " << var << std::endl;
}

void writeVarForMatlib(std::ofstream& ofs, std::string name, double var)
{
  ofs << name << " %f 1 " << var << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::string var)
{
  ofs << name << " %s 1 " << var << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<int> var)
{
  ofs << name << " %d " << var.size();
  for(int i=0; i<var.size(); i++)
  {
    ofs << " " << var[i];
  }
  ofs << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<double> var)
{
  ofs << name << " %f " << var.size();
  for(int i=0; i<var.size(); i++)
  {
    ofs << " " << var[i];
  }
  ofs << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<float> var)
{
  ofs << name << " %f " << var.size();
  for(int i=0; i<var.size(); i++)
  {
    ofs << " " << var[i];
  }
  ofs << std::endl;
}

void writeVarForMatlab(std::ofstream& ofs, std::string name, std::vector<std::string> var)
{
  ofs << name << " %s " << var.size();
  for(int i=0; i<var.size(); i++)
  {
    ofs << " " << var[i];
  }
  ofs << std::endl;
}