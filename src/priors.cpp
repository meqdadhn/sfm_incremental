#include "sfm/priors.h"

#include "sfm/geometry.h"
#include "sfm/photogrammetry.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>


namespace sfm
{
namespace
{
bool IsNumber(const std::string &s)
{
  char *end = nullptr;
  std::strtod(s.c_str(), &end);
  return end != s.c_str() && *end == '\0';
}
} // namespace

int LoadTrajectoryPriors(const TrajectoryParams &params, std::vector<Image> *images)
{
  for (Image &image : *images)
    image.has_prior = image.has_prior_rotation = false;
  if (params.source == "none")
    return 0;

  if (params.source == "file")
  {
    std::ifstream in(params.file);
    if (!in)
      throw std::runtime_error("Cannot open trajectory file " + params.file);
    std::map<std::string, ImageId> by_name;
    for (const Image &image : *images)
      by_name[image.name] = image.id;
    std::string line;
    int row = 0;
    while (std::getline(in, line))
    {
      std::istringstream ss(line);
      std::vector<std::string> tok;
      for (std::string t; ss >> t;)
        tok.push_back(t);
      if (tok.empty() || tok[0][0] == '#')
        continue;
      if (IsNumber(tok[0]) && tok.size() >= 6) // original format: omega phi kappa X Y Z, image order
      {
        if (row < static_cast<int>(images->size()))
        {
          Image &image = (*images)[row];
          image.prior_position = Eigen::Vector3d(std::stod(tok[3]), std::stod(tok[4]), std::stod(tok[5]));
          image.has_prior = true;
          image.prior_rotation =
              PoseFromOPK(DegToRad(std::stod(tok[0])), DegToRad(std::stod(tok[1])), DegToRad(std::stod(tok[2])), image.prior_position).R;
          image.has_prior_rotation = true;
        }
        ++row;
      }
      else if (tok.size() >= 4 && by_name.count(tok[0])) // name X Y Z ...
      {
        Image &image = (*images)[by_name[tok[0]]];
        image.prior_position = Eigen::Vector3d(std::stod(tok[1]), std::stod(tok[2]), std::stod(tok[3]));
        image.has_prior = true;
      }
    }
  }
  else
  {
    throw std::invalid_argument("Unknown trajectory source: " + params.source);
  }
  return static_cast<int>(std::count_if(images->begin(), images->end(), [](const Image &im) { return im.has_prior; }));
}

} // namespace sfm
