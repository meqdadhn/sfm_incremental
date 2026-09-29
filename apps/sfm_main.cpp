/*******************************************************************************
* @file    sfm_main.cpp
* @brief   Usage: sfm_main <config.yaml> [-v <verbosity>]
*******************************************************************************/

#include <cstring>
#include <exception>
#include <iostream>

#include <glog/logging.h>

#include "sfm/config.h"
#include "sfm/pipeline.h"

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " <config.yaml> [-v <verbosity>]\n";
    return 1;
  }
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  for (int i = 2; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "-v"))
      FLAGS_v = std::atoi(argv[i + 1]);
  google::InitGoogleLogging(argv[0]);

  try
  {
    sfm::Pipeline pipeline(sfm::LoadConfig(argv[1]));
    return pipeline.Run() ? 0 : 2;
  }
  catch (const std::exception &e)
  {
    LOG(ERROR) << e.what();
    return 1;
  }
}
