// daspr_vision.hpp
// Written by Devon Ash
// Copyright of DASpR Inc
//=======================================================
// include guard

#ifndef __THUNDERBOTS_VISION_H_INCLUDED__
#define __THUNDERBOTS_VISION_H_INCLUDED__

// ----- C++ Libraries -----
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <vector>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <map>

// ----- External Libraries -----
#include <cv_bridge/cv_bridge.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"
#include <opencv2/nonfree/nonfree.hpp>
#include <sensor_msgs/image_encodings.h>
#include "opencv2/core/core.hpp"
#include <sensor_msgs/image_encodings.h>
#include <assert.h>
//#include <omp.h>
#include <opencv2/gpu/gpu.hpp>

#endif
