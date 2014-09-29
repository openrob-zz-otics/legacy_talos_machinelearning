cmake_minimum_required(VERSION 2.8.3)
project(talos_machinelearning)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs cv_bridge message_generation)
find_package(OpenCV)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

#######################################
## Declare ROS messages and services ##
#######################################

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
INCLUDE_DIRS include/${PROJECT_NAME}
#  INCLUDE_DIRS include
#  LIBRARIES daspr_machinelearning
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
CATKIN_DEPENDS sensor_msgs geometry_msgs roscpp nodelet image_transport opencv2 message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(include/${PROJECT_NAME})
## Declare a cpp library
# add_library(daspr_machinelearning
#   src/${PROJECT_NAME}/daspr_machinelearning.cpp
# )
add_library(normalization src/Normalization.cpp)
add_library(dataset src/Dataset.cpp)
add_library(experiment src/Experiment.cpp)
add_library(unsupervisedfeatures src/UnsupervisedFeatures.cpp)
add_library(supervisedfeatures src/SupervisedFeatures.cpp)
add_library(ROC src/ROC.cpp)
add_library(confusionmatrix src/ConfusionMatrix.cpp)
add_library(featureextractor src/FeatureExtractor.cpp)
## Declare a cpp executable
# add_executable(daspr_machinelearning_node src/daspr_machinelearning_node.cpp)
add_executable(general_object_recognition src/GeneralObjectRecognitionModel.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(daspr_machinelearning_node daspr_machinelearning_generate_messages_cpp)

## Specify libraries to link a library or executable target against
 target_link_libraries(general_object_recognition
   ${catkin_LIBRARIES} 
 )

target_link_libraries(dataset ${catkin_LIBRARIES})
target_link_libraries(normalization ${catkin_LIBRARIES})
target_link_libraries(general_object_recognition experiment)
target_link_libraries(general_object_recognition dataset)
target_link_libraries(general_object_recognition normalization)
target_link_libraries(general_object_recognition unsupervisedfeatures)
target_link_libraries(general_object_recognition supervisedfeatures)
target_link_libraries(supervisedfeatures ${OpenCV_LIBRARIES})
target_link_libraries(general_object_recognition ${OpenCV_LIBRARIES})
target_link_libraries(general_object_recognition featureextractor)
target_link_libraries(featureextractor ${OpenCV_LIBRARIES})
target_link_libraries(general_object_recognition confusionmatrix)
#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS daspr_machinelearning daspr_machinelearning_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_daspr_machinelearning.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
