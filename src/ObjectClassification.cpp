// Author: Devon Ash
// contact: noobaca2@gmail.com
// Produced for licensing by the Owners of Developers of Automated Space Robotics Inc.
// DASPR Inc. 

#include "ObjectClassification.h"

namespace enc = sensor_msgs::image_encodings;
		

// ATTENTION:
// These values are given during the setup time for competition. 
const std::map<std::string, std::string> ObjectClassification::object_class_locations = ObjectClassification::init_object_class_locations();

ObjectClassification::ObjectClassification()
{
		
}

ObjectClassification::~ObjectClassification()
{

}


void ObjectClassification::save_image(const sensor_msgs::ImageConstPtr& image)
{
	// do stuff
	// Convert the image to a openCV image and save it in the vector.
	if (image)
	{

	} else { std::cout << "Nothing " << std::endl; }
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(image, enc::BGR8);
	}	
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_brdige exception: %s", e.what());
		return;
	}

	this->recent_images.push_back(cv_ptr->image);
	std::cout << "Saved image " << std::endl;
	if (this->recent_images.size() > 10)
	{
		this->recent_images.clear();
	}
}

// Topic: topic to subscribe to for images
ObjectClassification::ObjectClassification(std::string topic)
{
	this->camera_topic = topic;
	// Read the images from the camera topic.
	// /camera/image_raw for integrated cam
	
}

// The getObjectsInScene service callback
// Bruteforce searches for all of the objects in the scene,
// Matches against every set of images in the dataset
bool ObjectClassification::getObjectsInScene(vision::GetObjectsInScene::Request &req, vision::GetObjectsInScene::Response &res)
{
	// Pass it into classification algorithm
	// Create real objects once classification algorithm determines what it is
	
	// Put into vector and return
	std::vector<RealObject> vec;
		
	return true;
}


// The containsObject service callback
bool ObjectClassification::containsObject(vision::Contains::Request &req, vision::Contains::Response &res)
{
	return true;
}
	
// The findobject service callback
bool ObjectClassification::findObject(vision::FindObject::Request &req, vision::FindObject::Response &res)
{
	struct RealObject obj;
	cv::Mat pic;
	obj.picture = pic;

	return true;
}
	

// Description: Matches objects in the real world scene to suggested images.
// Careful as this is an expensive matching. 
//
// Knowledge: We are given what objects we have to find and their locations. 
//
// Requirements:
// A database of all the image descriptors
// The subset of requested objects (so we will already know what to grab)
// 
// Output:
// The classify will match the image given from the current scene to the image in
// the database and then localize it. 
//
bool ObjectClassification::match(vision::Match::Request &req, vision::Match::Response &res)
{
	// For the competition it really depends how much we'll use
	// Instance recognition, and how much we'll use
	// General recognition. 	

	// We're going to be using instance recognition since only the required
	// parts of the competition tasks use instance recognition
	// and other tasks will use general recognition. 

	// Then we can just research matching of keypoints. 

	// Load relevant images
	cv::Mat* scene = convertToCvImage(req.image);
	if (scene->data) { ROS_INFO("VISION: Scene image loaded."); }

	std::vector<std::string> req_objs = req.objects_to_check;

	// Each Object will be assigned to an object class.
	// Each Object class will be asisgned to an object location.
	// Need a mapping from locations to objects and vice versa. We can
	// Statically define them in a pre-competition file. 
	
	// Ok, so match will attempt all of the images. 
	std::map<std::string, cv::Mat>::iterator map_iterator;

	for (map_iterator = (this->dataset.begin()); map_iterator != (this->dataset.end()); map_iterator++)
	{
		// Matching function for scene to the image. 
	}
	// Get the images from the map.
	
	struct RealObject object;
	cv::Mat pics;
	object.picture = pics;

	return true;
}

cv::Mat* ObjectClassification::convertToCvImage(const sensor_msgs::Image& image)
{
	cv_bridge::CvImagePtr cv_ptr;
	cv_ptr = cv_bridge::toCvCopy(image, enc::BGR8);
	
	return &cv_ptr->image;
}

char* getFileType(char* filename)
{
	// Returns the file type of a given filename.
	char* stf = strtok(filename, ".");
	return strtok(NULL, ".");
}

// Gets the images and their names and stores them into a map by object name. 
void ObjectClassification::loadDataset(std::string dataset_directory)
{
	this->dataset_directory = dataset_directory;	
	struct dirent *entry;
	DIR *pDIR;	
	pDIR = opendir(dataset_directory.c_str());

	if (pDIR)
	{
		while ((pDIR != NULL) && (entry = readdir(pDIR)))
		{
			if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
			{
				char* filetype = getFileType(entry->d_name);
				if (strcmp(filetype, "jpg") == 0)
				{
					// Load the file into an cv::Mat
					cv::Mat image = cv::imread(entry->d_name);
					this->dataset.insert(std::pair<std::string, cv::Mat>(entry->d_name, image));
					ROS_INFO("Loaded %s.%s to visual memory", entry->d_name, filetype);
				}
			}


		}		
	}
	 else
	{
		ROS_INFO("Invalid directory: %s", dataset_directory.c_str());
	}
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "object_classification");
	ros::NodeHandle nh("~");

	std::string topic;
	std::string dataset_directory;
	
	nh.getParam("camera_topic", topic);
	nh.getParam("dataset_directory", dataset_directory);

	if (topic.empty())
	{
		ROS_INFO("Please set a topic to subscribe to. [camera_topic:=topic_name]");
		return 0;
	}
	
	if (dataset_directory.empty())
	{
		ROS_INFO("Please set the path for the image matching. [dataset_directory:=dataset_path]");
		return 0;
	}

	ObjectClassification oc(topic);
	oc.loadDataset(dataset_directory);
	ROS_INFO("Loaded dataset: %s", dataset_directory.c_str());

	// Get the node to subscribe to the image topic
	ros::Subscriber subscriber = nh.subscribe(oc.camera_topic, 1, &ObjectClassification::save_image, &oc);
	ROS_INFO("Subscribed to camera_topic");

	// Set up the services for the node. 
	ros::ServiceServer classify_service = nh.advertiseService("classify", &ObjectClassification::match, &oc);
	ROS_INFO("VISION: classify_service operational.");

	ros::ServiceServer find_object_service = nh.advertiseService("find_object", &ObjectClassification::findObject, &oc);
	ROS_INFO("VISION: find_object_service operational.");

	ros::ServiceServer contains_object = nh.advertiseService("contains_object", &ObjectClassification::containsObject, &oc);
	ROS_INFO("VISION: contains_object_service operational.");

	ros::ServiceServer get_objects_in_scene_service = nh.advertiseService("get_objects_in_scene", &ObjectClassification::getObjectsInScene, &oc);
	ROS_INFO("VISION: get_objects_in_scene_service operational.");

	ros::spin();
}

