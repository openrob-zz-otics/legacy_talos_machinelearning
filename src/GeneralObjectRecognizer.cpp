// ObjectRecognizer.cpp
// Written by Devon Ash
// Copyright of Thunderbots@HomeLeague, UBC
//=======================================================

#include "GeneralObjectRecognizer.hpp"
#include <map>

using namespace std;
using namespace cv;

ObjectRecognizer * recognizer;
string CURRENT_DIRECTORY;

string ObjectRecognizer::getClassName(string directory)
{

	// Returns last directory name
	istringstream iss(directory);
	string token;
	string last;
	while (getline(iss, token, '/'))
	{
		last = token;
	}
	return last;
}

void ObjectRecognizer::normalize(Mat& input_image, Mat& output_image)
{
	if (!input_image.data) 
	{
		 ROS_INFO("Error with input image"); 
		return; 
	}
	
	// 3. DoG bandpass filter to avoid aliasing.
	Mat dog_mat;
	Mat dog1, dog2;
	GaussianBlur(input_image, dog1, Size(11, 11), 0);
	GaussianBlur(input_image, dog2, Size(151, 151), 0);
	dog_mat = (dog1 - dog2);

	// 4. Normalize image to standard resolution
	resize(dog_mat, output_image, Size(NORMALIZED_HEIGHT, NORMALIZED_WIDTH)); 
}

void ObjectRecognizer::generateClassLabels(string data_dir)
{
	/*for (int k = 0; k < argc; k++)
	{
		string d(argv[k]);
		ROS_INFO("Response %d is %s", k, argv[k]);
		this->class_labels.insert( pair < int, string >(k, this->getClassName(d) ));
	}*/

	DIR *dir = opendir(data_dir.c_str());
	struct dirent *entry = readdir(dir);

	int class_index = 0;

	while (entry != NULL)
	{
		// Get the class labels	
		if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
		{
			string temp_dir(data_dir + entry->d_name);		
	
			this->class_labels.insert( pair <int, string>(class_index, this->getClassName(temp_dir)));
		
			ROS_INFO("Response %d is %s", class_index, this->getClassName(temp_dir).c_str());
		}

		entry = readdir(dir);
		class_index++;
	}
}

ObjectRecognizer::~ObjectRecognizer()
{

}

ObjectRecognizer::ObjectRecognizer()
{

}

// Input: The number of classes and the class labels [Directory names in this case]
ObjectRecognizer::ObjectRecognizer(string svm_path, string vocab_path, string data_dir)
{

	Ptr<FeatureDetector> detector(new SurfFeatureDetector(400) );
	Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor);
	Ptr<DescriptorMatcher> matcher(new BruteForceMatcher<L2<float> >());

	this->extractor = extractor;
	this->matcher = matcher;
	this->detector = detector;

	SVM_SAVE_NAME = svm_path;
	VOCABULARY_SAVE_NAME = vocab_path;
	DATA_DIRECTORY = data_dir;	

	this->generateClassLabels(data_dir);
}

void ObjectRecognizer::startStream()
{


	// Load SVM
	// Load vocabulary
	// Begin normalizing images.
	CvSVM svm;
	ROS_INFO("Starting video stream..");
	ROS_INFO("Loading %s SVM", SVM_SAVE_NAME.c_str());
	svm.load(SVM_SAVE_NAME.c_str());
	ROS_INFO("SVM Loaded!");

	ROS_INFO("Loading vocabulary %s.yml", VOCABULARY_SAVE_NAME.c_str());
	Mat vocabulary;		
	FileStorage fs("vocabulary_10000clusters.yml", FileStorage::READ);
	fs["vocabulary"] >> vocabulary;

	imshow("vocabulary", vocabulary);
	waitKey(0);

	BOWImgDescriptorExtractor bowide(this->extractor, this->matcher);

	bowide.setVocabulary(vocabulary);
	ROS_INFO("Set vocab");

//	imshow("vocabulary", vocabulary);
//	waitKey(0);

	// Iterate the data directory
	// 5 images from each folder
	// Classify them and drew feature descriptors on them. 
	
	struct dirent *entry;	
	DIR *dir = opendir(DATA_DIRECTORY.c_str());
	ROS_INFO("opening directory %s", DATA_DIRECTORY.c_str());	
	entry = readdir(dir);
	
	while (entry != NULL)
	{
		if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0)
		{
			ROS_INFO("Looking at %s", entry->d_name);
			struct dirent *inner_entry;
			string temp_dir(DATA_DIRECTORY);
			temp_dir.append(entry->d_name);
			DIR *inner_dir = opendir(temp_dir.c_str());
			inner_entry = readdir(inner_dir);
			
			// 5 sample classifications per class
			for (int i = 0; i < 5; i++)
			{

				if (strcmp(inner_entry->d_name, ".") != 0 && strcmp(inner_entry->d_name, "..") != 0)
				{
	
				string sample_path(temp_dir+"/"+inner_entry->d_name);			
				ROS_INFO("Sampple image path %s", sample_path.c_str());

				ROS_INFO("Opening image %s", sample_path.c_str());
				Mat sample = imread(sample_path.c_str());
				imshow("Pre-normalized", sample);
				waitKey(0);

				Mat normalized;
				normalize(sample, normalized);
				imshow("Normalized", normalized);
				waitKey(0);

				vector<KeyPoint> keypoints;
				Mat response;
				Mat keypointed;
				detector->detect(normalized, keypoints);
				bowide.compute(normalized, keypoints, response);

				drawKeypoints(normalized, keypoints, keypointed, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				imshow("Keypointed", keypointed);
				waitKey(0);
				
				bowide.compute(normalized, keypoints, response);

				if (!response.data)
				{
					ROS_INFO("Invalid Response: No object detected");
				}
				else
				{
					float resp = svm.predict(response, false);
					string class_name = this->class_labels.at((int)resp).c_str();
					ROS_INFO("Class %s detected with label [%f]", class_name.c_str(), resp);
					
				}

				}
					inner_entry = readdir(inner_dir);
			}	
			closedir(inner_dir);		
		}
		entry = readdir(dir);

	}
/*

	VideoCapture capture(0);
	if(!capture.isOpened()) return;
	cv::Mat mImage;
	int framecount = 0;

	while(1)
	{
	   	capture >> mImage;
		// Process image
		Mat normalized;
		normalize(mImage, normalized);
		vector<KeyPoint> keypoints;
		detector->detect(normalized, keypoints);

		Mat response;

		bowide.compute(normalized, keypoints, response);

		if (!response.data)
		{ 
			ROS_INFO("Invalid response %d", keypoints.size()); 
		} 
		else 
		{
			float resp = svm.predict(response);
			ROS_INFO("Response %f", resp);
			string class_name = this->class_labels.at((int)resp).c_str();
			ROS_INFO("Object: %s detected", class_name.c_str());

			
		} 

		Mat keypointed;
		drawKeypoints(mImage, keypoints, keypointed, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		imshow("Keypointed", keypointed);
		//imshow("Img",mImage);
         	waitKey(30);

	}
		*/
}

int main(int argc, char** argv)
{

	ros::init(argc, argv, "general_object_recognizer");
	ros::NodeHandle nh("~");

	string svm_path;
	string vocab_path;
	string data_path;

	if (strcmp(argv[1], "help") == 0)
	{
		ROS_INFO("USAGE: general_object_recognizer _data_path:=path_name _svm_path:=path_name _vocab_path:=path_name");
		return 0;
	}


	nh.getParam("data_path", data_path);
	nh.getParam("svm_path", svm_path);
	nh.getParam("vocab_path", vocab_path);

	if (svm_path.empty() || vocab_path.empty() || data_path.empty())
	{
		ROS_INFO("USAGE: general_object_recognizer _data_path:=path_name _svm_path:=path_name _vocab_path:=path_name");
		return 0;
	}

	recognizer = new ObjectRecognizer(svm_path, vocab_path, data_path);
	recognizer->startStream();
	ROS_INFO("Stream started");

	return 0;

}

