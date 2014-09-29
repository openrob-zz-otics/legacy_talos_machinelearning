// ObjectTrainer.cpp
// Written by Devon Ash
// Copyright of Thunderbots@HomeLeague, UBC
//=======================================================


#include <ObjectTrainer.hpp>
#include <unistd.h>
#include <time.h>

struct TrainingRow;
string CURRENT_DIRECTORY;

struct class_analysis
{
	int correct;
	int incorrect;
	int class_label;
	int total_samples;
};


map<int, struct class_analysis> class_analysis_map;


string ObjectTrainer::getClassName(string& directory)
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


char* ObjectTrainer::getFileType(char* filename)
{
	// Returns the file type of a given filename.
	char* stf = strtok(filename, ".");
	return strtok(NULL, ".");
}

void ObjectTrainer::trainSVM(Mat& trainingMatrix, Mat& trainingLabels) 
{

	//ROS_INFO("Training SVM...");

	CvTermCriteria tc = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001);
	CvSVMParams param = CvSVMParams();

	param.svm_type = CvSVM::NU_SVC;
	param.kernel_type = CvSVM::RBF;

	param.degree = 0; // For poly
	param.gamma = 9; // For poly/rgbf/sigmoid
	param.coef0 = 0;  // For poly/sigmoid

	param.C = 10000; // Optimization constant
	param.nu = 0.05;
	param.p = 0.0;

	param.class_weights = NULL;
	param.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	param.term_crit.max_iter = 1000;
	param.term_crit.epsilon = 1e-3;	

	//ROS_INFO("Training rows (#Images) %d Training cols (#Features): %d", trainingMatrix.rows, trainingMatrix.cols);
	//ROS_INFO("Labelled rows (#Images) %d", trainingLabels.rows);

	this->svm.train_auto( trainingMatrix, trainingLabels, Mat(), Mat(), param, 10 );
	this->svm.save( SVM_SAVE_NAME.c_str());
	//ROS_INFO("Trained SVM!");
	//ROS_INFO("Saving SVM as %s in your current directory.", SVM_SAVE_NAME.c_str());
	


}

void ObjectTrainer::normalize(string image_file, Mat& output_image)
{
	Mat image;

	image = imread( image_file.c_str(), CV_8UC1);
	if (!image.data) { ROS_INFO("Error reading Image %s", image_file.c_str()); return ;}
	// 3. DoG bandpass filter to avoid aliasing.
	Mat dog_mat;
	Mat dog1, dog2;
	GaussianBlur(image, dog1, Size(this->parameters.dog_lower,this->parameters.dog_lower), 0);
	GaussianBlur(image, dog2, Size(this->parameters.dog_upper, this->parameters.dog_upper), 0);
	dog_mat = (dog1 - dog2);
	// 4. Normalize image to standard resolution
	resize(dog_mat, output_image, Size(this->parameters.normalized_image_width, this->parameters.normalized_image_height)); 
	assert(output_image.type() == CV_8U);
}

void ObjectTrainer::computeTrainingMatrix(Mat& trainingMatrix, Mat& trainingLabels, Mat& vocabulary)
{
	
	//ROS_INFO("Computing training matrix");
	BOWImgDescriptorExtractor bowide(this->extractor, this->matcher);
	assert(vocabulary.type() == CV_32F);
	bowide.setVocabulary(vocabulary);

	//int divider = training_rows.size() - training_rows.size()*0.3;
	// Select 30% of training rows randomly, remove them from the vector, and put them into a testing row
	int test_size = training_rows.size()*0.3;

	//ROS_INFO("Using %d out of %d images for training (70 percent)", divider, training_rows.size());
	//ROS_INFO("Randomly selecting %d images from the training set for testing", test_size);
	
	vector<TrainingRow*> testing_rows;
	

		//ROS_INFO("Creating testing matrix");
		//ROS_INFO("Training size before test sample removal: %d", training_rows.size());
		for (int p = 0; p < test_size; p++)
		{
			int rand_val = rand() % training_rows.size();
			testing_rows.push_back(training_rows.at(rand_val));
			training_rows.erase(training_rows.begin()+rand_val);	
			//ROS_INFO("Added testing sample");	
			//int rand_val = rand() % training_rows.size();
			//testing_rows.push_back(training_rows.at(rand_val));
			//training_rows.erase(training_rows.begin()+rand_val);
		}
		//ROS_INFO("Training size after test sample removal: %d", training_rows.size());

		assert(testing_rows.size() == test_size);

		//ROS_INFO("Creating training matrix");
		for (int i = 0; i < training_rows.size(); i++)
		{

			Mat normalizedMat;
			Mat descriptors;	
			Mat response_histogram;
			response_histogram.convertTo(response_histogram, CV_32F);
			vector<KeyPoint> keypoints;

			const char* imageFilename = training_rows.at(i)->image_name.c_str();

			normalize(imageFilename, normalizedMat);
			detector->detect(normalizedMat, keypoints);
			bowide.compute(normalizedMat, keypoints, response_histogram);
			if (response_histogram.data)
			{
				response_histogram.convertTo(response_histogram, CV_32F);
				assert( response_histogram.type() == CV_32F);
				trainingMatrix.push_back(response_histogram);
				//ROS_INFO("Added class %d sample!", training_rows.at(i)->class_label);
				trainingLabels.push_back(training_rows.at(i)->class_label);
			}
		}

		// Training SVM
		//ROS_INFO("Training matrix generated");
		//ROS_INFO("Training SVM #$#$#$#");
		this->trainSVM(trainingMatrix, trainingLabels);
		// Done training SVM
		
		// Begin testing
		int correct = 0;
		int incorrect = 0;
		
		CvSVM svm;
		svm.load(SVM_SAVE_NAME.c_str());
		float num_test_images = this->testingMatrix.rows;

		// Initialized the analysis stuffs
		ROS_INFO("Initialized class analysis map");

		for (int i = 1; i < this->parameters.number_of_classes; i++)
		{
			struct class_analysis analysis;
			analysis.class_label = i;
			analysis.correct = 0;
			analysis.incorrect = 0;
			analysis.total_samples = 0;
			class_analysis_map.insert( pair< int, struct class_analysis>(i, analysis) );
		}

		for (int k = 0; k < testing_rows.size(); k++)
		{
			// Add this to the test data set.
			Mat normalizedMat;
			Mat descriptors;	
			Mat response_histogram;
			response_histogram.convertTo(response_histogram, CV_32F);
			vector<KeyPoint> keypoints;

			const char* imageFilename = testing_rows.at(k)->image_name.c_str();

			normalize(imageFilename, normalizedMat);
			detector->detect(normalizedMat, keypoints);
			bowide.compute(normalizedMat, keypoints, response_histogram);

			if (response_histogram.data)
			{
				
				class_analysis_map[testing_rows.at(k)->class_label].total_samples++;

				response_histogram.convertTo(response_histogram, CV_32F);
				assert( response_histogram.type() == CV_32F);
				//this->testingMatrix.push_back(response_histogram);
				//this->testingLabels.push_back(testing_rows.at(k)->class_label);

				// Added stuff below
				int prediction = svm.predict(response_histogram);

				if (prediction == testing_rows.at(k)->class_label) 
				{
				 	correct++; 

					class_analysis_map[testing_rows.at(k)->class_label].correct++;
				} else 
				{ 
					incorrect++; 	
					class_analysis_map[testing_rows.at(k)->class_label].incorrect++;
				}

					
			}


		}
	
		ROS_INFO("################ Model Information ##################");
		ROS_INFO("Number of classes:  %d", this->parameters.number_of_classes-1);
		ROS_INFO("Number of clusters: %d", this->parameters.number_of_clusters);
		ROS_INFO("Samples per class:  %d", this->parameters.number_of_samples_per_class);
		ROS_INFO("DOG_LOWER_KERNEL:   %d", this->parameters.dog_lower);
		ROS_INFO("DOG_UPPER_KERNEL:   %d", this->parameters.dog_upper);
		ROS_INFO("Image height:       %d", this->parameters.normalized_image_height);
		ROS_INFO("Image width:        %d", this->parameters.normalized_image_width);
		 

		ROS_INFO("############## Experimental Information #############");
		ROS_INFO("Total correct classifications %d", correct);
		ROS_INFO("Total incorrect classifications %d", incorrect);
		
		for (int i = 1; i < this->parameters.number_of_classes; i++)
		{
	
			struct class_analysis analysis = class_analysis_map[i];


			ROS_INFO("----------------------------------");	
			ROS_INFO("Class: %s", class_labels[analysis.class_label].c_str());
			ROS_INFO("Correct: %d", (analysis.total_samples-analysis.incorrect));
			ROS_INFO("Incorrect: %d", analysis.incorrect);
			ROS_INFO("Total tested: %d", analysis.total_samples);

			ROS_INFO("Accuracy on %s is [%d%]", class_labels[analysis.class_label].c_str(), ((analysis.correct*100)/analysis.total_samples));

		}

		ROS_INFO("Analysis complete");

		float ovrall = (float)correct / (correct+incorrect);

		ROS_INFO("Overall accuracy %f", ovrall);
}


// For all the folders in the data directory	
	// Process all the images in that directory
	// The class of that image is the name of the folder and its label
	// is the index of that directory
void ObjectTrainer::computeVocabulary(Mat& output_vocabulary)
{

	TermCriteria TC( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 27, 0.001 );
	BOWKMeansTrainer bow ( KMEANS_CLUSTERS, TC, KMEANS_ATTEMPTS, KMEANS_PP_CENTERS);

	int class_index = 1;

	const char* PATH = DATA_DIRECTORY.c_str();
	//ROS_INFO("DATA DIR %s", PATH);
	DIR *dir = opendir(PATH);
	struct dirent *entry = readdir(dir);

	while (entry != NULL)
	{
		// Iterate over that directory to parse its images
		if (entry->d_type == DT_DIR && (strcmp(entry->d_name, ".") != 0) && (strcmp(entry->d_name,"..") != 0))
		{
			// Print out the inner directories
			ROS_INFO("Image directory: %s Index: %d", entry->d_name, class_index);
			
			string temp_dir;
			temp_dir = DATA_DIRECTORY;
			temp_dir.append("/");
			temp_dir.append(entry->d_name);
			string image_class_directory(temp_dir);
			DIR *inner_dir = opendir(image_class_directory.c_str());

			struct dirent *inner_entry = readdir(inner_dir);

			while (inner_entry != NULL)
			{

				class_labels.insert( pair< int, string>(class_index, entry->d_name));

				if (strcmp(inner_entry->d_name, ".") != 0 && strcmp(inner_entry->d_name, "..") != 0)
				{	

					string inner_temp_dir;
					temp_dir = DATA_DIRECTORY+"/";
					temp_dir.append(entry->d_name).append("/").append(inner_entry->d_name);
					char* filetype = getFileType(inner_entry->d_name);
					if ((filetype != NULL) && strcmp(filetype, "jpg") == 0)
					{
						// This will be the image
						NUM_IMAGES++;

	
						TrainingRow* row = new TrainingRow;
						row->image_name = temp_dir;
						row->class_label = class_index;

						training_rows.push_back(row);

						Mat descriptors;
						Mat normalizedMat;

						// Uncomment for visualization.
						//Mat pre_norm;
						//pre_norm = imread(row->image_name.c_str());
						//imshow("Pre-Normalization", pre_norm);
						//waitKey(0);

						normalize(row->image_name, normalizedMat);

						//imshow("Normalized", normalizedMat);
						//waitKey(0);

						vector<KeyPoint> keypoints;
						detector->detect(normalizedMat, keypoints);
						extractor->compute(normalizedMat, keypoints, descriptors);
						for (int i = 0; i < descriptors.rows; i++)
						{
							if (!descriptors.empty()) 
							{ 
								Mat m = descriptors.row(i);
								m.convertTo(m, CV_32F);
								bow.add(m); 	
							//	ROS_INFO("Added descriptors");
							}
						}
						//Mat keypointed;
						//drawKeypoints(normalizedMat, keypoints, keypointed, Scalar(255,255,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		
						//imshow("Keypointed image", keypointed);
						//waitKey(0);
						
					}
				}

				inner_entry = readdir(inner_dir);
		
				//if ((NUM_IMAGES % this->parameters.number_of_samples_per_class) == 0)
				//{
			//		ROS_INFO("Reached the maximum allowable samples per class %d: ", this->parameters.number_of_samples_per_class);
			//		break;
			//	}
		
			}

			class_index++;
			closedir(inner_dir);
		}
		entry = readdir(dir);	
	}

	this->parameters.number_of_classes = class_index;


	output_vocabulary = bow.cluster();
	
	if (output_vocabulary.data)
	{
		//ROS_INFO("Vocabulary made.");
	}

	FileStorage fs(VOCABULARY_SAVE_NAME+".yml", FileStorage::WRITE);
	fs << "vocabulary" << output_vocabulary;

	closedir(dir);
}

void ObjectTrainer::computeHistogram(string image_file, Mat& vocabulary, Mat& response_histogram)
{
	//ROS_INFO("Computing histogram");

	BOWImgDescriptorExtractor bowide(this->extractor, this->matcher);
	bowide.setVocabulary(vocabulary);

	Mat normalized_image;
	normalize(image_file, normalized_image);			
	
	vector<KeyPoint> keypoints;
	detector->detect(normalized_image, keypoints);	
	
	bowide.compute(normalized_image, keypoints, response_histogram);
	//ROS_INFO("Computed histogram");

}

float ObjectTrainer::testSVM()
{
	//ROS_INFO("Testing SVM implementation.");


	CvSVM svm;
	svm.load(SVM_SAVE_NAME.c_str());
	
	float num_test_images = testingMatrix.rows;
	float correct;
	float incorrect;


	for (int m = 0; m < (int)num_test_images; m++)
	{
	
		int prediction = svm.predict(testingMatrix.row(m));
		if (prediction == testingLabels.at<int>(m, 0)) { correct++; } else { incorrect++; }
		ROS_INFO("Predicted class: %d Actual class: %d", prediction, testingLabels.at<int>(m, 0));
		
		// Show the image it thought was another class
		
	}

	
	ROS_INFO("################ Model Information ##################");
	ROS_INFO("Number of classes:  %d", this->parameters.number_of_classes-1);
	ROS_INFO("Number of clusters: %d", this->parameters.number_of_clusters);
	ROS_INFO("Samples per class:  %d", this->parameters.number_of_samples_per_class);
	ROS_INFO("DOG_LOWER_KERNEL:   %d", this->parameters.dog_lower);
	ROS_INFO("DOG_UPPER_KERNEL:   %d", this->parameters.dog_upper);
	ROS_INFO("Image height:       %d", this->parameters.normalized_image_height);
	ROS_INFO("Image width:        %d", this->parameters.normalized_image_width);
	 

	ROS_INFO("############## Experimental Information #############");
	ROS_INFO("Correct classifications %f", correct);
	ROS_INFO("Incorrect classifications %f", incorrect);
	

	return (float)(correct/num_test_images);
}


ObjectTrainer::~ObjectTrainer() 
{
	for (int k = 0; k < training_rows.size(); k++) { delete training_rows.at(k); }
}

ObjectTrainer::ObjectTrainer()
: SVM_SAVE_NAME("TrainedSVM"), VOCABULARY_SAVE_NAME("vocabulary"), KMEANS_ATTEMPTS(3), KMEANS_CLUSTERS(500), NUM_IMAGES(0) 
{

}

ObjectTrainer::ObjectTrainer(string svm_save, string vocab_save, string data_dir, struct ModelParameters model_params) 
: NUM_IMAGES(0), SVM_SAVE_NAME("TrainedSVM"), VOCABULARY_SAVE_NAME("vocabulary")
{
	//SVM_SAVE_NAME = CURRENT_DIRECTORY + "/config/TrainedSVM";
	//VOCABULARY_SAVE_NAME = CURRENT_DIRECTORY + "/config/vocabulary.yml";

	this->parameters = model_params;

	SVM_SAVE_NAME = svm_save;
	VOCABULARY_SAVE_NAME = vocab_save;
	DATA_DIRECTORY = data_dir;

	//ROS_INFO("Saving SVM to %s", SVM_SAVE_NAME.c_str());
	//ROS_INFO("Saving Vocabulary to %s", VOCABULARY_SAVE_NAME.c_str());

	Ptr<FeatureDetector> detector(new SurfFeatureDetector( 400 ));
	Ptr<DescriptorExtractor> extractor(new SurfDescriptorExtractor);
	Ptr<DescriptorMatcher> matcher(new BruteForceMatcher<L2<float> >());

	KMEANS_CLUSTERS = model_params.number_of_clusters;
	KMEANS_ATTEMPTS = 3;
	this->extractor = extractor;
	this->matcher = matcher;
	this->detector = detector;

	//ROS_INFO("Computing vocab..");
	this->computeVocabulary(vocabulary);
	this->computeTrainingMatrix(trainingMatrix, trainingLabels, vocabulary);
	//this->trainSVM(trainingMatrix, trainingLabels);
	//float accuracy_pct = this->testSVM();
	//ROS_INFO("Accuracy of SVM %f percent", accuracy_pct);
}


int main(int argc, char** argv)
{

	clock_t begin, end;
	begin = clock();
	ros::init(argc, argv, "object_trainer_test");
	ros::NodeHandle nh("~");


	string data_dir;
	string vocabulary_dir;
	string svm_dir;
	string result_savename;

	int dog_lower = 0;
	int dog_upper = 0;
	int number_of_samples_per_class = 0;
	int number_of_clusters = 0;
	int normalized_image_height = 0;
	int normalized_image_width = 0;

	struct ModelParameters model_params;

	if (strcmp(argv[1], "help") == 0)
	{
		ROS_INFO("Usage: object_trainer _data_directory:='directory_name' _vocabulary_directory:='directory_name' _svm_directory:='directory_name' _image_height:='height' _image_width:='width' _dog_lower:='lower' _dog_upper:='upper' _number_of_samples_per_class:='number' _result_savename='savename'");
		return 0;
	}

	nh.getParam("image_height", normalized_image_height);
	nh.getParam("image_width", normalized_image_width);
	nh.getParam("dog_lower", dog_lower);
	nh.getParam("dog_upper", dog_upper);
	nh.getParam("number_of_samples_per_class", number_of_samples_per_class);
	nh.getParam("number_of_clusters", number_of_clusters);
	nh.getParam("data_directory", data_dir);
	nh.getParam("vocabulary_directory", vocabulary_dir);
	nh.getParam("svm_directory", svm_dir);
	nh.getParam("result_savename", result_savename);

	ROS_INFO("HerE");
	if ((data_dir.empty() || svm_dir.empty() || vocabulary_dir.empty()))
	{
		//ROS_INFO("Either your SVM, Data, or Vocab directories were not correct. Please enter the correct parameters ***as absolute paths***.");
		//ROS_INFO("Usage: object_trainer _data_directory:='directory_name' _vocabulary_directory:='directory_name' _svm_directory:='directory_name'");
		return 0;
	}


	if (dog_lower == 0)
	{
		//ROS_INFO("Using default lower gaussian kernel value: 11");
		model_params.dog_lower = 11;
	} else
	{
		model_params.dog_lower = dog_lower;
	}

	if (dog_upper == 0)
	{
		//ROS_INFO("Using default upper gaussian kernel value: 151");
		model_params.dog_upper = 151;
	} else
	{
		model_params.dog_upper = dog_upper;
	}

	if (number_of_clusters == 0)
	{
		//ROS_INFO("Using default number of clusters: 3000");
		model_params.number_of_clusters = 500;
	} else
	{
		model_params.number_of_clusters = number_of_clusters;
	}

	if (number_of_samples_per_class == 0)
	{
		//ROS_INFO("Using default number of samples per class: 100");
		model_params.number_of_samples_per_class = 50;
	} else
	{
		model_params.number_of_samples_per_class = number_of_samples_per_class;
	}

	if (normalized_image_height == 0)
	{
		//ROS_INFO("Using default image height: 256px");
		model_params.normalized_image_height = 256;
	} else
	{
		model_params.normalized_image_height = normalized_image_height;
	}

	if (normalized_image_width == 0)
	{
		//ROS_INFO("Using default image width: 256px");
		model_params.normalized_image_width = 256;

	} else
	{
		model_params.normalized_image_width = normalized_image_width;
	}

	//Finding the current directory
	char buf[PATH_MAX];
	string dir(getwd(buf));
	CURRENT_DIRECTORY = dir;
	//ROS_INFO("Current dir %s", CURRENT_DIRECTORY.c_str());
	//ROS_INFO("Creating vocabulary.yml and TrainedSVM.txt!");

	ObjectTrainer trainer(svm_dir, vocabulary_dir, data_dir, model_params);
	//ROS_INFO("Successfully created vocabulary.yml and TrainedSVM.txt!");
	
	end = clock();
	double time_spent;
	time_spent = (double)(end-begin)/CLOCKS_PER_SEC;

	ROS_INFO("Time taken to train: %f seconds", time_spent);

	return 0;
}

