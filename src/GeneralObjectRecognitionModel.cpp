

#include "GeneralObjectRecognitionModel.h"
#include "Normalization.h"

GeneralObjectRecognitionModel::~GeneralObjectRecognitionModel()
{

}

GeneralObjectRecognitionModel::GeneralObjectRecognitionModel(Dataset d)
{
	this->dataset = d;
}

GeneralObjectRecognitionModel::GeneralObjectRecognitionModel()
{

}

void GeneralObjectRecognitionModel::addExperiment(Experiment& e)
{

}

void GeneralObjectRecognitionModel::train()
{
	// TODO get rid of the SupervisedFeatures. We only need keypoints for matching the histograms
	// The supervised features will be used for training, though. 
	// For generating the clusters, we need the extracted features .

	// Normalize it.
	//std::vector<cv::Mat> normalized;
	std::vector<SupervisedFeatures> features;
	cv::TermCriteria TC( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 27, 0.001 );
	cv::BOWKMeansTrainer bow ( 62, TC, 1, cv::KMEANS_PP_CENTERS);

	cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor);
	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector(400));
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BruteForceMatcher<cv::L2<float> >());

	//Normalization::differenceOfGaussians(normalized, 11, 151);
	FeatureExtractor::extractSURF(this->dataset.loaded_training_set, features);

	ROS_INFO("Featsize %d", features.size());
	for (int i =0; i < features.size(); i++)
	{
		
		cv::Mat feats = features[i].features;
		ROS_INFO("Feats dims %d %d", feats.rows, feats.cols);

		if (!feats.empty())
		{
			ROS_INFO("Featsdata");
			for (int k = 0; k < feats.rows; k++)
			{
				cv::Mat row = feats.row(k);
				ROS_INFO("Dims %d %d", row.rows, row.cols);
				row.convertTo(row, CV_32F);
				
				if (!row.empty())
				{
					bow.add(row);
				}
			}
		}		
	}

	ROS_INFO("Clustering.");
	cv::Mat vocabulary = bow.cluster();

	// Need to save the vocabulary. 
	
	cv::FileStorage fs("vocabulary.yml", cv::FileStorage::WRITE);
	fs << "vocabulary" << vocabulary;

	cv::BOWImgDescriptorExtractor bowide(extractor, matcher);
	bowide.setVocabulary(vocabulary);

	// Do matching
	cv::Mat training_matrix;
	cv::Mat label_matrix;
	//ObjectTrainer::computeTrainingMatrix(training_matrix, label_matrix);

	for (int i = 0; i < features.size(); i++)
	{
		SupervisedFeatures feats = features[i];
		std::vector<cv::KeyPoint> keypoints = feats.keypoints;
		cv::Mat response_histogram;
		
		bowide.compute(feats.original_image, keypoints, response_histogram);
		if (response_histogram.data)
		{
			training_matrix.push_back(response_histogram);
			ROS_INFO("Class %d %s", this->dataset.class_labels[feats.class_name], feats.class_name.c_str());
			label_matrix.push_back(this->dataset.class_labels[feats.class_name]);
		}
	}

	// Create the SVM so we can train it.
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

	CvSVM svm;

	ROS_INFO("Done clustering, starting training.");
	svm.train_auto( training_matrix, label_matrix, cv::Mat(), cv::Mat(), param, 10 );
	svm.save("svm");
}

void GeneralObjectRecognitionModel::demonstrate()
{
	// TODO	
	// Demonstrate should give a live demonstration of the program including things like
	// A live stream which recognizes which objects are in the stream and classifies the image
	// Should give top 5 things of which it thinks it is by pooling results over 30 frames (1 second)
	// and the classification will be what is the majority of that 30 frames.
}

void GeneralObjectRecognitionModel::show()
{
	// TODO
	// Should visualize the data
	// Visualize the classification, the clusters, and other things. 
}

void GeneralObjectRecognitionModel::confusionMatrix(ConfusionMatrix& mtx)
{
	// Make the confusion matrix for this svm. 
	CvSVM svm;	
	svm.load("svm");

	cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor);
	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector(400));
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BruteForceMatcher<cv::L2<float> >());

	cv::BOWImgDescriptorExtractor bowide(extractor, matcher);
	cv::FileStorage fs("vocabulary.yml", cv::FileStorage::READ);
	cv::Mat vocabulary;	
	fs["vocabulary"] >> vocabulary;
	bowide.setVocabulary(vocabulary);


	// predict against all of the loaded test images. 
	// Since we know their actual class we can make a confusion matrix. 
	std::multimap<std::string, cv::Mat>::iterator itr, itr_s;

	int ground_truth = 0;
	cv::Mat cf_matrix = cv::Mat::zeros((this->dataset.classes).size(), (this->dataset).classes.size(), CV_32S);
	mtx.confusion_matrix = cf_matrix;
	ROS_INFO("Generating confusion matrix");

	for (itr = this->dataset.loaded_test_set.begin(); itr != this->dataset.loaded_test_set.end(); itr = itr_s)
	{
		std::string key = (*itr).first;
		std::pair<std::multimap<std::string, cv::Mat>::iterator, std::multimap<std::string, 	cv::Mat>::iterator> key_range = this->dataset.loaded_test_set.equal_range(key);

		mtx.class_labels.push_back((*itr).first);
		for (itr_s = key_range.first; itr_s != key_range.second; ++itr_s)
		{
			std::string class_name = (*itr_s).first;
			cv::Mat image = (*itr_s).second;

			std::vector<cv::KeyPoint> keypoints;
			cv::Mat feats;
			
			detector->detect(image, keypoints);
			extractor->compute(image, keypoints, feats);
			bowide.compute(image, keypoints, feats);
		
			float prediction = svm.predict(feats);
			int pred = int(prediction);

			mtx.confusion_matrix.at<int>(ground_truth, pred)++;
		}
		ground_truth++;
	} 

	std::string print_output;
}

float GeneralObjectRecognitionModel::accuracy(ConfusionMatrix& matrix)
{
	// Make the confusion matrix for this svm. 
	CvSVM svm;	
	svm.load("svm");

	cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor);
	cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector(400));
	cv::Ptr<cv::DescriptorMatcher> matcher(new cv::BruteForceMatcher<cv::L2<float> >());

	cv::BOWImgDescriptorExtractor bowide(extractor, matcher);
	cv::FileStorage fs("vocabulary.yml", cv::FileStorage::READ);
	cv::Mat vocabulary;	
	fs["vocabulary"] >> vocabulary;
	bowide.setVocabulary(vocabulary);


	// predict against all of the loaded test images. 
	// Since we know their actual class we can make a confusion matrix. 
	std::multimap<std::string, cv::Mat>::iterator itr, itr_s;

	for (itr = this->dataset.loaded_test_set.begin(); itr != this->dataset.loaded_test_set.end(); itr = itr_s)
	{
		std::string key = (*itr).first;
		std::pair<std::multimap<std::string, cv::Mat>::iterator, std::multimap<std::string, 	cv::Mat>::iterator> key_range = this->dataset.loaded_test_set.equal_range(key);

		for (itr_s = key_range.first; itr_s != key_range.second; ++itr_s)
		{
			std::string class_name = (*itr_s).first;
			cv::Mat image = (*itr_s).second;

			std::vector<cv::KeyPoint> keypoints;
			cv::Mat feats;
			
			detector->detect(image, keypoints);
			extractor->compute(image, keypoints, feats);
			bowide.compute(image, keypoints, feats);
		
			float prediction = svm.predict(feats);
			ROS_INFO("Prediction %f", prediction);
		}
	} 
	return 0.0;
}

void GeneralObjectRecognitionModel::load(std::string model_path)
{
	// TODO
	// Should load all of the parameters of the model and resume the save points from the experiments
	// This is beneficial in the case of running thousands of experiments and not having to repeat ones that 
	// may take 1-2 hours per experiment.
}

void GeneralObjectRecognitionModel::save(std::string model_path)
{
	// TODO
	// Should save all of the parameters of the model and all the situational experiments
}

int main(int argc, char ** argv)
{
	ros::init(argc, argv, "general_object_recognition");
	ros::NodeHandle nh("~");
	// create the general object recognition model

	// The dataset automatically partitions it into a 70/30 split for generalization and cross validation testing
	Dataset data_dir("/home/marco/Downloads/data-10-classes");
	GeneralObjectRecognitionModel model(data_dir);
	model.train();

	ConfusionMatrix matrix;
	model.confusionMatrix(matrix);
	model.accuracy(matrix);
	
	std::string print_out;	
	matrix.print(print_out);
	// train model.
	// model.train();
	
	//model.train();
	//model.demonstrate();
	//model.show();
	//model.test();

	return 0;
}

