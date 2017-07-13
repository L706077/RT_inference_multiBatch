#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <string>
#include <utility>
#include <stdlib.h>
#include <stdio.h>

/////////////get class path////////////
#include <unistd.h>
#include <dirent.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"

//using namespace nvinfer1;
//using namespace nvcaffeparser1;

//using namespace cv;
typedef std::pair<std::string,float> mate;
#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}


// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 192;
static const int INPUT_W = 192;
static const int CHANNEL_NUM = 3;
//static const int OUTPUT_SIZE = 1498; //1498
int OUTPUT_SIZE = 1498; //1498    ********************Define by yourself*****************
//static const int BATCH_SIZE = 8;
static const int MaxBatchSize_ = 10;
int BATCH_SIZE;

const std::string Model_  = "fr_1498.caffemodel";
const std::string Deploy_ = "deploy.prototxt";
const std::string Mean_   = "mean.binaryproto";
const std::string Label_  = "labels.txt";
const std::string Path_   = "./fr_model/";

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "softmax";
//const char* OUTPUT_BLOB_NAME = "fc11_dropout";
//=================================================

//float prob[OUTPUT_SIZE];
//float *prob=(float*)malloc(OUTPUT_SIZE*sizeof(float));

//================================================
// Logger for GIE info/warning/errors
class Logger : public nvinfer1::ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;


std::string locateFile(const std::string& input)
{
	std::string file = Path_ + input;
	struct stat info;
	int i, MAX_DEPTH = 1;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

	return file;
}


void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 nvinfer1::IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
															  locateFile(modelFile).c_str(),
															  *network,
															  nvinfer1::DataType::kFLOAT);

	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 30); //1 << 20

	//builder->setHalf2Mode(true);

	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	//////(TensorRT1.0)// engine->serialize(gieModelStream);
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	nvcaffeparser1::shutdownProtobufLibrary();
}


void doInference(nvinfer1::IExecutionContext& context, float* input, float* output, int batchSize)
{
	const nvinfer1::ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * CHANNEL_NUM * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * CHANNEL_NUM * sizeof(float), cudaMemcpyHostToDevice, stream));

	context.enqueue(batchSize, buffers, stream, nullptr);

	//CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));

	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of p. */
static std::vector<int> Argmax(const float *p, int N) {
  	std::vector<std::pair<float, int> > pairs;
  	for (size_t i = 0; i < OUTPUT_SIZE; ++i)
    		pairs.push_back(std::make_pair(p[i], i));
  	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  	std::vector<int> result;
  	for (int i = 0; i < N; ++i)
    		result.push_back(pairs[i].second);
  	return result;
}

void preprocess(cv::Mat& img, cv::Size input_geometry_)
{
	cv::Mat sample, sample_resized;
	input_geometry_ = cv::Size(INPUT_W, INPUT_H);

	if (img.channels() == 3 && CHANNEL_NUM == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && CHANNEL_NUM == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && CHANNEL_NUM == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && CHANNEL_NUM == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	if (sample.size() != input_geometry_)
	    cv::resize(sample, sample_resized, input_geometry_);

	else
	    sample_resized = sample;

	img = sample_resized;
}

std::vector<std::string> GetClassName(const std::string& InputFolder)
{
	std::vector<std::string> OutFolder;
	std::string path_find =  "./" + InputFolder;
	DIR* dir;
	struct dirent * ptr;
	dir=opendir(path_find.c_str());
	if (dir != NULL)
	{
		while ((ptr = readdir(dir)) != NULL)
		{
			std::string path_in(ptr->d_name);
			if(path_in == "." || path_in == "..")
			{
				//
			}
			else
			{
				OutFolder.push_back(path_in);
				std::cout<<"path_in:  "<<path_in<< std::endl;
			}
		}
	}
	return OutFolder;
}




int main(int argc, char** argv)
{
		clock_t t1, t2, t3, t4, t5, t6, t7;
		std::vector<cv::Mat> Imgs;
		std::vector<std::string> Imgs_name;
	

		bool GetFeature = true;
			std::string filename, InputFolder;
			std::cout << "input floder name?" << std::endl;
			std::cout << "./InputFolder/*" << std::endl;
			std::cout << ">>";
			std::cin >> InputFolder;
			std::vector<std::string> FolderVector;
			FolderVector = GetClassName(InputFolder);
			std::cout << "FolderVector.size()=" << FolderVector.size() << std::endl;

			std::cout << "./BatchSize/*" << std::endl;
			std::cout << ">>";
			std::cin >> BATCH_SIZE;

//=======================================================================
t1=clock();	

		// create a GIE model from the caffe model and serialize it to a stream
	    	nvinfer1::IHostMemory *gieModelStream{nullptr};
		// caffeToGIEModel(Deploy_, Model_, std::vector < std::string > { OUTPUT_BLOB_NAME }, BATCH_SIZE, gieModelStream);
		caffeToGIEModel(Deploy_, Model_, std::vector < std::string > { OUTPUT_BLOB_NAME }, MaxBatchSize_, gieModelStream);

t2=clock();
		// deserialize the engine 
		nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
		nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
	   	if (gieModelStream) gieModelStream->destroy();

		nvinfer1::IExecutionContext *context = engine->createExecutionContext();
		std::cout<<"engine builded!!!!"<< std::endl;

t3=clock();	
//========================================================================
		for (int i = 0; i < FolderVector.size(); i++)
		{		
				std::string path_find = "./" + InputFolder + "/" + FolderVector[i];
				DIR* dir;
				struct dirent * ptr;
				dir = opendir(path_find.c_str());
		
				if (dir != NULL)
				{	
					while ((ptr = readdir(dir)) != NULL)
					{	
						std::string path_in(ptr->d_name);
						if (path_in == "." || path_in == "..")
						{
							//
						}
						else
						{		
							std::string CheckPath="./" + InputFolder + "/" + FolderVector[i] + "/" + path_in;
							cv::Mat img_input = cv::imread("./" + InputFolder + "/" + FolderVector[i] + "/" + path_in, 1);

							if (!img_input.empty())
							{
									cv::Mat sample,Img;
									cv::Size input_geometry_;
									input_geometry_ = cv::Size(INPUT_W, INPUT_H);
									preprocess(img_input,input_geometry_);
									Img = img_input;
									Imgs.push_back(Img);
									Imgs_name.push_back(path_in);
							}
						}
					}
				}
		
				closedir(dir);
		}

		std::cout<<"Imgs.size(): "<<Imgs.size()<<std::endl;
//========================================================================

		unsigned int fileData[Imgs.size()*INPUT_H*INPUT_W*CHANNEL_NUM];
		int num_time=0; 
		cv::Mat channel[CHANNEL_NUM];
		for(int p_n=0;p_n<Imgs.size();p_n++)
		{
			cv::split(Imgs[p_n],channel);
			for(int k=0;k<CHANNEL_NUM;k++)
			{	
				for(int i=0;i<INPUT_H;i++)
				{
					for(int j=0;j<INPUT_W;j++)
					{
						fileData[num_time]=(int)channel[k].at<uchar>(i,j);
						num_time++;			
					}
				}
			}
		}
		
		std::cout<<" total image pixels : "<< num_time<<std::endl;
//========================================================================
				
		// parse the mean file 
		nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
		nvcaffeparser1::IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile(Mean_).c_str());
		parser->destroy();
		const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());
		
//========================================================================

		float data[Imgs.size()*INPUT_H*INPUT_W*CHANNEL_NUM];
		num_time=0;
		for (int i = 0; i < Imgs.size();i++)
		{
			for(int j=0;j<INPUT_H*INPUT_W*CHANNEL_NUM;j++)
			{	
				data[num_time] = float(fileData[(i*INPUT_H*INPUT_W*CHANNEL_NUM)+j])-meanData[j];
				num_time++;
			}
		}
		meanBlob->destroy();
		
		float *prob=(float*)malloc(Imgs.size()*OUTPUT_SIZE*sizeof(float));		
t4=clock();
		doInference(*context,data, prob, BATCH_SIZE);
t5=clock();


////============================================================================================
////============================================================================================

		//load labels.txt data
		std::vector<std::string> labels_;
		std::ifstream labels(Path_ + Label_);
		std::string line;
		while (std::getline(labels, line))
			labels_.push_back(std::string(line));	

		float img_prob[OUTPUT_SIZE];
		for(int i=0; i<Imgs.size(); i++)
		{
			int x=i*OUTPUT_SIZE;
			int y=(i+1)*OUTPUT_SIZE;
			int run_time=0;

			//cut concat inference feature vector
			for(int j=x; j<y; j++)
			{
				img_prob[run_time]=prob[j];
				run_time++;
			}
				
			//find top 5
			std::vector<int>maxN=Argmax(img_prob,5);  //find top 5 probility IDs index[1498] (labels.txt)
			std::vector<mate> predictions;            //typedef std::pair<std::string,float> mate;
			
			for(int k=0; k<5; k++)
			{
				int idx=maxN[k];
				predictions.push_back(std::make_pair(labels_[idx],img_prob[idx]));   //make_pair( label IDs name , probility value)
			}
	
			/* Print the top N predictions. */
			std::cout << "------ Prediction for :"<< Imgs_name[i] << " ------" << std::endl;
			for (size_t l = 0; l < predictions.size(); ++l) 
			{
				mate pred_result = predictions[l];
				std::cout << std::fixed << pred_result.second << " - \"" << pred_result.first << "\"" << std::endl;
			}
		}
		
		// destroy the engine
		context->destroy();
		engine->destroy();
		runtime->destroy();
		free(prob);

////============================================================================================
////============================================================================================
t6=clock();

		std::cout<<"t2-t1 time:"<<(double)(t2-t1)/(CLOCKS_PER_SEC)<<"s"<<" (create GIE model) "<<std::endl;
		std::cout<<"t3-t2 time:"<<(double)(t3-t2)/(CLOCKS_PER_SEC)<<"s"<<" (build engine "<<std::endl;
		std::cout<<"t4-t3 time:"<<(double)(t4-t3)/(CLOCKS_PER_SEC)<<"s"<<" (read image and mean) "<<std::endl;
		std::cout<<"t5-t4 time:"<<(double)(t5-t4)/(CLOCKS_PER_SEC)<<"s"<<" (doInference) "<<std::endl;
		std::cout<<"t6-t5 time:"<<(double)(t6-t5)/(CLOCKS_PER_SEC)<<"s"<<" (show top 5 result) "<<std::endl;
		std::cout<<"t6-t1 time:"<<(double)(t6-t1)/(CLOCKS_PER_SEC)<<"s"<<" (total time) "<<std::endl;
		return 0;
}





















