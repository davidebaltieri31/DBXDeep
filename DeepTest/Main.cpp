//#include "DBX_Neuron.h"
//#include "DBX_Cost.h"
//#include "DBX_Layer.h"
#include "Dataset.h"
//#include "DBX_Network.h"
#include <opencv2\opencv.hpp>
//#include "TensorTest.h"
#include "DBXNetwork.h"
#include "DBXCost.h"
#include "DBX_Viz.h"
int main(int argc, char** argv)
{
	ColorGradient::add_color(0.0f, cv::Vec3b(0,0,255));
	//ColorGradient::add_color(0.5f, cv::Vec3b(0, 255, 255));
	ColorGradient::add_color(1.0f, cv::Vec3b(0, 255, 255));

	DBX::NN::Network mn;
	mn.set_loader(new MNISTLoaderMat(0));
	mn.load_dataset();
	mn.add_input_layer();
	
	mn.add_conv_layer(DBX::WeightInitializerType::GlorotNormal, DBX::SGDUpdaterType::AdaDeltaSGDUpdater, 32, 5, 5);
	mn.add_visualization_layer(1);
	mn.add_batchnormalization_layer(DBX::SGDUpdaterType::AdaDeltaSGDUpdater);
	mn.add_relu_layer();
	mn.add_max_pooling(2, 2);
    
	mn.add_conv_layer(DBX::WeightInitializerType::GlorotNormal, DBX::SGDUpdaterType::AdaDeltaSGDUpdater, 64, 3, 3);
	mn.add_batchnormalization_layer(DBX::SGDUpdaterType::AdaDeltaSGDUpdater);
	mn.add_relu_layer();

	mn.add_conv_layer(DBX::WeightInitializerType::GlorotNormal, DBX::SGDUpdaterType::AdaDeltaSGDUpdater, 64, 3, 3);
	mn.add_batchnormalization_layer(DBX::SGDUpdaterType::AdaDeltaSGDUpdater);
	mn.add_relu_layer();

	mn.add_max_pooling(2, 2);
	
	//mn.add_fully_connected_layer(400,DBX::WeightInitializerType::GlorotNormal,DBX::SGDUpdaterType::AdaDeltaSGDUpdater);
	//mn.add_batchnormalization_layer(DBX::SGDUpdaterType::AdaDeltaSGDUpdater);
	//mn.add_relu_layer();
	
	mn.add_fully_connected_layer(100, DBX::WeightInitializerType::GlorotNormal, DBX::SGDUpdaterType::AdaDeltaSGDUpdater);
	//mn.add_batchnormalization_layer(DBX::SGDUpdaterType::AdaDeltaSGDUpdater);
	mn.add_relu_layer();
	
	mn.add_fully_connected_layer(10, DBX::WeightInitializerType::GlorotNormal, DBX::SGDUpdaterType::AdaDeltaSGDUpdater);
	mn.add_softmax_layer();
	
	mn.add_cost_layer(new DBX::NN::QuadraticCost());
	
	Params training_params;
	training_params.set_int_param("num_threads", 8);
	training_params.set_int_param("num_cores", 4);
	training_params.set_float_param("weights_costant_scale", 1.0f);
	training_params.set_int_param("epochs",30);
	training_params.set_int_param("mini_batch_size", 100);
	training_params.set_float_param("momentum", 0.0f);
	training_params.set_float_param("learning_rate", 0.01f);
	training_params.set_float_param("lambda", 0.0f);
	training_params.set_float_param("sgd_delta_decay", 0.9f);
	training_params.set_float_param("sgd_delta_decay_2", 0.999f);
	mn.init_network(training_params);

	mn.train(training_params);

	//mn.gradient_check_full(3, training_params, 0.0001, 10);
	//mn.gradient_check_conv(1, training_params, 0.00001, 5);

	/*Network n;
	std::vector<int> layers;
	layers.push_back(784);
	layers.push_back(50);
	layers.push_back(10);
	n.init(layers, new SigmoidNeuron(), new CrossEntropyCost(new SigmoidNeuron()));
	n.SGD(training_data, 5, 10, 0.5, 5.0, 0.1, test_data);*/

	/*MultiLayerNetwork mn;
	mn.set_loader(new MNISTLoader());
	mn.load_dataset();
	mn.add_fully_connected_layer(50, new SigmoidNeuron());
	mn.add_fully_connected_output_layer(new SigmoidNeuron(), new CrossEntropyCost(new SigmoidNeuron()));
	mn.init_network();
	Params training_params;
	training_params.set_int_param("epochs", 5);
	training_params.set_int_param("mini_batch_size", 10);
	training_params.set_double_param("momentum", 0.1);
	training_params.set_double_param("eta", 0.5);
	training_params.set_double_param("lambda", 5.0);
	mn.train(training_params);*/

	cv::waitKey(-1);

	/*Network<SigmoidNeuron,CrossEntropyCost<SigmoidNeuron>> n;
	std::vector<int> layers;
	layers.push_back(784);
	layers.push_back(2500);
	layers.push_back(10);
	n.init(layers);

	//for (int i = 0; i < 100; ++i)
	//	n.SGD(training_data, 1, 10, i*0.003, test_data);
	n.SGD(training_data, 20, 10, 0.5, 5.0, 0.1, test_data);

	n.add_layer(2000);

	n.SGD(training_data, 20, 10, 0.2, 5.0, 0.1, test_data);
	
	n.add_layer(1500);

	n.SGD(training_data, 20, 10, 0.1, 5.0, 0.1, test_data);

	n.add_layer(1000);

	n.SGD(training_data, 20, 10, 0.1, 5.0, 0.1, test_data);

	n.add_layer(500);

	n.SGD(training_data, 20, 10, 0.1, 5.0, 0.1, test_data);*/

	/*int num_corretti = 0;
	for (int i = 0; i < test_data.size(); ++i) {
		myMat result = n.feedforward(test_data[i].first);
		int r1 = n.maxVal(result);
		int r2 = n.maxVal(test_data[i].second);
		if (r1 == r2) ++num_corretti;
	}

	std::cout << "precision:" << float(num_corretti) / float(test_data.size())*100.0f << "/%" << std::endl;
	*/
	return 0;
}