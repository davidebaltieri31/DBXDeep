#include "Dataset.h"
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

	std::cin.ignore();
	return 0;
}