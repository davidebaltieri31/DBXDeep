#pragma once
#pragma once
#include "Dataset.h"
#include "DBXLayer.h"
#include "DBXFullyConnected.h"
#include "DBXCost.h"
#include "DBXInputLayer.h"
#include "DBXSigmoid.h"
#include "DBXReLU.h"
#include "DBXSoftMax.h"
#include "DBXBatchNormalizationLayer.h"
#include "DBXConvolutionLayer.h"
#include "DBXConvVizLayer.h"
#include "DBXMaxPooling.h"
#include <chrono>

namespace DBX
{
	namespace NN
	{
		class Network {
		protected:
			std::vector<std::shared_ptr<Layer>> m_network;
			std::shared_ptr<Cost> m_cost;
			//training
			std::shared_ptr<DatasetLoader> m_loader = nullptr;
			std::vector<std::pair<myMat, myMat>> training_data;
			std::vector<std::pair<myMat, myMat>> evaluation_data;
			std::vector<std::pair<myMat, myMat>> test_data;

			int maxVal(int r, Eigen::Tensor<float, 4> data) {
				double max_val = -DBL_MAX;
				int selected = -1;
				for (int i = 0; i < data.dimension(3); ++i) {
					if (data(r, 0, 0, i) > max_val) {
						max_val = data(r, 0, 0, i);
						selected = i;
					}
				}
				return selected;
			}

			int maxVal(Eigen::Tensor<float, 4> result, Eigen::Tensor<float, 4> gt) {
				int num_correct = 0;
				for (int j = 0; j < result.dimension(0); ++j) {
					int r1 = maxVal(j, result);
					int r2 = maxVal(j, gt);
					if (r1 == r2) ++num_correct;
				}
				return num_correct;
			}

			double evaluate(std::vector<std::pair<myMat, myMat>>& dataset) {
				int sum = 0;
				int mini_batch_size = 100;
				int data_cols = int(dataset[0].first.cols());
				int data_rows = int(dataset[0].first.rows());
				int label_size = int(dataset[0].second.cols());
				int num_batches = int(dataset.size()) / mini_batch_size;
				Eigen::Tensor<float, 4> batch(mini_batch_size, 1, data_rows, data_cols);
				Eigen::Tensor<float, 4> batch_labels(mini_batch_size, 1, 1, label_size);
				for (int k = 0; k < num_batches; ++k)
				{
					for (int j = 0; j < mini_batch_size; ++j) {
						for (int y = 0; y < data_rows; ++y) {
							for (int x = 0; x < data_cols; ++x) {
								batch(j, 0, y, x) = dataset[k*mini_batch_size + j].first(y, x);
							}
						}
						for (int x = 0; x < label_size; ++x) {
							batch_labels(j, 0, 0, x) = dataset[k*mini_batch_size + j].second(0, x);
						}
					}

					Eigen::Tensor<float, 4> in = batch;
					for (auto& layer : m_network) {
						Eigen::Tensor<float, 4> out = layer->forward(in);
						in = out;
					}

					sum += maxVal(in, batch_labels);
				}
				return double(sum) / double(dataset.size());
			}

			void train_mini_batch(Eigen::Tensor<float, 4>& batch, Eigen::Tensor<float, 4>& batch_labels) {
				Eigen::Tensor<float, 4> in = batch;
				//forward pass
				for (auto& layer : m_network) {
					Eigen::Tensor<float, 4> out = layer->forward(in);
					in = out;
				}
				//backward pass
				int layers = int(m_network.size()) - 1;
				Eigen::Tensor<float, 4> cost_derivative = m_cost->cost_derivative(in, batch_labels);
				Eigen::Tensor<float, 4> in_back = cost_derivative;
				for (int i = layers; i >= 0; --i) {
					Eigen::Tensor<float, 4> out = m_network[i]->backprop(in_back);
					in_back = out;
				}
				for (int i = layers; i >= 0; --i) {
					m_network[i]->update(int(batch.dimension(0)));
				}
			}

			float train_epoch(int epoch, Params& training_params) {
				std::chrono::time_point<std::chrono::system_clock> t_a = std::chrono::system_clock::now();
				for (auto& layer : m_network) {
					layer->reset(training_params);
				}
				//random shuffle of dataset
				std::random_shuffle(training_data.begin(), training_data.end());
				//vars
				int mini_batch_size = training_params.get_int_param("mini_batch_size");
				int data_cols = int(training_data[0].first.cols());
				int data_rows = int(training_data[0].first.rows());
				int label_size = int(training_data[0].second.cols());
				int num_batches = int(training_data.size()) / mini_batch_size;
				int n_test = int(evaluation_data.size());

				set_network_to_training();

				Eigen::Tensor<float, 4> batch(mini_batch_size, 1, data_rows, data_cols);
				Eigen::Tensor<float, 4> batch_labels(mini_batch_size, 1, 1, label_size);
				std::cout << std::endl << "batch 0 of " << num_batches;
				uint64_t elapsed_millis = 0;
				for (int k = 0; k < num_batches; ++k)
				{
					std::chrono::time_point<std::chrono::system_clock> te_a = std::chrono::system_clock::now();
					for (int j = 0; j < mini_batch_size; ++j) {
						for (int y = 0; y < data_rows; ++y) {
							for (int x = 0; x < data_cols; ++x) {
								batch(j, 0, y, x) = training_data[k*mini_batch_size + j].first(y, x);
							}
						}
						for (int x = 0; x < label_size; ++x) {
							batch_labels(j, 0, 0, x) = training_data[k*mini_batch_size + j].second(0, x);
						}
					}
					train_mini_batch(batch, batch_labels);
					std::chrono::time_point<std::chrono::system_clock> te_b = std::chrono::system_clock::now();
					elapsed_millis += std::chrono::duration_cast<std::chrono::milliseconds>(te_b - te_a).count();
					std::cout << "\r" << "batch " << k + 1 << " of " << num_batches << " in avg " << elapsed_millis / (k + 1) << "ms";
				}
				std::chrono::time_point<std::chrono::system_clock> t_b = std::chrono::system_clock::now();
				double val = -1.0;
				if (n_test > 0)
				{
					set_network_to_inference();
					val = evaluate(evaluation_data);
					std::cout << std::endl << "Epoch " << epoch << " precision eval set:" << val << std::endl;
					val = evaluate(test_data);
					std::cout << "Epoch " << epoch << " precision test set:" << val << std::endl;
				}
				else std::cout << std::endl << "Epoch " << epoch << " complete!" << std::endl;

				std::chrono::time_point<std::chrono::system_clock> t_c = std::chrono::system_clock::now();

				uint64_t elapsed_seconds1 = std::chrono::duration_cast<std::chrono::milliseconds>(t_b - t_a).count();
				uint64_t elapsed_seconds2 = std::chrono::duration_cast<std::chrono::milliseconds>(t_c - t_b).count();

				std::cout << "Time: " << elapsed_seconds1 << "::::" << elapsed_seconds2 << std::endl;
				return float(val);
			}

		public:
			void add_layer(Layer* layer) {
				if (m_network.size() > 0) layer->link(m_network[m_network.size() - 1], nullptr);
				else layer->link(nullptr, nullptr);
				m_network.push_back(std::shared_ptr<Layer>(layer));
			}

			void insert_layer(Layer* layer, Params& training_params) {
				layer->link(m_network[m_network.size() - 2], m_network[m_network.size() - 1]);
				m_network.insert(m_network.end() - 1, std::shared_ptr<Layer>(layer));
				m_network[m_network.size() - 2]->init(training_params);
				m_network[m_network.size() - 1]->init(training_params);
			}

			void set_loader(DatasetLoader* loader) {
				m_loader = std::shared_ptr<DatasetLoader>(loader);
			}

			bool load_dataset() {
				return m_loader->load(training_data, evaluation_data, test_data);
			}

			void add_input_layer() {
				add_layer(new InputLayer(m_loader));
			}

			void add_fully_connected_layer(int size, WeightInitializerType init_type, SGDUpdaterType sgd_type) {
				add_layer(new FullyConnectedLayer(init_type, sgd_type, size));
			}

			void add_sigmoid_layer() {
				add_layer(new SigmoidLayer());
			}

			void add_relu_layer() {
				add_layer(new ReLULayer());
			}

			void add_softmax_layer() {
				add_layer(new SoftMaxLayer());
			}

			void add_conv_layer(WeightInitializerType init_type, SGDUpdaterType sgd_type, int num_kernels, int kernel_width, int kernel_height) {
				add_layer(new ConvolutionLayer(init_type, sgd_type, num_kernels, kernel_width, kernel_height));
			}

			void add_batchnormalization_layer(SGDUpdaterType sgd_type) {
				add_layer(new BatchNormLayer(sgd_type));
			}

			void add_visualization_layer(int parent_layer_id) {
				add_layer(new ConVizLayer(parent_layer_id));
			}

			void add_max_pooling(int kernel_width, int kernel_height) {
				add_layer(new MaxPooling(kernel_width, kernel_height));
			}

			//void add_convolution_layer(int num_kernels, int kernel_width, int kernel_height, Neuron* neuron) {
			//	add_layer(new ConvolutionLayer(neuron, num_kernels, kernel_width, kernel_height));
			//}

			void add_cost_layer(Cost* cost) {
				m_cost = std::unique_ptr<Cost>(cost);
			}

			//void add_maxpooling(int kernel_width, int kernel_height) {
			//	add_layer(new MaxPoolingLayer(kernel_width, kernel_height));
			//}

			//void insert_fully_connected_layer(int size, Neuron* neuron) {
			//	insert_layer(new FullyConnectedLayer(neuron, size));
			//}

			bool init_network(Params& training_params) {
				training_params.set_float_param("num_training_sample", float(training_data.size()));
				bool success = true;
				for (auto& layer : m_network) {
					success = success && layer->init(training_params);
				}
				return success;
			}

			void set_network_to_training() {
				for (auto& layer : m_network) {
					layer->set_inference_mode(false);
				}
			}
			void set_network_to_inference() {
				for (auto& layer : m_network) {
					layer->set_inference_mode(true);
				}
			}

			myMat test(myMat data) {
				Eigen::Tensor<float, 4> in(1, 1, data.rows(), data.cols());
				for (int y = 0; y < data.rows(); ++y) {
					for (int x = 0; x < data.cols(); ++x) {
						in(0, 0, y, x) = data(y, x);
					}
				}
				for (auto& layer : m_network) {
					Eigen::Tensor<float, 4> out = layer->forward(in);
					in = out;
				}
				myMat res(1, in.dimension(3));
				for (int y = 0; y < in.dimension(3); ++y) {
					res(0, y) = in(0, 0, 0, y);
				}
				return res;
			}

			void train(Params& training_params) {
				
				int epochs = training_params.get_int_param("epochs");
				for (int i = 0; i < epochs; ++i) {
					float res = train_epoch(i, training_params);
				}
			}

			//void enable_viz(int layer, bool enabled) {
			//	m_network[layer]->enable_viz(enabled);
			//}
			void gradient_check_full(int layer, Params& training_params, float epsilon, int num) {
				std::chrono::time_point<std::chrono::system_clock> t_a = std::chrono::system_clock::now();
				for (auto& layer : m_network) {
					layer->reset(training_params);
				}
				//random shuffle of dataset
				std::random_shuffle(training_data.begin(), training_data.end());
				//vars
				int mini_batch_size = 1;
				int data_cols = int(training_data[0].first.cols());
				int data_rows = int(training_data[0].first.rows());
				int label_size = int(training_data[0].second.cols());
				int num_batches = int(training_data.size()) / mini_batch_size;
				int n_test = num;

				Eigen::Tensor<float, 4> batch(mini_batch_size, 1, data_rows, data_cols);
				Eigen::Tensor<float, 4> batch_labels(mini_batch_size, 1, 1, label_size);

				for (int j = 0; j < mini_batch_size; ++j) {
					for (int y = 0; y < data_rows; ++y) {
						for (int x = 0; x < data_cols; ++x) {
							batch(j, 0, y, x) = training_data[j].first(y, x);
						}
					}
					for (int x = 0; x < label_size; ++x) {
						batch_labels(j, 0, 0, x) = training_data[j].second(0, x);
					}
				}
				std::random_device rd;
				std::mt19937 r_gen(rd());
				for (int i = 0; i < n_test; ++i) {
					FullyConnectedLayer* clayer = (FullyConnectedLayer*)m_network[layer].get();
					std::uniform_int_distribution<int> r1(0, clayer->weight.dimension(0) - 1);
					std::uniform_int_distribution<int> r2(0, clayer->weight.dimension(1) - 1);
					int sel_dim0 = r1(r_gen);
					int sel_dim1 = r2(r_gen);
					//compute numerical cost
					float numerical_gradient = 0.0f;
					Eigen::Tensor<float, 2> A, B;
					{
						set_network_to_inference();
						float old_w = clayer->weight(sel_dim0, sel_dim1);
						clayer->weight(sel_dim0, sel_dim1) = old_w - epsilon;
						Eigen::Tensor<float, 4> in = batch;
						for (auto& layer : m_network) {
							Eigen::Tensor<float, 4> out = layer->forward(in);
							in = out;
						}
						A = m_cost->cost(in, batch_labels);

						clayer->weight(sel_dim0, sel_dim1) = old_w + epsilon;
						in = batch;
						for (auto& layer : m_network) {
							Eigen::Tensor<float, 4> out = layer->forward(in);
							in = out;
						}
						B = m_cost->cost(in, batch_labels);
						clayer->weight(sel_dim0, sel_dim1) = old_w;
					}
					numerical_gradient = (B(0, 0) - A(0, 0)) / (epsilon + epsilon);
					float computed_gradient = 0.0f;
					{
						set_network_to_training();
						Eigen::Tensor<float, 4> in = batch;
						//forward pass
						for (auto& layer : m_network) {
							Eigen::Tensor<float, 4> out = layer->forward(in);
							in = out;
						}
						//backward pass
						int layers = int(m_network.size()) - 1;
						Eigen::Tensor<float, 4> cost_derivative = m_cost->cost_derivative(in, batch_labels);
						Eigen::Tensor<float, 4> in_back = cost_derivative;
						for (int i = layers; i >= 0; --i) {
							Eigen::Tensor<float, 4> out = m_network[i]->backprop(in_back);
							in_back = out;
						}

						computed_gradient = clayer->nabla_w(sel_dim0, sel_dim1);
					}
					std::cout << "GradientCheck:" << computed_gradient << " vs " << numerical_gradient << " diff " << numerical_gradient - computed_gradient << std::endl;
				}
			}

			void gradient_check_conv(int layer, Params& training_params, float epsilon, int num) {
				std::chrono::time_point<std::chrono::system_clock> t_a = std::chrono::system_clock::now();
				for (auto& layer : m_network) {
					layer->reset(training_params);
				}
				//random shuffle of dataset
				std::random_shuffle(training_data.begin(), training_data.end());
				//vars
				int mini_batch_size = 1;
				int data_cols = int(training_data[0].first.cols());
				int data_rows = int(training_data[0].first.rows());
				int label_size = int(training_data[0].second.cols());
				int num_batches = int(training_data.size()) / mini_batch_size;
				int n_test = num;

				Eigen::Tensor<float, 4> batch(mini_batch_size, 1, data_rows, data_cols);
				Eigen::Tensor<float, 4> batch_labels(mini_batch_size, 1, 1, label_size);

				for (int j = 0; j < mini_batch_size; ++j) {
					for (int y = 0; y < data_rows; ++y) {
						for (int x = 0; x < data_cols; ++x) {
							batch(j, 0, y, x) = training_data[j].first(y, x);
						}
					}
					for (int x = 0; x < label_size; ++x) {
						batch_labels(j, 0, 0, x) = training_data[j].second(0, x);
					}
				}
				std::random_device rd;
				std::mt19937 r_gen(rd());
				for (int i = 0; i < n_test; ++i) {
					ConvolutionLayer* clayer = (ConvolutionLayer*)m_network[layer].get();
					std::uniform_int_distribution<int> r1(0, clayer->kernels.dimension(0) - 1);
					std::uniform_int_distribution<int> r2(0, clayer->kernels.dimension(1) - 1);
					std::uniform_int_distribution<int> r3(0, clayer->kernels.dimension(2) - 1);
					std::uniform_int_distribution<int> r4(0, clayer->kernels.dimension(3) - 1);
					int sel_dim0 = r1(r_gen);
					int sel_dim1 = r2(r_gen);
					int sel_dim2 = r3(r_gen);
					int sel_dim3 = r4(r_gen);
					//compute numerical cost
					float numerical_gradient = 0.0f;
					Eigen::Tensor<float, 2> A, B;
					{
						set_network_to_inference();
						float old_w = clayer->kernels(sel_dim0, sel_dim1, sel_dim2, sel_dim3);
						clayer->kernels(sel_dim0, sel_dim1, sel_dim2, sel_dim3) = old_w - epsilon;
						Eigen::Tensor<float, 4> in = batch;
						for (auto& layer : m_network) {
							Eigen::Tensor<float, 4> out = layer->forward(in);
							in = out;
						}
						A = m_cost->cost(in, batch_labels);

						clayer->kernels(sel_dim0, sel_dim1, sel_dim2, sel_dim3) = old_w + epsilon;
						in = batch;
						for (auto& layer : m_network) {
							Eigen::Tensor<float, 4> out = layer->forward(in);
							in = out;
						}
						B = m_cost->cost(in, batch_labels);
						clayer->kernels(sel_dim0, sel_dim1, sel_dim2, sel_dim3) = old_w;
					}
					numerical_gradient = (B(0, 0) - A(0, 0)) / (epsilon + epsilon);
					float computed_gradient = 0.0f;
					{
						set_network_to_training();
						Eigen::Tensor<float, 4> in = batch;
						//forward pass
						for (auto& layer : m_network) {
							Eigen::Tensor<float, 4> out = layer->forward(in);
							in = out;
						}
						//backward pass
						int layers = int(m_network.size()) - 1;
						Eigen::Tensor<float, 4> in_back = batch_labels;
						for (int i = layers; i >= 0; --i) {
							Eigen::Tensor<float, 4> out = m_network[i]->backprop(in_back);
							in_back = out;
						}

						computed_gradient = clayer->nabla_w(sel_dim0, sel_dim1, sel_dim2, sel_dim3);
					}
					std::cout << "GradientCheck:" << computed_gradient << " vs " << numerical_gradient << " diff " << numerical_gradient - computed_gradient << std::endl;
				}
			}
		};
	}
}