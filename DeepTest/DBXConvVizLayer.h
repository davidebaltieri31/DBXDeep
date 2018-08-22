#pragma once

#include "DBXLayer.h"
#include <ppl.h>
#include <opencv2\opencv.hpp>
#include "DBXConvolutionLayer.h"
#include "DBX_Viz.h"
#undef max
namespace DBX
{
	namespace NN
	{
		class ConVizLayer : public Layer
		{
#pragma region MEMBER VARIABLES
		protected:
			cv::Mat3b old_kernels;
			int layer_id;
#pragma endregion

#pragma region CONSTRUCTORS	
		public:

			ConVizLayer(int father_layer_id) : Layer() {
				layer_id = father_layer_id;
			}

			ConVizLayer(const ConVizLayer& l) = delete;

			ConVizLayer(ConVizLayer&& l) : Layer(std::move(l)) {
			}

			ConVizLayer& operator=(ConVizLayer& l) = delete;

			ConVizLayer& operator=(ConVizLayer&& l) {
				Layer::operator=(std::move(l));
				if (this != &l) {

				}
				return *this;
			}

			~ConVizLayer() {}
#pragma endregion

#pragma region MEMBER METHODS
			bool init(Params& params) {
				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_parent->output_tensor_dims[1];
				m_output_tensor_dims[2] = m_parent->output_tensor_dims[2];
				m_output_tensor_dims[3] = m_parent->output_tensor_dims[3];
				return true;
			}

			bool reset(Params& params) {
				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_parent->output_tensor_dims[1];
				m_output_tensor_dims[2] = m_parent->output_tensor_dims[2];
				m_output_tensor_dims[3] = m_parent->output_tensor_dims[3];
				old_kernels = cv::Mat3b();
				return true;
			}

			Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& batch) {
				return batch;
			}

			Eigen::Tensor<float, 4> backprop(Eigen::Tensor<float, 4>& gradient) {
				ConvolutionLayer* father = (ConvolutionLayer*)m_parent.get();
				if (father != nullptr) {
					float min_V = FLT_MAX;
					float max_V = -FLT_MAX;
					for (int j = 0; j < father->kernels.dimension(0); ++j) {
						for (int k = 0; k < father->kernels.dimension(1); ++k) {
							for (int y = 0; y < father->kernels.dimension(2); ++y) {
								for (int x = 0; x < father->kernels.dimension(3); ++x) {
									float v = father->kernels(j, k, y, x);
									min_V = (v < min_V) ? v : min_V;
									max_V = (v > max_V) ? v : max_V;
								}
							}
						}
					}
					{

						std::string name = std::string("kernels-layer-") + std::to_string(layer_id);
						cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
						int num_kernels = father->kernels.dimension(0)*father->kernels.dimension(1);
						int num_kernels_w = int(std::ceil(sqrt(num_kernels)));
						int num_kernels_h = int(std::ceil(sqrt(num_kernels)));
						int kernel_w = father->kernels.dimension(3);
						int kernel_h = father->kernels.dimension(2);
						cv::Mat3b kernels(num_kernels_h*kernel_h, num_kernels_w*kernel_w, cv::Vec3b(0, 0, 0));
						for (int j = 0; j < father->kernels.dimension(0); ++j) {
							for (int k = 0; k < father->kernels.dimension(1); ++k) {
								int i = j*father->kernels.dimension(1) + k;
								int ix = i % num_kernels_w;
								int iy = i / num_kernels_w;
								for (int y = 0; y < father->kernels.dimension(2); ++y) {
									for (int x = 0; x < father->kernels.dimension(3); ++x) {
										kernels(y + iy*kernel_h, x + ix*kernel_w) = ColorGradient::calc_color(father->kernels(j, k, y, x), min_V, max_V);
									}
								}
							}
						}
						cv::imshow(name, kernels);
						if (!old_kernels.empty()) {
							std::string name2 = std::string("kernels-layer-") + std::to_string(layer_id) + std::string("-diff");
							cv::namedWindow(name2, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
							cv::imshow(name2, kernels - old_kernels);
						}
						if (old_kernels.empty()) old_kernels = kernels;
					}

					{
						std::string name = std::string("kernels-layer-") + std::to_string(layer_id) + std::string("-values");
						cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
						int num_kernels = father->kernels.dimension(0)*father->kernels.dimension(1);
						int num_kernels_w = int(std::floor(sqrt(num_kernels)));
						int num_kernels_h = int(std::ceil(sqrt(num_kernels)));
						int kernel_w = father->kernels.dimension(3);
						int kernel_h = father->kernels.dimension(2);
						int y_cell = 30;
						int x_cell = 100;
						cv::Mat3b kernels(num_kernels_h*kernel_h*y_cell, num_kernels_w*kernel_w*x_cell, cv::Vec3b(0, 0, 0));
						for (int j = 0; j < father->kernels.dimension(0); ++j) {
							for (int k = 0; k < father->kernels.dimension(1); ++k) {
								int i = j*father->kernels.dimension(1) + k;
								int ix = i % num_kernels_w;
								int iy = i / num_kernels_w;
								for (int y = 0; y < father->kernels.dimension(2); ++y) {
									for (int x = 0; x < father->kernels.dimension(3); ++x) {
										cv::putText(kernels, std::to_string(father->kernels(j, k, y, x)),
											cv::Point2d((x + ix*kernel_w)*x_cell + 10, (y + iy*kernel_h)*y_cell + 10),
											CV_FONT_HERSHEY_PLAIN, 1.0f, ColorGradient::calc_color(father->kernels(j, k, y, x), min_V, max_V));
									}
								}
							}
						}
						cv::imshow(name, kernels);
					}
					cv::waitKey(1);
				}
				return gradient;
			}

			void update(int batch_size) {
				
			}

			void load_data(ByteBuffer& buffer) {

			}

			void save_data(ByteBuffer& buffer) {

			}
#pragma endregion
		};
	}
}
