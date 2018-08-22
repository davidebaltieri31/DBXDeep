#pragma once

#include "DBXLayer.h"
#include <ppl.h>

namespace DBX
{
	namespace NN
	{
		class ConvolutionLayer : public Layer
		{
#pragma region MEMBER VARIABLES
		protected:
			typedef Eigen::Tensor<float, 2>::DimensionPair DimPair;

			int32_t m_input_width;
			int32_t m_input_height;
			int32_t m_input_kernels;

			int32_t m_output_width;
			int32_t m_output_height;

			int32_t m_num_kernels;
			int32_t m_kernel_width;
			int32_t m_kernel_height;

			Eigen::Tensor<float, 3> m_bias;
			Eigen::Tensor<float, 4> m_kernels;

			//training
			Eigen::Tensor<float, 4> m_z;
			Eigen::Tensor<float, 4> m_input_activation;
			Eigen::Tensor<float, 3> m_nabla_b;
			Eigen::Tensor<float, 4> m_nabla_w;

		public:
			Eigen::Tensor<float, 3>& bias;
			Eigen::Tensor<float, 4>& kernels;
			Eigen::Tensor<float, 3>& nabla_b;
			Eigen::Tensor<float, 4>& nabla_w;
#pragma endregion

#pragma region CONSTRUCTORS	
		private:
			ConvolutionLayer() : Layer(), bias(m_bias), kernels(m_kernels), nabla_b(m_nabla_b), nabla_w(m_nabla_w) {}

		public:

			ConvolutionLayer(WeightInitializerType init_type, SGDUpdaterType sgd_type, int num_kernels, int kernel_width, int kernel_height) : 
				Layer(init_type, sgd_type), bias(m_bias), kernels(m_kernels), nabla_b(m_nabla_b), nabla_w(m_nabla_w) {
				this->m_num_kernels = num_kernels;
				this->m_kernel_width = kernel_width;
				this->m_kernel_height = kernel_height;
			}

			ConvolutionLayer(const ConvolutionLayer& l) = delete;

			ConvolutionLayer(ConvolutionLayer&& l) : Layer(std::move(l)), bias(m_bias), kernels(m_kernels), nabla_b(m_nabla_b), nabla_w(m_nabla_w) {
				this->m_input_width = l.m_input_width;
				this->m_input_height = l.m_input_height;
				this->m_input_kernels = l.m_input_kernels;

				this->m_output_width = l.m_output_width;
				this->m_output_height = l.m_output_height;

				this->m_num_kernels = l.m_num_kernels;
				this->m_kernel_width = l.m_kernel_width;
				this->m_kernel_height = l.m_kernel_height;

				std::swap(this->m_bias, l.m_bias);
				std::swap(this->m_kernels, l.m_kernels);

				std::swap(this->m_z, l.m_z);
				std::swap(this->m_input_activation, l.m_input_activation);
			}

			ConvolutionLayer& operator=(ConvolutionLayer& l) = delete;

			ConvolutionLayer& operator=(ConvolutionLayer&& l) {
				Layer::operator=(std::move(l));
				if (this != &l) {
					this->m_input_width = l.m_input_width;
					this->m_input_height = l.m_input_height;
					this->m_input_kernels = l.m_input_kernels;

					this->m_output_width = l.m_output_width;
					this->m_output_height = l.m_output_height;

					this->m_num_kernels = l.m_num_kernels;
					this->m_kernel_width = l.m_kernel_width;
					this->m_kernel_height = l.m_kernel_height;

					std::swap(this->m_bias, l.m_bias);
					std::swap(this->m_kernels, l.m_kernels);

					std::swap(this->m_z, l.m_z);
					std::swap(this->m_input_activation, l.m_input_activation);
				}
				return *this;
			}

			~ConvolutionLayer() {}
#pragma endregion

#pragma region MEMBER METHODS
			bool init(Params& params) {
				//has no parent, error, return false
				if (m_parent == nullptr || m_initializer == nullptr)
					return false;

				m_input_width = m_parent->output_tensor_dims[3];
				m_input_height = m_parent->output_tensor_dims[2];
				m_input_kernels = m_parent->output_tensor_dims[1];
				m_output_width = m_input_width - m_kernel_width + 1;
				m_output_height = m_input_height - m_kernel_height + 1;

				m_initializer->init(params, m_input_width*m_input_height*m_input_kernels, m_output_width*m_output_height*m_num_kernels);
				m_sgd_updater->init(params);

				//init layer bias, a simple vector of size equal to the number of outputs of this layer
				int bias_x = m_output_width;
				int bias_y = m_output_height;
				m_bias = Eigen::Tensor<float, 3>(m_num_kernels, bias_y, bias_x);
				for (int i = 0; i < m_num_kernels; ++i) {
					for (int y = 0; y < bias_y; ++y) {
						for (int x = 0; x < bias_x; ++x) {
							m_bias(i, y, x) = m_initializer->generate();
						}
					}
				}

				//init weights, parent can be a convolutional layer so input size is  the size of the parent tensor 
				//(minus the first dimension which is the batch size)
				m_kernels = Eigen::Tensor<float, 4>(m_num_kernels, m_input_kernels, m_kernel_height, m_kernel_width);
				for (int i = 0; i < m_num_kernels; ++i) {
					for (int j = 0; j < m_input_kernels; ++j) {
						for (int y = 0; y < m_kernel_height; ++y) {
							for (int x = 0; x < m_kernel_width; ++x) {
								m_kernels(i, j, y, x) = m_initializer->generate();
							}
						}
					}
				}

				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_num_kernels;
				m_output_tensor_dims[2] = m_output_height;
				m_output_tensor_dims[3] = m_output_width;


				return true;
			}

			bool reset(Params& params) {
				//has no parent, error, return false
				if (m_parent == nullptr || m_initializer == nullptr)
					return false;

				m_input_width = m_parent->output_tensor_dims[3];
				m_input_height = m_parent->output_tensor_dims[2];
				m_input_kernels = m_parent->output_tensor_dims[1];
				m_output_width = m_input_width - m_kernel_width + 1;
				m_output_height = m_input_height - m_kernel_height + 1;

				m_initializer->init(params, m_input_width*m_input_height*m_input_kernels, m_output_width*m_output_height*m_num_kernels);
				m_sgd_updater->init(params);

				//init layer bias, a simple vector of size equal to the number of outputs of this layer
				if (m_bias.dimension(0) != m_num_kernels || m_bias.dimension(1) != m_output_height || m_bias.dimension(2) != m_output_width) {
					int bias_x = m_output_width;
					int bias_y = m_output_height;
					m_bias = Eigen::Tensor<float, 3>(m_num_kernels, bias_y, bias_x);
					for (int i = 0; i < m_num_kernels; ++i) {
						for (int y = 0; y < bias_y; ++y) {
							for (int x = 0; x < bias_x; ++x) {
								m_bias(i, y, x) = m_initializer->generate();
							}
						}
					}
				}

				//init weights, parent can be a convolutional layer so input size is  the size of the parent tensor 
				//(minus the first dimension which is the batch size)
				if (m_kernels.dimension(0) != m_num_kernels || m_kernels.dimension(1) != m_input_kernels || 
					m_kernels.dimension(2) != m_kernel_height || m_kernels.dimension(3) != m_kernel_width) {
					m_kernels = Eigen::Tensor<float, 4>(m_num_kernels, m_input_kernels, m_kernel_height, m_kernel_width);
					for (int i = 0; i < m_num_kernels; ++i) {
						for (int j = 0; j < m_input_kernels; ++j) {
							for (int y = 0; y < m_kernel_height; ++y) {
								for (int x = 0; x < m_kernel_width; ++x) {
									m_kernels(i, j, y, x) = m_initializer->generate();
								}
							}
						}
					}
				}

				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_num_kernels;
				m_output_tensor_dims[2] = m_output_height;
				m_output_tensor_dims[3] = m_output_width;

				return true;
			}

			Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& batch) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);

				m_input_activation = Eigen::Tensor<float, 4>(batch.dimensions());
				m_input_activation.device(thread_pool_device) = batch;

				//compute convolution
				{
					Eigen::array<ptrdiff_t, 3> conv_res_convolution_dims = { 0,1,2 };
					std::array<Eigen::DenseIndex, 2> conv_res_reshaped_dims = { m_output_height, m_output_width };
					m_z = Eigen::Tensor<float, 4>(batch.dimension(0), m_num_kernels, m_output_height, m_output_width);
					//for (int i = 0; i < batch.dimension(0); ++i) {
					int dim = batch.dimension(0);
					Concurrency::parallel_for(0, dim, [&](size_t i)
					{
						Eigen::Tensor<float, 3> element_input_activation = Eigen::Tensor<float, 3>(m_input_activation.dimension(1), m_input_activation.dimension(2), m_input_activation.dimension(3));
						Eigen::Tensor<float, 3> selected_kernel = Eigen::Tensor<float, 3>(m_kernels.dimension(1), m_kernels.dimension(2), m_kernels.dimension(3));
						Eigen::Tensor<float, 3> conv_res = Eigen::Tensor<float, 3>(1, m_output_height, m_output_width);
						Eigen::Tensor<float, 2> conv_res_reshaped = Eigen::Tensor<float, 2>(m_output_height, m_output_width);

						element_input_activation = m_input_activation.chip(i, 0);
						for (int j = 0; j < m_num_kernels; ++j) {
							selected_kernel = m_kernels.chip(j, 0);
							conv_res = element_input_activation.convolve(selected_kernel, conv_res_convolution_dims);
							conv_res_reshaped = conv_res.reshape(conv_res_reshaped_dims);
							m_z.chip(i, 0).chip(j, 0) = conv_res_reshaped;
						}
					});
				}
				//add bias
				for (int j = 0; j < m_z.dimension(0); ++j) m_z.chip(j, 0).device(thread_pool_device) += m_bias;

				return m_z;
			}

			Eigen::Tensor<float, 4> backprop(Eigen::Tensor<float, 4>& gradient) {

				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);

				m_nabla_b = Eigen::Tensor<float, 3>(m_bias.dimension(0), m_bias.dimension(1), m_bias.dimension(2)); m_nabla_b.setZero();
				m_nabla_w = Eigen::Tensor<float, 4>(m_kernels.dimension(0), m_kernels.dimension(1), m_kernels.dimension(2), m_kernels.dimension(3)); m_nabla_w.setZero();

				std::array<Eigen::DenseIndex, 4> tensor_dims = { gradient.dimension(0), m_num_kernels, m_output_height, m_output_width };
				Eigen::Tensor<float, 4> delta = Eigen::Tensor<float, 4>(tensor_dims);
				delta.device(thread_pool_device) = gradient.reshape(tensor_dims);

				Eigen::Tensor<float, 4> ret = Eigen::Tensor<float, 4>(gradient.dimension(0), m_input_kernels, m_input_height, m_input_width); ret.setZero();
				{
					int pad_val_x = (m_kernel_width + int(std::ceil(double(m_kernel_width / 2.0)))) / 2;
					int pad_val_y = (m_kernel_height + int(std::ceil(double(m_kernel_height / 2.0)))) / 2;
					Eigen::array<std::pair<int, int>, 4> paddings = { std::make_pair(0, 0) ,std::make_pair(0, 0) ,std::make_pair(pad_val_y, pad_val_y) ,std::make_pair(pad_val_x, pad_val_x) };
					Eigen::Tensor<float, 4> delta_expanded = delta.pad(paddings);
					Eigen::array<bool, 3> reverse = { false, true, true };
					Eigen::array<ptrdiff_t, 3> dims = { 0,1,2 };
					std::array<Eigen::DenseIndex, 2> conv_res_reshaped_dims = { m_input_height, m_input_width };

					int dim = delta.dimension(0);
					//for (int i = 0; i < delta.dimension(0); ++i) {
					Concurrency::parallel_for(0, dim, [&](size_t i)
					{
						Eigen::Tensor<float, 3> element_delta_expanded = Eigen::Tensor<float, 3>(delta_expanded.dimension(1), delta_expanded.dimension(2), delta_expanded.dimension(3));
						Eigen::Tensor<float, 3> selected_kernel = Eigen::Tensor<float, 3>(m_kernels.dimension(0), m_kernels.dimension(2), m_kernels.dimension(3));
						Eigen::Tensor<float, 3> selected_kernel_rotated = Eigen::Tensor<float, 3>(m_kernels.dimension(0), m_kernels.dimension(2), m_kernels.dimension(3));
						Eigen::Tensor<float, 3> conv_res = Eigen::Tensor<float, 3>(1, m_input_height, m_input_width);
						Eigen::Tensor<float, 2> conv_res_reshaped = Eigen::Tensor<float, 2>(m_input_height, m_input_width);

						element_delta_expanded = delta_expanded.chip(i, 0);
						for (int j = 0; j < m_input_kernels; ++j) {
							selected_kernel = m_kernels.chip(j, 1);
							selected_kernel_rotated = selected_kernel.reverse(reverse);
							conv_res = element_delta_expanded.convolve(selected_kernel_rotated, dims);
							conv_res_reshaped = conv_res.reshape(conv_res_reshaped_dims);
							ret.chip(i, 0).chip(j, 0) += conv_res_reshaped;
						}
					});
				}

				for (int j = 0; j < delta.dimension(0); ++j) m_nabla_b.device(thread_pool_device) += delta.chip(j, 0);

				{
					Eigen::array<ptrdiff_t, 2> dims2 = { 1,2 };

					//for (int j = 0; j < m_num_kernels; ++j) {
					Concurrency::parallel_for(0, m_num_kernels, [&](size_t j)
					{
						Eigen::Tensor<float, 3> element_input_activation = Eigen::Tensor<float, 3>(m_input_activation.dimension(1), m_input_activation.dimension(2), m_input_activation.dimension(3));
						Eigen::Tensor<float, 3> element_delta = Eigen::Tensor<float, 3>(delta.dimension(1), delta.dimension(2), delta.dimension(3));
						Eigen::Tensor<float, 2> element_delta_for_kernel = Eigen::Tensor<float, 2>(delta.dimension(2), delta.dimension(3));
						Eigen::Tensor<float, 3> conv_res2 = Eigen::Tensor<float, 3>(m_nabla_w.dimension(1), m_nabla_w.dimension(2), m_nabla_w.dimension(3));
						for (int i = 0; i < m_input_activation.dimension(0); ++i) {
							element_input_activation = m_input_activation.chip(i, 0);
							element_delta = delta.chip(i, 0);
							element_delta_for_kernel = element_delta.chip(j, 0);
							conv_res2 = element_input_activation.convolve(element_delta_for_kernel, dims2);
							m_nabla_w.chip(j, 0) += conv_res2;
						}
					});
				}

				return ret;
			}

			void update(int batch_size) {
				m_sgd_updater->update_sgd(m_bias, m_kernels, m_nabla_b, m_nabla_w, batch_size);
			}

			void load_data(ByteBuffer& buffer) {

			}

			void save_data(ByteBuffer& buffer) {

			}
#pragma endregion
		};
	}
}
