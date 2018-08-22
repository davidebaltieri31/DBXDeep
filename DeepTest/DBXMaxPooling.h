#pragma once

#include "DBXLayer.h"
#include <ppl.h>

namespace DBX
{
	namespace NN
	{
		class MaxPooling : public Layer
		{
#pragma region MEMBER VARIABLES
		protected:
			int m_kernel_width = 2;
			int m_kernel_height = 2;
			int32_t m_input_width;
			int32_t m_input_height;
			int32_t m_input_kernels;
			int32_t m_output_width;
			int32_t m_output_height;
			Eigen::Tensor<int, 5> m_pooling_indices;
#pragma endregion

#pragma region CONSTRUCTORS	
		public:

			MaxPooling(int kernel_width, int kernel_height) : Layer() {
				m_kernel_width = kernel_width;
				m_kernel_height = kernel_height;
			}

			MaxPooling(const MaxPooling& l) = delete;

			MaxPooling(MaxPooling&& l) : Layer(std::move(l)) {
			}

			MaxPooling& operator=(MaxPooling& l) = delete;

			MaxPooling& operator=(MaxPooling&& l) {
				Layer::operator=(std::move(l));
				if (this != &l) {

				}
				return *this;
			}

			~MaxPooling() {}
#pragma endregion

#pragma region MEMBER METHODS
			bool init(Params& params) {
				m_input_width = m_parent->output_tensor_dims[3];
				m_input_height = m_parent->output_tensor_dims[2];
				m_input_kernels = m_parent->output_tensor_dims[1];
				m_output_height = (m_input_height / m_kernel_height);
				m_output_width = (m_input_width / m_kernel_width);

				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_input_kernels;
				m_output_tensor_dims[2] = m_output_height;
				m_output_tensor_dims[3] = m_output_width;
				return true;
			}

			bool reset(Params& params) {
				m_input_width = m_parent->output_tensor_dims[3];
				m_input_height = m_parent->output_tensor_dims[2];
				m_input_kernels = m_parent->output_tensor_dims[1];
				m_output_height = (m_input_height / m_kernel_height);
				m_output_width = (m_input_width / m_kernel_width);

				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_input_kernels;
				m_output_tensor_dims[2] = m_output_height;
				m_output_tensor_dims[3] = m_output_width;
				return true;
			}

			Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& batch) {
				m_pooling_indices = Eigen::Tensor<int, 5>(batch.dimension(0), m_input_kernels, m_output_height, m_output_width, 2);
				Eigen::Tensor<float, 4> out = Eigen::Tensor<float, 4>(batch.dimension(0), m_input_kernels, m_output_height, m_output_width);
				int dim = batch.dimension(0);
				Concurrency::parallel_for(0, dim, [&](size_t j)
				{
					for (int k = 0; k < m_input_kernels; ++k) {
						for (int y = 0, ry = 0; y < m_input_height; y += m_kernel_height, ++ry) {
							for (int x = 0, rx = 0; x < m_input_width; x += m_kernel_width, ++rx) {
								int max_x = -1;
								int max_y = -1;
								float max_v = -FLT_MAX;
								for (int y1 = 0; y1 < m_kernel_height; ++y1) {
									for (int x1 = 0; x1 < m_kernel_width; ++x1) {
										int tx = x1 + x;
										int ty = y1 + y;
										if (batch(j, k, ty, tx) > max_v) {
											max_x = tx;
											max_y = ty;
											max_v = batch(j, k, ty, tx);
										}
									}
								}
								out(j, k, ry, rx) = max_v;
								m_pooling_indices(j, k, ry, rx, 0) = max_x;
								m_pooling_indices(j, k, ry, rx, 1) = max_y;
							}
						}
					}
				});
				return out;
			}

			Eigen::Tensor<float, 4> backprop(Eigen::Tensor<float, 4>& gradient) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);
				Eigen::Tensor<float, 4> out = Eigen::Tensor<float, 4>(gradient.dimension(0), m_input_kernels, m_input_height, m_input_width);
				std::array<Eigen::DenseIndex, 4> tensor_dims = { gradient.dimension(0), m_input_kernels, m_output_height, m_output_width };
				Eigen::Tensor<float, 4> delta = Eigen::Tensor<float, 4>(tensor_dims);
				{
					delta.device(thread_pool_device) = gradient.reshape(tensor_dims);
				}
				out.setZero();
				int dim = gradient.dimension(0);
				Concurrency::parallel_for(0, dim, [&](size_t j)
				{
					for (int k = 0; k < m_input_kernels; ++k) {
						for (int y = 0; y < m_output_height; ++y) {
							for (int x = 0; x < m_output_width; ++x) {
								int tx = m_pooling_indices(j, k, y, x, 0);
								int ty = m_pooling_indices(j, k, y, x, 1);
								out(j, k, ty, tx) = delta(j, k, y, x);
							}
						}
					}
				});
				return out;
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
