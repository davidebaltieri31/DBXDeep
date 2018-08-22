#pragma once

#include "DBXLayer.h"
#include <ppl.h>
#undef max
namespace DBX
{
	namespace NN
	{
		class ReLULayer : public Layer
		{
#pragma region MEMBER VARIABLES
		protected:
			//training
			Eigen::Tensor<float, 4> m_input_activation;
#pragma endregion

#pragma region CONSTRUCTORS	
		public:

			ReLULayer() : Layer() {
			}

			ReLULayer(const ReLULayer& l) = delete;

			ReLULayer(ReLULayer&& l) : Layer(std::move(l)) {
			}

			ReLULayer& operator=(ReLULayer& l) = delete;

			ReLULayer& operator=(ReLULayer&& l) {
				Layer::operator=(std::move(l));
				if (this != &l) {

				}
				return *this;
			}

			~ReLULayer() {}
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
				return true;
			}

			Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& batch) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);

				m_input_activation = batch;

				return compute(batch);
			}

			Eigen::Tensor<float, 4> backprop(Eigen::Tensor<float, 4>& gradient) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);

				std::array<Eigen::DenseIndex, 4> output_tensor_dims = { gradient.dimension(0), m_output_tensor_dims[1], m_output_tensor_dims[2], m_output_tensor_dims[3] };

				Eigen::Tensor<float, 4> prime = compute_prime(m_input_activation);

				Eigen::Tensor<float, 4> r = Eigen::Tensor<float, 4>(prime.dimensions());
				r.device(thread_pool_device) = gradient.reshape(output_tensor_dims) * prime;

				return r;
			}

			void update(int batch_size) {
			}

			void load_data(ByteBuffer& buffer) {

			}

			void save_data(ByteBuffer& buffer) {

			}

		protected:
	
			//compute the sigmoid activation for every element of the tensor
			Eigen::Tensor<float, 4> compute(Eigen::Tensor<float, 4>& data) {
				Eigen::Tensor<float, 4> r(data.dimension(0), data.dimension(1), data.dimension(2), data.dimension(3));
				int dim = data.dimension(0)*data.dimension(1)*data.dimension(2)*data.dimension(3);
				Concurrency::parallel_for(0, dim, [&](size_t i)
				{
					r.data()[i] = std::max(0.0f, data.data()[i]);
				});
				return r;
			}

			//compute the derivative of the sigmoid activation
			Eigen::Tensor<float, 4> compute_prime(Eigen::Tensor<float, 4>& data) {
				Eigen::Tensor<float, 4> r(data.dimension(0), data.dimension(1), data.dimension(2), data.dimension(3));
				int dim = data.dimension(0)*data.dimension(1)*data.dimension(2)*data.dimension(3);
				Concurrency::parallel_for(0, dim, [&](size_t i)
				{
					r.data()[i] = data.data()[i]>0.0f ? 1.0f : 0.0f;
				});
				return r;
			}
#pragma endregion
		};
	}
}
