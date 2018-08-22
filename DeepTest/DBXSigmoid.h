#pragma once

#include "DBXLayer.h"

namespace DBX
{
	namespace NN
	{
		class SigmoidLayer : public Layer
		{
#pragma region MEMBER VARIABLES
		protected:
			//training
			Eigen::Tensor<float, 4> m_input_activation;
#pragma endregion

#pragma region CONSTRUCTORS	
		public:

			SigmoidLayer() : Layer() {
			}

			SigmoidLayer(const SigmoidLayer& l) = delete;

			SigmoidLayer(SigmoidLayer&& l) : Layer(std::move(l)) {
			}

			SigmoidLayer& operator=(SigmoidLayer& l) = delete;

			SigmoidLayer& operator=(SigmoidLayer&& l) {
				Layer::operator=(std::move(l));
				if (this != &l) {

				}
				return *this;
			}

			~SigmoidLayer() {}
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
			Eigen::Tensor<float, 4> compute(Eigen::Tensor<float, 4>& data) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);
				Eigen::Tensor<float, 4> r = Eigen::Tensor<float, 4>(data.dimensions());
				r.device(thread_pool_device) = ((data*-1.0f).exp() + 1.0f).inverse();
				return r;
			}

			//compute the derivative of the sigmoid activation
			Eigen::Tensor<float, 4> compute_prime(Eigen::Tensor<float, 4>& data) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);
				Eigen::Tensor<float, 4> sig_data = compute(data);
				Eigen::Tensor<float, 4> r = Eigen::Tensor<float, 4>(data.dimensions());
				//r.device(thread_pool_device) = sig_data*((sig_data*-1.0f) + 1.0f);
				r.device(thread_pool_device) = sig_data*(1.0f-sig_data);
				return r;
			}
#pragma endregion
		};
	}
}
