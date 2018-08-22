#pragma once

#include "DBXLayer.h"

namespace DBX
{
	namespace NN
	{
		class FullyConnectedLayer : public Layer
		{
#pragma region MEMBER VARIABLES
		protected:
			typedef Eigen::Tensor<float, 2>::DimensionPair DimPair;

			int32_t m_input_size;

			int32_t m_output_size;

			Eigen::Tensor<float, 1> m_bias;
			Eigen::Tensor<float, 2> m_weight;

			//training
			Eigen::Tensor<float, 2> m_z;
			Eigen::Tensor<float, 2> m_input_activation;
			Eigen::Tensor<float, 1> m_nabla_b;
			Eigen::Tensor<float, 2> m_nabla_w;

			//temporary training vars
			Eigen::Tensor<float, 2> delta;
			Eigen::Tensor<float, 2> r_temp;
			Eigen::Tensor<float, 4> r;
		public:
			Eigen::Tensor<float, 1>& bias;
			Eigen::Tensor<float, 2>& weight;
			Eigen::Tensor<float, 1>& nabla_b;
			Eigen::Tensor<float, 2>& nabla_w;
#pragma endregion

#pragma region CONSTRUCTORS	
		private:
			FullyConnectedLayer() : Layer(), bias(m_bias), weight(m_weight), nabla_b(m_nabla_b), nabla_w(m_nabla_w) {}
		
		public:
			
			FullyConnectedLayer(WeightInitializerType init_type, SGDUpdaterType sgd_type, int size) : Layer(init_type, sgd_type), bias(m_bias), weight(m_weight), nabla_b(m_nabla_b), nabla_w(m_nabla_w) {
				this->m_output_size = size;
			}

			FullyConnectedLayer(const FullyConnectedLayer& l) = delete;

			FullyConnectedLayer(FullyConnectedLayer&& l) : Layer(std::move(l)), bias(m_bias), weight(m_weight), nabla_b(m_nabla_b), nabla_w(m_nabla_w) {
				this->m_output_size = l.m_output_size;
				this->m_input_size = l.m_input_size;

				this->m_bias = std::move(l.m_bias);
				this->m_weight = std::move(l.m_weight);

				this->m_z = std::move(l.m_z);
				this->m_input_activation = std::move(l.m_input_activation);
				this->m_nabla_b = std::move(l.m_nabla_b);
				this->m_nabla_w = std::move(l.m_nabla_w);
			}

			FullyConnectedLayer& operator=(FullyConnectedLayer& l) = delete;

			FullyConnectedLayer& operator=(FullyConnectedLayer&& l) {
				Layer::operator=(std::move(l));
				if (this != &l) {
					this->m_output_size = l.m_output_size;
					this->m_input_size = l.m_input_size;

					this->m_bias = std::move(l.m_bias);
					this->m_weight = std::move(l.m_weight);

					this->m_z = std::move(l.m_z);
					this->m_input_activation = std::move(l.m_input_activation);
					this->m_nabla_b = std::move(l.m_nabla_b);
					this->m_nabla_w = std::move(l.m_nabla_w);
				}
				return *this;
			}

			~FullyConnectedLayer() {}
#pragma endregion

#pragma region MEMBER METHODS
			bool init(Params& params) {
				//has no parent, error, return false
				if (m_parent == nullptr || m_initializer == nullptr) 
					return false;

				m_input_size = m_parent->output_tensor_dims[1] * m_parent->output_tensor_dims[2] * m_parent->output_tensor_dims[3];

				m_initializer->init(params, m_input_size, m_output_size);
				m_sgd_updater->init(params);
				
				//init layer bias, a simple vector of size equal to the number of outputs of this layer
				m_bias = Eigen::Tensor<float, 1>(m_output_size);
				m_nabla_b = Eigen::Tensor<float, 1>(m_output_size);
				for (int x = 0; x < m_output_size; ++x) {
					m_bias(x) = m_initializer->generate();
				}

				//init weights, parent can be a convolutional layer so input size is  the size of the parent tensor 
				//(minus the first dimension which is the batch size)
				
				m_weight = Eigen::Tensor<float, 2>(m_input_size, m_output_size);
				m_nabla_w = Eigen::Tensor<float, 2>(m_input_size, m_output_size);
				for (int y = 0; y < m_input_size; ++y) {
					for (int x = 0; x < m_output_size; ++x) {
						m_weight(y, x) = m_initializer->generate();
					}
				}

				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = 1;
				m_output_tensor_dims[2] = 1;
				m_output_tensor_dims[3] = m_output_size;

				
				return true;
			}

			bool reset(Params& params) {
				//has no parent, error, return false
				if (m_parent == nullptr || m_initializer == nullptr)
					return false;

				m_input_size = m_parent->output_tensor_dims[1] * m_parent->output_tensor_dims[2] * m_parent->output_tensor_dims[3];

				m_initializer->init(params, m_input_size, m_output_size);
				m_sgd_updater->init(params);

				//init layer bias, a simple vector of size equal to the number of outputs of this layer
				if (m_bias.dimension(0) != m_output_size) {
					m_bias = Eigen::Tensor<float, 1>(m_output_size);
					for (int x = 0; x < m_output_size; ++x) {
						m_bias(x) = 0.0f;
					}
				}

				//init weights, parent can be a convolutional layer so input size is  the size of the parent tensor 
				//(minus the first dimension which is the batch size)
				if (m_weight.dimension(0) != m_input_size || m_weight.dimension(1) != m_output_size) {
					m_weight = Eigen::Tensor<float, 2>(m_input_size, m_output_size);
					for (int y = 0; y < m_input_size; ++y) {
						for (int x = 0; x < m_output_size; ++x) {
							m_weight(y, x) = m_initializer->generate();
						}
					}
				}

				m_output_tensor_dims[0] = 1;
				m_output_tensor_dims[1] = 1;
				m_output_tensor_dims[2] = 1;
				m_output_tensor_dims[3] = m_output_size;

				

				return true;
			}

			Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& batch) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);

				std::array<Eigen::DenseIndex, 2> input_matrix_dims = { batch.dimension(0), m_input_size };
				std::array<Eigen::DenseIndex, 4> output_tensor_dims = { batch.dimension(0), 1, 1, m_output_size };

				//as this is a fully connected layer rearrange the batch dimensions combining image and kernels
				if(m_input_activation.dimension(0) != batch.dimension(0) || m_input_activation.dimension(1) != m_input_size)
					m_input_activation = Eigen::Tensor<float, 2>(input_matrix_dims);
				m_input_activation.device(thread_pool_device) = batch.reshape(input_matrix_dims);

				//multiply imputs by weights
				Eigen::array<DimPair, 1> product_dims;
				product_dims[0] = DimPair(1, 0);
				if(m_z.dimension(0) != batch.dimension(0) || m_z.dimension(1) != m_output_size)
					m_z = Eigen::Tensor<float, 2>(batch.dimension(0), m_output_size);
				m_z.device(thread_pool_device) = m_input_activation.contract(m_weight, product_dims);

				//add bias
				for (int j = 0; j < m_z.dimension(0); ++j) m_z.chip(j, 0).device(thread_pool_device) += m_bias;

				return m_z.reshape(output_tensor_dims);
			}

			Eigen::Tensor<float, 4> backprop(Eigen::Tensor<float, 4>& gradient) {

				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);

				std::array<Eigen::DenseIndex, 2> input_matrix_dims = { gradient.dimension(0), m_output_size };
				std::array<Eigen::DenseIndex, 4> output_tensor_dims = { gradient.dimension(0), 1, 1, m_input_size };

				//temp delta weight and delta bias
				m_nabla_b.setZero();
				m_nabla_w.setZero();

				//compute this layer energy gradient
				if(delta.dimension(0) != gradient.dimension(0) || delta.dimension(1) != m_output_size)
					delta = Eigen::Tensor<float, 2>(input_matrix_dims);
				delta.device(thread_pool_device) = gradient.reshape(input_matrix_dims);

				//compute the energy gradient to be sent to parent layer
				if(r_temp.dimension(0)!=gradient.dimension(0) || r_temp.dimension(1)!= m_input_size)
					r_temp = Eigen::Tensor<float, 2>(gradient.dimension(0), m_input_size);
				if(r.dimension(0)!=gradient.dimension(0) || r.dimension(1)!=1 || r.dimension(2)!=1 || r.dimension(3)!= m_input_size)
					r = Eigen::Tensor<float, 4>(output_tensor_dims);
				Eigen::array<DimPair, 1> out_product_dims;
				out_product_dims[0] = DimPair(1, 1);
				r_temp.device(thread_pool_device) = delta.contract(m_weight, out_product_dims);
				r.device(thread_pool_device) = r_temp.reshape(output_tensor_dims);

				//compute delta bias
				for (int j = 0; j < delta.dimension(0); ++j) m_nabla_b.device(thread_pool_device) += delta.chip(j, 0);

				//compute delta weights
				Eigen::array<DimPair, 1> weight_product_dims;
				weight_product_dims[0] = DimPair(0, 0);
				m_nabla_w.device(thread_pool_device) = m_input_activation.contract(delta, weight_product_dims);

				return r;
			}

			void update(int batch_size) {
				m_sgd_updater->update_sgd(m_bias, m_weight, m_nabla_b, m_nabla_w, batch_size);
			}
			
			void load_data(ByteBuffer& buffer) {

			}

			void save_data(ByteBuffer& buffer) {

			}
#pragma endregion
		};
	}
}