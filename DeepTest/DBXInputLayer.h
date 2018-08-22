#pragma once

#pragma once

#include "DBXLayer.h"
#include "Dataset.h"

namespace DBX
{
	namespace NN
	{
		class InputLayer : public Layer
		{
#pragma region MEMBER VARIABLES
		protected:			
			std::shared_ptr<DatasetLoader> m_dataset_loader;
#pragma endregion

#pragma region CONSTRUCTORS	
		private:
			InputLayer() : Layer() {}

		public:

			InputLayer(std::shared_ptr<DatasetLoader> loader) : Layer() {
				m_dataset_loader = std::move(loader);
			}

			InputLayer(const InputLayer& l) = delete;

			InputLayer(InputLayer&& l) : Layer(std::move(l)) {
				this->m_dataset_loader = std::move(l.m_dataset_loader);
			}

			InputLayer& operator=(InputLayer& l) = delete;

			InputLayer& operator=(InputLayer&& l) {
				Layer::operator=(std::move(l));
				if (this != &l) {
					this->m_dataset_loader = std::move(l.m_dataset_loader);
				}
				return *this;
			}

			~InputLayer() {}
#pragma endregion

#pragma region MEMBER METHODS
			bool init(Params& params) {
				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_dataset_loader->get_size(1);
				m_output_tensor_dims[2] = m_dataset_loader->get_size(2);
				m_output_tensor_dims[3] = m_dataset_loader->get_size(3);
				return true;
			}

			bool reset(Params& params) {
				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_dataset_loader->get_size(1);
				m_output_tensor_dims[2] = m_dataset_loader->get_size(2);
				m_output_tensor_dims[3] = m_dataset_loader->get_size(3);
				return true;
			}

			Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& batch) {
				return batch;
			}

			Eigen::Tensor<float, 4> backprop(Eigen::Tensor<float, 4>& gradient) {
				return  gradient;
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