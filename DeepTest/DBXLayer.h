#pragma once

#define EIGEN_USE_THREADS
#include <vector>
#include <math.h>
#include <Eigen\Eigen>
#include <unsupported\Eigen\CXX11\Tensor>
#include "Params.h"
#include "WeightInitializer.h"
#include "DBXUtils.h"
#include "ByteBuffer.h"
#include "SGDUpdater.h"

namespace DBX
{
	namespace NN
	{
		class Layer
		{
#pragma region MEMBER VARIABLES
		protected:
			std::shared_ptr<Layer> m_parent = nullptr;
			std::shared_ptr<Layer> m_child = nullptr;

			std::array<int32_t, 4> m_output_tensor_dims;

			std::unique_ptr<WeightInitializer> m_initializer;
			std::unique_ptr<SGDUpdater> m_sgd_updater;

			bool m_inference_mode;
		public:
			const std::shared_ptr<Layer>& parent;
			const std::shared_ptr<Layer>& child;

			const std::array<int32_t, 4>& output_tensor_dims;

#pragma endregion

#pragma region CONSTRUCTORS	
		protected:
			Layer() : parent(m_parent), child(m_child), output_tensor_dims(m_output_tensor_dims) {
			}

		public:

			Layer(WeightInitializerType init_type, SGDUpdaterType sgd_type) : Layer() {
				m_initializer = std::unique_ptr<WeightInitializer>(WeightInitializer::create(init_type));
				m_sgd_updater = std::unique_ptr<SGDUpdater>(SGDUpdater::create(sgd_type));
			}

			Layer(const Layer& l) = delete;

			Layer(Layer&& l) : Layer() {
				this->m_parent = std::move(l.m_parent);
				this->m_child = std::move(l.m_child);
				this->m_output_tensor_dims = std::move(l.m_output_tensor_dims);
				this->m_initializer = std::move(l.m_initializer);
				this->m_sgd_updater = std::move(l.m_sgd_updater);
			}

			Layer& operator=(Layer& l) = delete;

			Layer& operator=(Layer&& l) {
				if (this != &l) {
					this->m_parent = std::move(l.m_parent);
					this->m_child = std::move(l.m_child);
					this->m_output_tensor_dims = std::move(l.m_output_tensor_dims);
					this->m_initializer = std::move(l.m_initializer);
					this->m_sgd_updater = std::move(l.m_sgd_updater);
				}
				return *this;
			}

			~Layer() { }
#pragma endregion

#pragma region MEMBER METHODS
			void link(std::shared_ptr<Layer> parent_, std::shared_ptr<Layer> child_) {
				m_parent = parent_;
				m_child = child_;
			}

			void set_initializer(WeightInitializerType type) {
				m_initializer = std::unique_ptr<WeightInitializer>(WeightInitializer::create(type));
			}

			void set_sgd_updater(SGDUpdaterType type) {
				m_sgd_updater = std::unique_ptr<SGDUpdater>(SGDUpdater::create(type));
			}

			void set_inference_mode(bool inference) {
				m_inference_mode = inference;
			}

			virtual bool init(Params& params) = 0;

			virtual bool reset(Params& params) = 0;

			virtual Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& batch) = 0;

			virtual Eigen::Tensor<float, 4> backprop(Eigen::Tensor<float, 4>& delta_x_w) = 0;

			virtual void update(int batch_size) = 0;

			virtual void load_data(ByteBuffer& buffer) = 0;

			virtual void save_data(ByteBuffer& buffer) = 0;
#pragma endregion
		};
	}
}