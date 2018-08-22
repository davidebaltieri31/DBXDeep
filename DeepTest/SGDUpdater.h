#pragma once
#define EIGEN_USE_THREADS
#include <Eigen\Eigen>
#include <unsupported\Eigen\CXX11\Tensor>
#include "Params.h"

namespace DBX
{
	enum class SGDUpdaterType {
		StandardSGDUpdater = 0,
		AdaGradSGDUpdater = 1,
		ModifiedAdaGradSGDUpdater = 2,
		AdaDeltaSGDUpdater = 3,
		AdamSGDUpdater = 4,
		RMSpropSGDUpdater = 5
	};

	class SGDUpdater {
	public:
		virtual void init(Params& params) = 0;

		virtual void update_sgd(Eigen::Tensor<float, 1>& bias,
								Eigen::Tensor<float, 2>& weight,
								Eigen::Tensor<float, 1>& nabla_b,
								Eigen::Tensor<float, 2>& nabla_w, 
								int batch_size) = 0;

		virtual void update_sgd(Eigen::Tensor<float, 3>& bias,
								Eigen::Tensor<float, 4>& weight,
								Eigen::Tensor<float, 3>& nabla_b,
								Eigen::Tensor<float, 4>& nabla_w,
								int batch_size) = 0;

		static SGDUpdater* create(SGDUpdaterType type);
	};

	class StandardSGDUpdater : public SGDUpdater {
	private:
		float learning_rate = 0.001f;
		float momentum = 0.5f;
		int num_threads = 8;
		int num_cores = 4;
		float lambda = 1.0f;
		float num_training_sample = 1.0f;

		Eigen::Tensor<float, 2> m_weight_velocity_2;
		Eigen::Tensor<float, 4> m_weight_velocity_4;
	public:
		void init(Params& params) {
			learning_rate = params.get_float_param("learning_rate");
			momentum = params.get_float_param("momentum");
			num_threads = params.get_int_param("num_threads");
			num_cores = params.get_int_param("num_cores");
			lambda = params.get_float_param("lambda");
			num_training_sample = params.get_float_param("num_training_sample");
			m_weight_velocity_2.setZero();
			m_weight_velocity_4.setZero();
		}

		void update_sgd(Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 2>& weight,
			Eigen::Tensor<float, 1>& nabla_b, Eigen::Tensor<float, 2>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_2.dimension(0) != weight.dimension(0) || m_weight_velocity_2.dimension(1) != weight.dimension(1)) {
				m_weight_velocity_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_weight_velocity_2.setZero();
			}

			m_weight_velocity_2.device(thread_pool_device) = momentum * m_weight_velocity_2 - ((learning_rate / batch_size)*nabla_w);
			weight.device(thread_pool_device) = ( 1.0f - (learning_rate*lambda) / num_training_sample ) * weight + m_weight_velocity_2;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}

		void update_sgd(Eigen::Tensor<float, 3>& bias, Eigen::Tensor<float, 4>& weight,
			Eigen::Tensor<float, 3>& nabla_b, Eigen::Tensor<float, 4>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_4.dimension(0) != weight.dimension(0) || m_weight_velocity_4.dimension(1) != weight.dimension(1) ||
				m_weight_velocity_4.dimension(2) != weight.dimension(2) || m_weight_velocity_4.dimension(3) != weight.dimension(3)) {
				m_weight_velocity_4 = Eigen::Tensor<float, 4>(weight.dimension(0), weight.dimension(1), weight.dimension(2), weight.dimension(3));
				m_weight_velocity_4.setZero();
			}

			m_weight_velocity_4.device(thread_pool_device) = momentum * m_weight_velocity_4 - ((learning_rate / batch_size)*nabla_w);
			weight.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_4;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}
	};

	class AdaGradSGDUpdater : public SGDUpdater {
	private:
		float learning_rate = 0.001f;
		float momentum = 0.5f;
		int num_threads = 8;
		int num_cores = 4;
		float lambda = 1.0f;
		float num_training_sample = 1;

		Eigen::Tensor<float, 2> m_weight_velocity_2;
		Eigen::Tensor<float, 4> m_weight_velocity_4;

		Eigen::Tensor<float, 2> m_ada_delta_2;
		Eigen::Tensor<float, 4> m_ada_delta_4;
	public:
		void init(Params& params) {
			learning_rate = params.get_float_param("learning_rate");
			momentum = params.get_float_param("momentum");
			num_threads = params.get_int_param("num_threads");
			num_cores = params.get_int_param("num_cores");
			lambda = params.get_float_param("lambda");
			num_training_sample = params.get_float_param("num_training_sample");
			if (learning_rate == 0.0f) learning_rate = 0.01f;
			if (num_training_sample == 0.0f) num_training_sample = 1.0f;
			m_ada_delta_2.setZero();
			m_ada_delta_4.setZero();
			m_weight_velocity_2.setZero();
			m_weight_velocity_4.setZero();
		}

		void update_sgd(Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 2>& weight,
			Eigen::Tensor<float, 1>& nabla_b, Eigen::Tensor<float, 2>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_2.dimension(0) != weight.dimension(0) || m_weight_velocity_2.dimension(1) != weight.dimension(1)) {
				m_weight_velocity_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_weight_velocity_2.setZero();
			}

			if (m_ada_delta_2.dimension(0) != weight.dimension(0) || m_ada_delta_2.dimension(1) != weight.dimension(1)) {
				m_ada_delta_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_delta_2.setZero();
			}

			//m_ada_temp_sqrt2.device(thread_pool_device) = (m_ada_delta_2 + 1.0E-6f).sqrt();
			//m_ada_temp_inv2.device(thread_pool_device) = m_ada_temp_sqrt2.inverse()*(learning_rate/batch_size);
			//m_weight_velocity_2.device(thread_pool_device) = momentum * m_weight_velocity_2 - m_ada_temp_inv2*nabla_w;
			
			m_ada_delta_2.device(thread_pool_device) = m_ada_delta_2 + (nabla_w*nabla_w) / (float(batch_size*batch_size));

			m_weight_velocity_2.device(thread_pool_device) = momentum * m_weight_velocity_2 - (((m_ada_delta_2 + 1.0E-6f).sqrt().inverse()*(learning_rate/batch_size))*nabla_w);
			weight.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_2;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}

		void update_sgd(Eigen::Tensor<float, 3>& bias, Eigen::Tensor<float, 4>& weight,
			Eigen::Tensor<float, 3>& nabla_b, Eigen::Tensor<float, 4>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_4.dimension(0) != weight.dimension(0) || m_weight_velocity_4.dimension(1) != weight.dimension(1) ||
				m_weight_velocity_4.dimension(2) != weight.dimension(2) || m_weight_velocity_4.dimension(3) != weight.dimension(3)) {
				m_weight_velocity_4 = Eigen::Tensor<float, 4>(weight.dimension(0), weight.dimension(1), weight.dimension(2), weight.dimension(3));
				m_weight_velocity_4.setZero();
			}

			if (m_ada_delta_4.dimension(0) != weight.dimension(0) || m_ada_delta_4.dimension(1) != weight.dimension(1) ||
				m_ada_delta_4.dimension(2) != weight.dimension(2) || m_ada_delta_4.dimension(3) != weight.dimension(3)) {
				m_ada_delta_4 = Eigen::Tensor<float, 4>(weight.dimension(0), weight.dimension(1), weight.dimension(2), weight.dimension(3));
				m_ada_delta_4.setZero();
			}
			
			m_ada_delta_4.device(thread_pool_device) = m_ada_delta_4 + (nabla_w*nabla_w) / (float(batch_size*batch_size));

			m_weight_velocity_4.device(thread_pool_device) = momentum * m_weight_velocity_4 - (((m_ada_delta_4 + 1.0E-6f).sqrt().inverse()*(learning_rate / batch_size))*nabla_w);
			weight.device(thread_pool_device) = (1.0f - learning_rate*(lambda / num_training_sample)) * weight + m_weight_velocity_4;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}
	};

	class ModifiedAdaGradSGDUpdater : public SGDUpdater {
	private:
		float learning_rate = 0.001f;
		float momentum = 0.5f;
		int num_threads = 8;
		int num_cores = 4;
		float lambda = 1.0f;
		float num_training_sample = 1;

		Eigen::Tensor<float, 2> m_weight_velocity_2;
		Eigen::Tensor<float, 4> m_weight_velocity_4;

		float m_num_steps = 0.0f;
		Eigen::Tensor<float, 2> m_ada_delta_2;
		Eigen::Tensor<float, 4> m_ada_delta_4;
	public:
		void init(Params& params) {
			learning_rate = params.get_float_param("learning_rate");
			momentum = params.get_float_param("momentum");
			num_threads = params.get_int_param("num_threads");
			num_cores = params.get_int_param("num_cores");
			lambda = params.get_float_param("lambda");
			num_training_sample = params.get_float_param("num_training_sample");
			if (learning_rate == 0.0f) learning_rate = 0.01f;
			if (num_training_sample == 0.0f) num_training_sample = 1.0f;
			m_num_steps = 0.0f;
			m_ada_delta_2.setZero();
			m_ada_delta_4.setZero();
			m_weight_velocity_2.setZero();
			m_weight_velocity_4.setZero();
		}

		void update_sgd(Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 2>& weight,
			Eigen::Tensor<float, 1>& nabla_b, Eigen::Tensor<float, 2>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_2.dimension(0) != weight.dimension(0) || m_weight_velocity_2.dimension(1) != weight.dimension(1)) {
				m_weight_velocity_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_weight_velocity_2.setZero();
			}

			if (m_ada_delta_2.dimension(0) != weight.dimension(0) || m_ada_delta_2.dimension(1) != weight.dimension(1)) {
				m_ada_delta_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_delta_2.setZero();
			}

			m_ada_delta_2.device(thread_pool_device) = m_ada_delta_2 * (m_num_steps / (m_num_steps + 1.0f)) + (nabla_w*nabla_w) / (batch_size*(m_num_steps + 1.0f));
			m_num_steps += 1.0f;

			//m_ada_temp_sqrt2.device(thread_pool_device) = (m_ada_delta_2 + 1.0E-6f).sqrt();
			//m_ada_temp_inv2.device(thread_pool_device) = m_ada_temp_sqrt2.inverse()*(learning_rate/batch_size);
			//m_weight_velocity_2.device(thread_pool_device) = momentum * m_weight_velocity_2 - m_ada_temp_inv2*nabla_w;
			
			m_weight_velocity_2.device(thread_pool_device) = momentum * m_weight_velocity_2 - (((m_ada_delta_2 + 1.0E-6f).sqrt().inverse()*(learning_rate / batch_size))*nabla_w);
			weight.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_2;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}

		void update_sgd(Eigen::Tensor<float, 3>& bias, Eigen::Tensor<float, 4>& weight,
			Eigen::Tensor<float, 3>& nabla_b, Eigen::Tensor<float, 4>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_4.dimension(0) != weight.dimension(0) || m_weight_velocity_4.dimension(1) != weight.dimension(1) ||
				m_weight_velocity_4.dimension(2) != weight.dimension(2) || m_weight_velocity_4.dimension(3) != weight.dimension(3)) {
				m_weight_velocity_4 = Eigen::Tensor<float, 4>(weight.dimension(0), weight.dimension(1), weight.dimension(2), weight.dimension(3));
				m_weight_velocity_4.setZero();
			}

			if (m_ada_delta_4.dimension(0) != weight.dimension(0) || m_ada_delta_4.dimension(1) != weight.dimension(1) ||
				m_ada_delta_4.dimension(2) != weight.dimension(2) || m_ada_delta_4.dimension(3) != weight.dimension(3)) {
				m_ada_delta_4 = Eigen::Tensor<float, 4>(weight.dimension(0), weight.dimension(1), weight.dimension(2), weight.dimension(3));
				m_ada_delta_4.setZero();
			}

			m_ada_delta_4.device(thread_pool_device) = m_ada_delta_4 * (m_num_steps / (m_num_steps + 1.0f)) + (nabla_w*nabla_w) / (batch_size*(m_num_steps + 1.0f));
			m_num_steps += 1.0f;

			m_weight_velocity_4.device(thread_pool_device) = momentum * m_weight_velocity_4 - (((m_ada_delta_4 + 1.0E-6f).sqrt().inverse()*(learning_rate / batch_size))*nabla_w);
			weight.device(thread_pool_device) = (1.0f - learning_rate*(lambda / num_training_sample)) * weight + m_weight_velocity_4;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}
	};

	class AdaDeltaSGDUpdater : public SGDUpdater {
	private:
		float learning_rate = 0.001f;
		float momentum = 0.5f;
		int num_threads = 8;
		int num_cores = 4;
		float lambda = 1.0f;
		float num_training_sample = 1;

		Eigen::Tensor<float, 2> m_weight_velocity_2;
		Eigen::Tensor<float, 4> m_weight_velocity_4;

		float m_ada_delta_decay = 0.9f;
		Eigen::Tensor<float, 2> m_ada_delta_2;
		Eigen::Tensor<float, 4> m_ada_delta_4;

		Eigen::Tensor<float, 2> m_ada_weight_2;
		Eigen::Tensor<float, 4> m_ada_weight_4;

		Eigen::Tensor<float, 2> m_ada_delta_bottom_2;
		Eigen::Tensor<float, 4> m_ada_delta_bottom_4;

		Eigen::Tensor<float, 2> m_ada_weight_top_2;
		Eigen::Tensor<float, 4> m_ada_weight_top_4;

		Eigen::Tensor<float, 2> m_ada_temp_weight_2;
		Eigen::Tensor<float, 4> m_ada_temp_weight_4;

		Eigen::Tensor<float, 2> m_ada_delta_weight_2;
		Eigen::Tensor<float, 4> m_ada_delta_weight_4;
	public:
		void init(Params& params) {
			learning_rate = params.get_float_param("learning_rate");
			momentum = params.get_float_param("momentum");
			num_threads = params.get_int_param("num_threads");
			num_cores = params.get_int_param("num_cores");
			lambda = params.get_float_param("lambda");
			num_training_sample = params.get_float_param("num_training_sample");
			m_ada_delta_decay = params.get_float_param("sgd_delta_decay");
			if (learning_rate == 0.0f) learning_rate = 0.01f;
			if (num_training_sample == 0.0f) num_training_sample = 1.0f;

			m_ada_delta_2.setZero();
			m_ada_delta_4.setZero();

			m_ada_weight_2.setZero();
			m_ada_weight_4.setZero();
			
			m_ada_delta_bottom_2.setZero();
			m_ada_delta_bottom_4.setZero();

			m_ada_weight_top_2.setZero();
			m_ada_weight_top_4.setZero();

			m_weight_velocity_2.setZero();
			m_weight_velocity_4.setZero();

			m_ada_temp_weight_2.setZero();
			m_ada_temp_weight_4.setZero();

			m_ada_delta_weight_2.setZero();
			m_ada_delta_weight_4.setZero();
		}

		void update_sgd(Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 2>& weight,
			Eigen::Tensor<float, 1>& nabla_b, Eigen::Tensor<float, 2>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_2.dimension(0) != weight.dimension(0) || m_weight_velocity_2.dimension(1) != weight.dimension(1)) {
				m_weight_velocity_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_weight_velocity_2.setZero();
			}

			if (m_ada_delta_2.dimension(0) != weight.dimension(0) || m_ada_delta_2.dimension(1) != weight.dimension(1)) {
				m_ada_delta_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_delta_2.setZero();
				m_ada_delta_bottom_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_delta_bottom_2.setZero();
				m_ada_weight_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_weight_2.setZero();
				m_ada_weight_top_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_weight_top_2.setZero();
				m_ada_temp_weight_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_temp_weight_2.setZero();
				m_ada_delta_weight_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_delta_weight_2.setZero();
			}

			m_ada_delta_2.device(thread_pool_device) = m_ada_delta_2 * m_ada_delta_decay + (nabla_w*nabla_w) * ((1.0f - m_ada_delta_decay) / (batch_size*batch_size));

			m_ada_delta_bottom_2.device(thread_pool_device) = (m_ada_delta_2 + 1.0E-6f).sqrt().inverse();
			m_ada_weight_top_2.device(thread_pool_device) = (m_ada_weight_2 + 1.0E-6f).sqrt();

			m_weight_velocity_2.device(thread_pool_device) = momentum * m_weight_velocity_2 - m_ada_weight_top_2*m_ada_delta_bottom_2*(1.0f / batch_size)*nabla_w;
			
			m_ada_temp_weight_2.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_2;
			
			m_ada_delta_weight_2.device(thread_pool_device) = m_ada_temp_weight_2 - weight;
			
			weight.device(thread_pool_device) = m_ada_temp_weight_2;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);

			m_ada_weight_2.device(thread_pool_device) = m_ada_weight_2 * m_ada_delta_decay + (m_ada_delta_weight_2*m_ada_delta_weight_2) * (1.0f - m_ada_delta_decay) ;

		}

		void update_sgd(Eigen::Tensor<float, 3>& bias, Eigen::Tensor<float, 4>& weight,
			Eigen::Tensor<float, 3>& nabla_b, Eigen::Tensor<float, 4>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_4.dimension(0) != weight.dimension(0) || m_weight_velocity_4.dimension(1) != weight.dimension(1) ||
				m_weight_velocity_4.dimension(2) != weight.dimension(2) || m_weight_velocity_4.dimension(3) != weight.dimension(3)) {
				m_weight_velocity_4 = Eigen::Tensor<float, 4>(weight.dimension(0), weight.dimension(1), weight.dimension(2), weight.dimension(3));
				m_weight_velocity_4.setZero();
			}

			if (m_ada_delta_4.dimension(0) != weight.dimension(0) || m_ada_delta_4.dimension(1) != weight.dimension(1) ||
				m_ada_delta_4.dimension(2) != weight.dimension(2) || m_ada_delta_4.dimension(3) != weight.dimension(3)) {
				m_ada_delta_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_delta_4.setZero();
				m_ada_delta_bottom_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_delta_bottom_4.setZero();
				m_ada_weight_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_weight_4.setZero();
				m_ada_weight_top_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_weight_top_4.setZero();
				m_ada_temp_weight_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_temp_weight_4.setZero();
				m_ada_delta_weight_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_delta_weight_4.setZero();
			}
			
			m_ada_delta_4.device(thread_pool_device) = m_ada_delta_4 * m_ada_delta_decay + (nabla_w*nabla_w) * ((1.0f - m_ada_delta_decay) / (batch_size*batch_size));

			m_ada_delta_bottom_4.device(thread_pool_device) = (m_ada_delta_4 + 1.0E-6f).sqrt().inverse();
			m_ada_weight_top_4.device(thread_pool_device) = (m_ada_weight_4 + 1.0E-6f).sqrt();

			m_weight_velocity_4.device(thread_pool_device) = momentum * m_weight_velocity_4 - m_ada_weight_top_4*m_ada_delta_bottom_4*(1.0f / batch_size)*nabla_w;

			m_ada_temp_weight_4.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_4;

			m_ada_delta_weight_4.device(thread_pool_device) = m_ada_temp_weight_4 - weight;

			weight.device(thread_pool_device) = m_ada_temp_weight_4;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);

			m_ada_weight_4.device(thread_pool_device) = m_ada_weight_4 * m_ada_delta_decay + (m_ada_delta_weight_4*m_ada_delta_weight_4) * (1.0f - m_ada_delta_decay);

		}
	};

	class RMSPropSGDUpdater : public SGDUpdater {
	private:
		float learning_rate = 0.001f;
		float momentum = 0.5f;
		int num_threads = 8;
		int num_cores = 4;
		float lambda = 1.0f;
		float num_training_sample = 1;

		Eigen::Tensor<float, 2> m_weight_velocity_2;
		Eigen::Tensor<float, 4> m_weight_velocity_4;

		float m_ada_delta_decay = 0.9f;
		Eigen::Tensor<float, 2> m_ada_delta_2;
		Eigen::Tensor<float, 4> m_ada_delta_4;

		Eigen::Tensor<float, 2> m_ada_delta_bottom_2;
		Eigen::Tensor<float, 4> m_ada_delta_bottom_4;
	public:
		void init(Params& params) {
			learning_rate = params.get_float_param("learning_rate");
			momentum = params.get_float_param("momentum");
			num_threads = params.get_int_param("num_threads");
			num_cores = params.get_int_param("num_cores");
			lambda = params.get_float_param("lambda");
			num_training_sample = params.get_float_param("num_training_sample");
			m_ada_delta_decay = params.get_float_param("sgd_delta_decay");
			if (learning_rate == 0.0f) learning_rate = 0.01f;
			if (num_training_sample == 0.0f) num_training_sample = 1.0f;

			m_ada_delta_2.setZero();
			m_ada_delta_4.setZero();

			m_ada_delta_bottom_2.setZero();
			m_ada_delta_bottom_4.setZero();

			m_weight_velocity_2.setZero();
			m_weight_velocity_4.setZero();
		}

		void update_sgd(Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 2>& weight,
			Eigen::Tensor<float, 1>& nabla_b, Eigen::Tensor<float, 2>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_2.dimension(0) != weight.dimension(0) || m_weight_velocity_2.dimension(1) != weight.dimension(1)) {
				m_weight_velocity_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_weight_velocity_2.setZero();
			}

			if (m_ada_delta_2.dimension(0) != weight.dimension(0) || m_ada_delta_2.dimension(1) != weight.dimension(1)) {
				m_ada_delta_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_delta_2.setZero();
				m_ada_delta_bottom_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_delta_bottom_2.setZero();
			}

			m_ada_delta_2.device(thread_pool_device) = m_ada_delta_2 * m_ada_delta_decay + (nabla_w*nabla_w) * ((1.0f - m_ada_delta_decay) / (batch_size*batch_size));

			m_ada_delta_bottom_2.device(thread_pool_device) = (m_ada_delta_2 + 1.0E-6f).sqrt().inverse();

			m_weight_velocity_2.device(thread_pool_device) = momentum * m_weight_velocity_2 - m_ada_delta_bottom_2*(learning_rate / batch_size)*nabla_w;

			weight.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_2;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}

		void update_sgd(Eigen::Tensor<float, 3>& bias, Eigen::Tensor<float, 4>& weight,
			Eigen::Tensor<float, 3>& nabla_b, Eigen::Tensor<float, 4>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_4.dimension(0) != weight.dimension(0) || m_weight_velocity_4.dimension(1) != weight.dimension(1) ||
				m_weight_velocity_4.dimension(2) != weight.dimension(2) || m_weight_velocity_4.dimension(3) != weight.dimension(3)) {
				m_weight_velocity_4 = Eigen::Tensor<float, 4>(weight.dimension(0), weight.dimension(1), weight.dimension(2), weight.dimension(3));
				m_weight_velocity_4.setZero();
			}

			if (m_ada_delta_4.dimension(0) != weight.dimension(0) || m_ada_delta_4.dimension(1) != weight.dimension(1) ||
				m_ada_delta_4.dimension(2) != weight.dimension(2) || m_ada_delta_4.dimension(3) != weight.dimension(3)) {
				m_ada_delta_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_delta_4.setZero();
				m_ada_delta_bottom_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_delta_bottom_4.setZero();
			}

			m_ada_delta_4.device(thread_pool_device) = m_ada_delta_4 * m_ada_delta_decay + (nabla_w*nabla_w) * ((1.0f - m_ada_delta_decay) / (batch_size*batch_size));

			m_ada_delta_bottom_4.device(thread_pool_device) = (m_ada_delta_4 + 1.0E-6f).sqrt().inverse();

			m_weight_velocity_4.device(thread_pool_device) = momentum * m_weight_velocity_4 - m_ada_delta_bottom_4*(learning_rate / batch_size)*nabla_w;

			weight.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_4;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}
	};

	class AdamSGDUpdater : public SGDUpdater {
	private:
		float learning_rate = 0.001f;
		float momentum = 0.5f;
		int num_threads = 8;
		int num_cores = 4;
		float lambda = 1.0f;
		float num_training_sample = 1;

		Eigen::Tensor<float, 2> m_weight_velocity_2;
		Eigen::Tensor<float, 4> m_weight_velocity_4;

		float m_ada_m_decay = 0.9f;
		float m_ada_v_decay = 0.999f;
		int m_num_steps = 0;

		Eigen::Tensor<float, 2> m_ada_m_2;
		Eigen::Tensor<float, 4> m_ada_m_4;

		Eigen::Tensor<float, 2> m_ada_v_2;
		Eigen::Tensor<float, 4> m_ada_v_4;

		Eigen::Tensor<float, 2> m_ada_m_bias_corrected_2;
		Eigen::Tensor<float, 4> m_ada_m_bias_corrected_4;

		Eigen::Tensor<float, 2> m_ada_v_bias_corrected_2;
		Eigen::Tensor<float, 4> m_ada_v_bias_corrected_4;
	public:
		void init(Params& params) {
			learning_rate = params.get_float_param("learning_rate");
			momentum = params.get_float_param("momentum");
			num_threads = params.get_int_param("num_threads");
			num_cores = params.get_int_param("num_cores");
			lambda = params.get_float_param("lambda");
			num_training_sample = params.get_float_param("num_training_sample");
			m_ada_m_decay = params.get_float_param("sgd_delta_decay");
			m_ada_v_decay = params.get_float_param("sgd_delta_decay_2");
			if (learning_rate == 0.0f) learning_rate = 0.01f;
			if (num_training_sample == 0.0f) num_training_sample = 1.0f;

			m_ada_m_2.setZero();
			m_ada_m_4.setZero();

			m_ada_m_bias_corrected_2.setZero();
			m_ada_m_bias_corrected_4.setZero();

			m_ada_v_2.setZero();
			m_ada_v_4.setZero();

			m_ada_v_bias_corrected_2.setZero();
			m_ada_v_bias_corrected_4.setZero();

			m_weight_velocity_2.setZero();
			m_weight_velocity_4.setZero();

			m_num_steps = 0;
		}

		void update_sgd(Eigen::Tensor<float, 1>& bias, Eigen::Tensor<float, 2>& weight,
			Eigen::Tensor<float, 1>& nabla_b, Eigen::Tensor<float, 2>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_2.dimension(0) != weight.dimension(0) || m_weight_velocity_2.dimension(1) != weight.dimension(1)) {
				m_weight_velocity_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_weight_velocity_2.setZero();
			}

			if (m_ada_m_2.dimension(0) != weight.dimension(0) || m_ada_m_2.dimension(1) != weight.dimension(1)) {
				m_ada_m_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_m_2.setZero();
				m_ada_v_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_v_2.setZero();

				m_ada_m_bias_corrected_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_m_bias_corrected_2.setZero();

				m_ada_v_bias_corrected_2 = Eigen::Tensor<float, 2>(weight.dimension(0), weight.dimension(1));
				m_ada_v_bias_corrected_2.setZero();
			}
			m_ada_m_2.device(thread_pool_device) = m_ada_m_2 * m_ada_m_decay + (nabla_w * ((1.0f - m_ada_m_decay) / batch_size));

			m_ada_v_2.device(thread_pool_device) = m_ada_v_2 * m_ada_v_decay + (nabla_w*nabla_w) * ((1.0f - m_ada_v_decay) / (batch_size*batch_size));

			m_num_steps++;

			m_ada_m_bias_corrected_2.device(thread_pool_device) = m_ada_m_2 * (1.0f / (1.0f - std::pow(m_ada_m_decay, m_num_steps)));

			m_ada_v_bias_corrected_2.device(thread_pool_device) = m_ada_v_2 * (1.0f / (1.0f - std::pow(m_ada_v_decay, m_num_steps)));

			m_weight_velocity_2.device(thread_pool_device) = momentum * m_weight_velocity_2 - (m_ada_v_bias_corrected_2.sqrt() + +1.0E-8f).inverse()*(learning_rate / batch_size)*m_ada_m_bias_corrected_2;

			weight.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_2;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}

		void update_sgd(Eigen::Tensor<float, 3>& bias, Eigen::Tensor<float, 4>& weight,
			Eigen::Tensor<float, 3>& nabla_b, Eigen::Tensor<float, 4>& nabla_w, int batch_size) {

			Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(num_threads);
			Eigen::ThreadPoolDevice thread_pool_device(&tp, num_cores);

			if (m_weight_velocity_4.dimension(0) != weight.dimension(0) || m_weight_velocity_4.dimension(1) != weight.dimension(1) ||
				m_weight_velocity_4.dimension(2) != weight.dimension(2) || m_weight_velocity_4.dimension(3) != weight.dimension(3)) {
				m_weight_velocity_4 = Eigen::Tensor<float, 4>(weight.dimension(0), weight.dimension(1), weight.dimension(2), weight.dimension(3));
				m_weight_velocity_4.setZero();
			}

			if (m_ada_m_4.dimension(0) != weight.dimension(0) || m_ada_m_4.dimension(1) != weight.dimension(1) ||
				m_ada_m_4.dimension(2) != weight.dimension(2) || m_ada_m_4.dimension(3) != weight.dimension(3)) {
				m_ada_m_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_m_4.setZero();
				m_ada_v_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_v_4.setZero();

				m_ada_m_bias_corrected_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_m_bias_corrected_4.setZero();

				m_ada_v_bias_corrected_4 = Eigen::Tensor<float, 4>(weight.dimensions());
				m_ada_v_bias_corrected_4.setZero();

			}

			m_ada_m_4.device(thread_pool_device) = m_ada_m_4 * m_ada_m_decay + (nabla_w * ((1.0f - m_ada_m_decay) / batch_size));

			m_ada_v_4.device(thread_pool_device) = m_ada_v_4 * m_ada_v_decay + (nabla_w*nabla_w) * ((1.0f - m_ada_v_decay) / (batch_size*batch_size));

			m_num_steps++;

			m_ada_m_bias_corrected_4.device(thread_pool_device) = m_ada_m_4 * (1.0f / (1.0f - std::pow(m_ada_m_decay, m_num_steps)));

			m_ada_v_bias_corrected_4.device(thread_pool_device) = m_ada_v_4 * (1.0f / (1.0f - std::pow(m_ada_v_decay, m_num_steps)));

			m_weight_velocity_4.device(thread_pool_device) = momentum * m_weight_velocity_4 - (m_ada_v_bias_corrected_4.sqrt() + +1.0E-8f).inverse()*(learning_rate / batch_size)*m_ada_m_bias_corrected_4;

			weight.device(thread_pool_device) = (1.0f - (learning_rate*lambda) / num_training_sample) * weight + m_weight_velocity_4;
			bias.device(thread_pool_device) -= ((learning_rate / batch_size)*nabla_b);
		}
	};

	SGDUpdater* SGDUpdater::create(SGDUpdaterType type) {
		switch (type) {
		case SGDUpdaterType::AdaGradSGDUpdater:
			return new AdaGradSGDUpdater();
			break;
		case SGDUpdaterType::ModifiedAdaGradSGDUpdater:
			return new ModifiedAdaGradSGDUpdater();
			break;
		case SGDUpdaterType::AdaDeltaSGDUpdater:
			return new AdaDeltaSGDUpdater();
			break;
		case SGDUpdaterType::AdamSGDUpdater:
			return new AdamSGDUpdater();
			break;
		case SGDUpdaterType::RMSpropSGDUpdater:
			return new RMSPropSGDUpdater();
			break;
		case SGDUpdaterType::StandardSGDUpdater:
		default:
			return new StandardSGDUpdater();
			break;
		}
	}
}