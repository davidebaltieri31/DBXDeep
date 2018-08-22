#pragma once
#include <cmath>
#include <random>
#include "Params.h"

namespace DBX
{
	enum class WeightInitializerType {
		HeNormal = 1,
		HeUniform = 2,
		GlorotNormal = 3,
		GlorotUniform = 4,
		XavierUniform = 5,
		Uniform = 6,
		Gaussian = 7,
		BasicGaussian = 8,
		Zero = 0
	};

	class WeightInitializer {
	protected:
		float m_weights_costant_scale = 1.0f;
		virtual void init_params(Params& params) {
			m_weights_costant_scale = params.get_float_param("weights_costant_scale");
		}
	public:
		virtual void init(Params& params, int fanIn, int fanOut) = 0;
		virtual float generate() = 0;
		static WeightInitializer* create(WeightInitializerType type);
	};

	class HeNormal : public WeightInitializer {
	protected:
		std::random_device m_rd;
		std::mt19937 m_gen;
		std::normal_distribution<float> m_generate;
	public:
		void init(Params& params, int fanIn, int fanOut) {
			init_params(params);
			m_gen = std::mt19937(m_rd());
			m_generate = std::normal_distribution<float>(0.0f, std::sqrt(2.0f / float(fanIn)));
		}
		float generate() {
			return m_generate(m_gen) * m_weights_costant_scale;
		}
	};

	class HeUniform : public WeightInitializer {
	protected:
		std::random_device m_rd;
		std::mt19937 m_gen;
		std::uniform_real_distribution<float> m_generate;
	public:
		void init(Params& params, int fanIn, int fanOut) {
			init_params(params);
			m_gen = std::mt19937(m_rd());
			float v = std::sqrt(6.0f / float(fanIn));
			m_generate = std::uniform_real_distribution<float>(-v, v);
		}
		float generate() {
			return m_generate(m_gen) * m_weights_costant_scale;
		}
	};

	class GlorotNormal : public WeightInitializer {
	protected:
		std::random_device m_rd;
		std::mt19937 m_gen;
		std::normal_distribution<float> m_generate;
	public:
		void init(Params& params, int fanIn, int fanOut) {
			init_params(params);
			m_gen = std::mt19937(m_rd());
			m_generate = std::normal_distribution<float>(0.0f, std::sqrt(2.0f / (float(fanIn)+float(fanOut)) ) );
		}
		float generate() {
			return m_generate(m_gen) * m_weights_costant_scale;
		}
	};

	class GlorotUniform : public WeightInitializer {
	protected:
		std::random_device m_rd;
		std::mt19937 m_gen;
		std::uniform_real_distribution<float> m_generate;
	public:
		void init(Params& params, int fanIn, int fanOut) {
			init_params(params);
			m_gen = std::mt19937(m_rd());
			float v = std::sqrt(6.0f / (float(fanIn) + float(fanOut)));
			m_generate = std::uniform_real_distribution<float>(-v, v);
		}
		float generate() {
			return m_generate(m_gen) * m_weights_costant_scale;
		}
	};

	class XavierUniform : public WeightInitializer {
	protected:
		std::random_device m_rd;
		std::mt19937 m_gen;
		std::uniform_real_distribution<float> m_generate;
	public:
		void init(Params& params, int fanIn, int fanOut) {
			init_params(params);
			m_gen = std::mt19937(m_rd());
			float v = std::sqrt(3.0f / float(fanIn));
			m_generate = std::uniform_real_distribution<float>(-v, v);
		}
		float generate() {
			return m_generate(m_gen) * m_weights_costant_scale;
		}
	};

	class Uniform : public WeightInitializer {
	protected:
		std::random_device m_rd;
		std::mt19937 m_gen;
		std::uniform_real_distribution<float> m_generate;
	public:
		void init(Params& params, int fanIn, int fanOut) {
			init_params(params);
			m_gen = std::mt19937(m_rd());
			float v = 1.0f / 20.0f;
			m_generate = std::uniform_real_distribution<float>(-v, v);
		}
		float generate() {
			return m_generate(m_gen) * m_weights_costant_scale;
		}
	};

	class Gaussian : public WeightInitializer {
	protected:
		std::random_device m_rd;
		std::mt19937 m_gen;
		std::normal_distribution<float> m_generate;
	public:
		void init(Params& params, int fanIn, int fanOut) {
			init_params(params);
			m_gen = std::mt19937(m_rd());
			m_generate = std::normal_distribution<float>(0.0f, std::sqrt(0.04f / float(fanIn)));
		}
		float generate() {
			return m_generate(m_gen) * m_weights_costant_scale;
		}
	};

	class BasicGaussian : public WeightInitializer {
	protected:
		std::random_device m_rd;
		std::mt19937 m_gen;
		std::normal_distribution<float> m_generate;
	public:
		void init(Params& params, int fanIn, int fanOut) {
			init_params(params);
			m_gen = std::mt19937(m_rd());
			m_generate = std::normal_distribution<float>(0.0f, 1.0f);
		}
		float generate() {
			return m_generate(m_gen) * m_weights_costant_scale;
		}
	};

	class Zero : public WeightInitializer {
	public:
		void init(Params& params, int fanIn, int fanOut) {}
		float generate() {
			return 0.0f;
		}
	};

	WeightInitializer* WeightInitializer::create(WeightInitializerType type) {
		switch (type) {
		case WeightInitializerType::Gaussian:
			return new Gaussian();
			break;
		case WeightInitializerType::GlorotNormal:
			return new GlorotNormal();
			break;
		case WeightInitializerType::GlorotUniform:
			return new GlorotUniform();
			break;
		case WeightInitializerType::HeNormal:
			return new HeNormal();
			break;
		case WeightInitializerType::HeUniform:
			return new HeUniform();
			break;
		case WeightInitializerType::Uniform:
			return new Uniform();
			break;
		case WeightInitializerType::XavierUniform:
			return new XavierUniform();
			break;
		case WeightInitializerType::BasicGaussian:
			return new BasicGaussian();
			break;
		case WeightInitializerType::Zero:
		default:
			return new Zero();
			break;
		}
	}
}