#pragma once
#include <vector>
#include <math.h>
#include <unsupported\Eigen\CXX11\Tensor>
#include "DBXLayer.h"

namespace DBX
{
	namespace NN
	{
		class Cost {
		public:
			virtual Eigen::Tensor<float, 2> cost(const Eigen::Tensor<float, 4>& output_activations, const Eigen::Tensor<float, 4>& labels) const = 0;
			virtual Eigen::Tensor<float, 4> cost_derivative(const Eigen::Tensor<float, 4>& output_activations, const Eigen::Tensor<float, 4>& labels) const = 0;
		};

		class QuadraticCost : public Cost {
		public:
			Eigen::Tensor<float, 2> cost(const Eigen::Tensor<float, 4>& output_activations, const Eigen::Tensor<float, 4>& labels) const override {
				Eigen::Tensor<float, 4> t = output_activations - labels;
				Eigen::array<int, 2> dims = { 1, 3 };
				Eigen::Tensor<float, 2> r = (t*t).sum(dims);
				r = r * (0.5f);
				return r;
			};

			Eigen::Tensor<float, 4> cost_derivative(const Eigen::Tensor<float, 4>& output_activations, const Eigen::Tensor<float, 4>& labels) const override {
				Eigen::Tensor<float, 4> t = output_activations - labels;
				return t;
			}
		};

		class CrossEntropyCost : public Cost {
		public:
			Eigen::Tensor<float, 2> cost(const Eigen::Tensor<float, 4>& output_activations, const Eigen::Tensor<float, 4>& labels) const override {
				Eigen::Tensor<float, 4> one_minus_oputput = (output_activations*-1.0f) + 1.0f;
				Eigen::Tensor<float, 4> l1 = (output_activations + 1.0E-8f).log();
				Eigen::Tensor<float, 4> l2 = (one_minus_oputput + 1.0E-8f).log();
				Eigen::Tensor<float, 4> minus_y = labels * -1.0f;
				Eigen::Tensor<float, 4> one_minus_y = minus_y + 1.0f;
				Eigen::Tensor<float, 4> t1 = (minus_y * l1);
				Eigen::Tensor<float, 4> t2 = (one_minus_y * l2);
				Eigen::Tensor<float, 4> t3 = t1 - t2;
				Eigen::array<int, 2> dims = { 1, 3 }; /* dimension to reduce */
				Eigen::Tensor<float, 2> r = t3.sum(dims);
				return r;
			}

			Eigen::Tensor<float, 4> cost_derivative(const Eigen::Tensor<float, 4>& output_activations, const Eigen::Tensor<float, 4>& labels) const override {
				Eigen::Tensor<float, 4> r = output_activations - labels;
				return r;
			}
		};
	}
}