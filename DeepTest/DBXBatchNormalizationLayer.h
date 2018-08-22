#pragma once

#include "DBXLayer.h"
#include <ppl.h>
#undef max
namespace DBX
{
	namespace NN
	{
		class BatchNormLayer : public Layer
		{
#pragma region MEMBER VARIABLES
		protected:
			Eigen::Tensor<float, 4> gamma;
			Eigen::Tensor<float, 4> beta;
			Eigen::Tensor<float, 4> pop_mean;
			Eigen::Tensor<float, 4> pop_var;

			int m_step = 0;
			//training
			Eigen::Tensor<float, 4> m_input_activation;
			Eigen::Tensor<float, 4> mu;
			Eigen::Tensor<float, 4> xmu;
			Eigen::Tensor<float, 4> sq;
			Eigen::Tensor<float, 4> var;
			Eigen::Tensor<float, 4> sqrtvar;
			Eigen::Tensor<float, 4> ivar;
			Eigen::Tensor<float, 4> xhat;
			Eigen::Tensor<float, 4> gammax;
			Eigen::Tensor<float, 4> out;

			Eigen::Tensor<float, 4> dbeta;
			Eigen::Tensor<float, 4> dgammax;
			Eigen::Tensor<float, 4> dgamma;
			Eigen::Tensor<float, 4> dxhat;
			Eigen::Tensor<float, 4> divar;
			Eigen::Tensor<float, 4> dxmu1;
			Eigen::Tensor<float, 4> dsqrtvar;
			Eigen::Tensor<float, 4> dvar;
			Eigen::Tensor<float, 4> dsq;
			Eigen::Tensor<float, 4> dxmu2;
			Eigen::Tensor<float, 4> dx1;
			Eigen::Tensor<float, 4> dx2;
			Eigen::Tensor<float, 4> dx;
			Eigen::Tensor<float, 4> dmu;
#pragma endregion

#pragma region CONSTRUCTORS	
		public:

			BatchNormLayer(SGDUpdaterType sgd_type) : Layer(WeightInitializerType::Zero, sgd_type) {
			}

			BatchNormLayer(const BatchNormLayer& l) = delete;

			BatchNormLayer(BatchNormLayer&& l) : Layer(std::move(l)) {
			}

			BatchNormLayer& operator=(BatchNormLayer& l) = delete;

			BatchNormLayer& operator=(BatchNormLayer&& l) {
				Layer::operator=(std::move(l));
				if (this != &l) {

				}
				return *this;
			}

			~BatchNormLayer() {}
#pragma endregion

#pragma region MEMBER METHODS
			bool init(Params& params) {
				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_parent->output_tensor_dims[1];
				m_output_tensor_dims[2] = m_parent->output_tensor_dims[2];
				m_output_tensor_dims[3] = m_parent->output_tensor_dims[3];

				gamma.setZero();
				beta.setZero();
				pop_mean.setZero();
				pop_var.setZero();

				m_step = 0;

				m_sgd_updater->init(params);
				return true;
			}

			bool reset(Params& params) {
				m_output_tensor_dims[0] = -1;
				m_output_tensor_dims[1] = m_parent->output_tensor_dims[1];
				m_output_tensor_dims[2] = m_parent->output_tensor_dims[2];
				m_output_tensor_dims[3] = m_parent->output_tensor_dims[3];

				//gamma.setZero();
				//beta.setZero();
				//pop_mean.setZero();
				//pop_var.setZero();

				m_step = 0;

				m_sgd_updater->init(params);
				return true;
			}

			Eigen::Tensor<float, 4> forward(Eigen::Tensor<float, 4>& batch) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);

				if (gamma.dimension(0) != 1 || gamma.dimension(1) != batch.dimension(1) ||
					gamma.dimension(2) != batch.dimension(2) || gamma.dimension(3) != batch.dimension(3)) {
					gamma = Eigen::Tensor<float, 4>(1, batch.dimension(1), batch.dimension(2), batch.dimension(3));
					gamma.setConstant(1.0f);

					beta = Eigen::Tensor<float, 4>(1, batch.dimension(1), batch.dimension(2), batch.dimension(3));
					beta.setZero();

					pop_mean = Eigen::Tensor<float, 4>(1, batch.dimension(1), batch.dimension(2), batch.dimension(3));
					pop_mean.setZero();

					pop_var = Eigen::Tensor<float, 4>(1, batch.dimension(1), batch.dimension(2), batch.dimension(3));
					pop_var.setZero();
				}

				int batch_size = batch.dimension(0); //N

				if (!m_inference_mode) {
					if (m_step == 0) {
						pop_mean.setZero();
						pop_var.setZero();
					}

					std::array<Eigen::DenseIndex, 4> tensor_dims = { 1, batch.dimension(1), batch.dimension(2), batch.dimension(3) };

					//step1: calculate mean
					//mu = 1. / N * np.sum(x, axis = 0)
					if (mu.dimension(0) != 1 || mu.dimension(1) != batch.dimension(1) || mu.dimension(2) != batch.dimension(2) || mu.dimension(3) != batch.dimension(3))
						mu = Eigen::Tensor<float, 4>(tensor_dims);
					Eigen::array<int, 1> dims = { 0 };
					mu.device(thread_pool_device) = (1.0f / batch_size) * (batch.sum(dims).reshape(tensor_dims));

					//#step2 : subtract mean vector of every trainings example
					//xmu = x - mu
					if (xmu.dimension(0) != batch.dimension(0) || xmu.dimension(1) != batch.dimension(1) || xmu.dimension(2) != batch.dimension(2) || xmu.dimension(3) != batch.dimension(3))
						xmu = Eigen::Tensor<float, 4>(batch.dimensions());
					Concurrency::parallel_for(0, int(batch.dimension(0)), [&](size_t i)
					{
						xmu.chip(i, 0) = batch.chip(i, 0) - mu.chip(0, 0);
					});

					//#step3 : calculation denominator
					//sq = xmu ** 2
					if (sq.dimension(0) != batch.dimension(0) || sq.dimension(1) != batch.dimension(1) || sq.dimension(2) != batch.dimension(2) || sq.dimension(3) != batch.dimension(3))
						sq = Eigen::Tensor<float, 4>(batch.dimensions());
					sq.device(thread_pool_device) = xmu * xmu;

					//#step4 : calculate variance
					//var = 1. / N * np.sum(sq, axis = 0)
					if (var.dimension(0) != 1 || var.dimension(1) != batch.dimension(1) || var.dimension(2) != batch.dimension(2) || var.dimension(3) != batch.dimension(3))
						var = Eigen::Tensor<float, 4>(tensor_dims);
					var.device(thread_pool_device) = (1.0f / batch_size) * sq.sum(dims).reshape(tensor_dims);

					//#step5 : add eps for numerical stability, then sqrt
					//sqrtvar = np.sqrt(var + eps)
					if (sqrtvar.dimension(0) != 1 || sqrtvar.dimension(1) != batch.dimension(1) || sqrtvar.dimension(2) != batch.dimension(2) || sqrtvar.dimension(3) != batch.dimension(3))
						sqrtvar = Eigen::Tensor<float, 4>(tensor_dims);
					sqrtvar.device(thread_pool_device) = (var + 1.0E-8f).sqrt();

					//#step6 : invert sqrtwar
					//ivar = 1. / sqrtvar
					if (ivar.dimension(0) != 1 || ivar.dimension(1) != batch.dimension(1) || ivar.dimension(2) != batch.dimension(2) || ivar.dimension(3) != batch.dimension(3))
						ivar = Eigen::Tensor<float, 4>(tensor_dims);
					ivar.device(thread_pool_device) = sqrtvar.inverse();

					//#step7 : execute normalization
					if (xhat.dimension(0) != batch.dimension(0) || xhat.dimension(1) != batch.dimension(1) || xhat.dimension(2) != batch.dimension(2) || xhat.dimension(3) != batch.dimension(3))
						xhat = Eigen::Tensor<float, 4>(batch.dimensions());
					Concurrency::parallel_for(0, int(batch.dimension(0)), [&](size_t i)
					{
						xhat.chip(i, 0) = xmu.chip(i, 0) * ivar.chip(0, 0);
					});

					//#step8 : Nor the two transformation steps
					if (gammax.dimension(0) != batch.dimension(0) || gammax.dimension(1) != batch.dimension(1) || gammax.dimension(2) != batch.dimension(2) || gammax.dimension(3) != batch.dimension(3))
						gammax = Eigen::Tensor<float, 4>(batch.dimensions());
					Concurrency::parallel_for(0, int(batch.dimension(0)), [&](size_t i)
					{
						gammax.chip(i, 0) = gamma.chip(0, 0) * xhat.chip(i, 0);
					});

					//#step9
					if (out.dimension(0) != batch.dimension(0) || out.dimension(1) != batch.dimension(1) || out.dimension(2) != batch.dimension(2) || out.dimension(3) != batch.dimension(3))
						out = Eigen::Tensor<float, 4>(batch.dimensions());
					Concurrency::parallel_for(0, int(batch.dimension(0)), [&](size_t i)
					{
						for (int i = 0; i < batch.dimension(0); ++i)
							out.chip(i, 0) = gammax.chip(i, 0) + beta.chip(0, 0);
					});

					pop_mean.device(thread_pool_device) = pop_mean * (float(m_step) / float(m_step + 1)) + mu * ((1.0f)/ float(m_step + 1));
					pop_var.device(thread_pool_device) = pop_var * (float(m_step) / float(m_step + 1)) + ( var * ((1.0f) / float(m_step + 1)) * (float(batch_size)/float(batch_size-1)));
					m_step += 1;
				}
				else {
					out = Eigen::Tensor<float, 4>(batch.dimensions());
					Concurrency::parallel_for(0, int(batch.dimension(0)), [&](size_t i)
					{
						out.chip(i,0) = gamma.chip(0, 0) * ((pop_var.chip(0, 0) + 1.0E-8f).sqrt().inverse()) * batch.chip(i, 0) +
							(beta.chip(0,0) - (gamma.chip(0, 0) * pop_mean.chip(0, 0) * ((pop_var.chip(0, 0) + 1.0E-8f).sqrt().inverse())));	
					});
				}
				return out;
			}

			Eigen::Tensor<float, 4> backprop(Eigen::Tensor<float, 4>& gradient) {
				Eigen::ThreadPoolTempl<Eigen::StlThreadEnvironment> tp(DBX_THREAD);
				Eigen::ThreadPoolDevice thread_pool_device(&tp, DBX_CORES);

				//#unfold the variables stored in cache
				//xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache
				//#get the dimensions of the input / output
				//N, D = dout.shape
				int batch_size = gradient.dimension(0);

				std::array<Eigen::DenseIndex, 4> tensor_dims = { 1, gradient.dimension(1), gradient.dimension(2), gradient.dimension(3) };

				//#step9
				//dbeta = np.sum(dout, axis = 0)
				Eigen::array<int, 1> dims = { 0 };
				if (dbeta.dimension(0) != 1 || dbeta.dimension(1) != gradient.dimension(1) || dbeta.dimension(2) != gradient.dimension(2) || dbeta.dimension(3) != gradient.dimension(3))
					dbeta = Eigen::Tensor<float, 4>(tensor_dims);
				dbeta.device(thread_pool_device) = gradient.sum(dims).reshape(tensor_dims);
				dgammax = gradient;

				//#step8
				//dgamma = np.sum(dgammax*xhat, axis = 0)
				//dxhat = dgammax * gamma
				if (dgamma.dimension(0) != 1 || dgamma.dimension(1) != gradient.dimension(1) || dgamma.dimension(2) != gradient.dimension(2) || dgamma.dimension(3) != gradient.dimension(3))
					dgamma = Eigen::Tensor<float, 4>(tensor_dims);
				dgamma.device(thread_pool_device) = (dgammax*xhat).sum(dims).reshape(tensor_dims);
				
				if (dxhat.dimension(0) != gradient.dimension(0) || dxhat.dimension(1) != gradient.dimension(1) || dxhat.dimension(2) != gradient.dimension(2) || dxhat.dimension(3) != gradient.dimension(3))
					dxhat = Eigen::Tensor<float, 4>(gradient.dimensions());
				Concurrency::parallel_for(0, int(gradient.dimension(0)), [&](size_t i)
				{
					dxhat.chip(i, 0) = dgammax.chip(i, 0) * gamma.chip(0, 0);
				});

				//#step7
				//divar = np.sum(dxhat*xmu, axis = 0)
				//dxmu1 = dxhat * ivar
				if (divar.dimension(0) != 1 || divar.dimension(1) != gradient.dimension(1) || divar.dimension(2) != gradient.dimension(2) || divar.dimension(3) != gradient.dimension(3))
					divar = Eigen::Tensor<float, 4>(tensor_dims);
				divar.device(thread_pool_device) = (dxhat*xmu).sum(dims).reshape(tensor_dims);
				
				if (dxmu1.dimension(0) != gradient.dimension(0) || dxmu1.dimension(1) != gradient.dimension(1) || dxmu1.dimension(2) != gradient.dimension(2) || dxmu1.dimension(3) != gradient.dimension(3))
					dxmu1 = Eigen::Tensor<float, 4>(gradient.dimensions());
				Concurrency::parallel_for(0, int(gradient.dimension(0)), [&](size_t i)
				{
					dxmu1.chip(i, 0) = dxhat.chip(i, 0) * ivar.chip(0, 0);
				});

				//#step6
				//dsqrtvar = -1. / (sqrtvar**2) * divar
				if (dsqrtvar.dimension(0) != 1 || dsqrtvar.dimension(1) != gradient.dimension(1) || dsqrtvar.dimension(2) != gradient.dimension(2) || dsqrtvar.dimension(3) != gradient.dimension(3))
					dsqrtvar = Eigen::Tensor<float, 4>(tensor_dims);
				dsqrtvar.device(thread_pool_device) = ((sqrtvar * sqrtvar).inverse()) * -1.0f * divar;

				//#step5
				//dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar
				if (dvar.dimension(0) != 1 || dvar.dimension(1) != gradient.dimension(1) || dvar.dimension(2) != gradient.dimension(2) || dvar.dimension(3) != gradient.dimension(3))
					dvar = Eigen::Tensor<float, 4>(tensor_dims);
				dvar.device(thread_pool_device) = 0.5f * ((var + 1.0E-8f).sqrt().inverse()) * dsqrtvar;

				//#step4
				//dsq = 1. / N * np.ones((N, D)) * dvar
				if (dsq.dimension(0) != gradient.dimension(0) || dsq.dimension(1) != gradient.dimension(1) || dsq.dimension(2) != gradient.dimension(2) || dsq.dimension(3) != gradient.dimension(3))
					dsq = Eigen::Tensor<float, 4>(gradient.dimensions());
				Concurrency::parallel_for(0, int(gradient.dimension(0)), [&](size_t i)
				{
					dsq.chip(i, 0) = (1.0f / batch_size) * dvar.chip(0, 0);
				});

				//#step3
				//dxmu2 = 2 * xmu * dsq
				if (dxmu2.dimension(0) != gradient.dimension(0) || dxmu2.dimension(1) != gradient.dimension(1) || dxmu2.dimension(2) != gradient.dimension(2) || dxmu2.dimension(3) != gradient.dimension(3))
					dxmu2 = Eigen::Tensor<float, 4>(gradient.dimensions());
				dxmu2.device(thread_pool_device) = 2.0f * xmu * dsq;

				//#step2
				//dx1 = (dxmu1 + dxmu2)
				//dmu = -1 * np.sum(dxmu1 + dxmu2, axis = 0)
				if (dx1.dimension(0) != gradient.dimension(0) || dx1.dimension(1) != gradient.dimension(1) || dx1.dimension(2) != gradient.dimension(2) || dx1.dimension(3) != gradient.dimension(3))
					dx1 = Eigen::Tensor<float, 4>(gradient.dimensions());
				dx1.device(thread_pool_device) = (dxmu1 + dxmu2);

				if (dmu.dimension(0) != 1 || dmu.dimension(1) != gradient.dimension(1) || dmu.dimension(2) != gradient.dimension(2) || dmu.dimension(3) != gradient.dimension(3))
					dmu = Eigen::Tensor<float, 4>(tensor_dims);
				dmu.device(thread_pool_device) = -1.0f * (dxmu1 + dxmu2).sum(dims).reshape(tensor_dims);

				//#step1
				//dx2 = 1. / N * np.ones((N, D)) * dmu
				if (dx2.dimension(0) != gradient.dimension(0) || dx2.dimension(1) != gradient.dimension(1) || dx2.dimension(2) != gradient.dimension(2) || dx2.dimension(3) != gradient.dimension(3))
					dx2 = Eigen::Tensor<float, 4>(gradient.dimensions());
				Concurrency::parallel_for(0, int(gradient.dimension(0)), [&](size_t i)
				{
					for (int i = 0; i < gradient.dimension(0); ++i)
						dx2.chip(i, 0) = (1.0f / batch_size) * dmu.chip(0, 0);
				});

				//#step0
				//dx = dx1 + dx2
				if (dx.dimension(0) != gradient.dimension(0) || dx.dimension(1) != gradient.dimension(1) || dx.dimension(2) != gradient.dimension(2) || dx.dimension(3) != gradient.dimension(3))
					dx = Eigen::Tensor<float, 4>(gradient.dimensions());
				dx.device(thread_pool_device) = dx1 + dx2;

				//beta -= dbeta;
				//gamma -= dgamma;
				return dx;
			}

			void update(int batch_size) {
				Eigen::Tensor<float, 4> tmp_w(2, beta.dimension(1), beta.dimension(2), beta.dimension(3));
				tmp_w.chip(0, 0) = beta.chip(0,0);
				tmp_w.chip(1, 0) = gamma.chip(0,0);
				Eigen::Tensor<float, 4> tmp_delta_w(2, beta.dimension(1), beta.dimension(2), beta.dimension(3));
				tmp_delta_w.chip(0, 0) = dbeta;
				tmp_delta_w.chip(1, 0) = dgamma;
				Eigen::Tensor<float, 3> tmp_b(beta.dimension(1), beta.dimension(2), beta.dimension(3)); tmp_b.setZero();
				Eigen::Tensor<float, 3> tmp_delta_b(beta.dimension(1), beta.dimension(2), beta.dimension(3)); tmp_delta_b.setZero();
				
				m_sgd_updater->update_sgd(tmp_b, tmp_w, tmp_delta_b, tmp_delta_w, batch_size);

				beta.chip(0, 0) = tmp_w.chip(0, 0);
				gamma.chip(0,0) = tmp_w.chip(1, 0);
			}

			void load_data(ByteBuffer& buffer) {

			}

			void save_data(ByteBuffer& buffer) {

			}
#pragma endregion
		};
	}
}
