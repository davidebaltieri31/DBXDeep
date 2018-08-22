#pragma once
#define EIGEN_USE_THREADS
#include <vector>
#include <math.h>
#include <Eigen\Eigen>
#include <unsupported\Eigen\CXX11\Tensor>
#include <random>
#include <ppl.h>
#include <opencv2\opencv.hpp>

class ColorGradient
{
protected:
	static std::vector<std::pair<float, cv::Vec3b>> m_gradients;

	static float linear_gradient(float start_value, float stop_value, float offset, float start_offset, float stop_offset) {
		return start_value + ((offset - start_offset) / (stop_offset - start_offset))*(stop_value - start_value);
	}
public:
	static void add_color(float offset, cv::Vec3b colore) {
		m_gradients.emplace_back(offset, colore);
		std::sort(m_gradients.begin(), m_gradients.end(), [](std::pair<float, cv::Vec3b>& a, std::pair<float, cv::Vec3b>& b) {
			return a.first < b.first; });
	}

	static cv::Vec3b calc_color(float val, float min_val, float max_val) {
		val = (val <= min_val) ? min_val+0.0001 : val;
		val = (val >= max_val) ? max_val-0.0001 : val;
		float v = (val - min_val) / (max_val - min_val);
		cv::Vec3b ret;
		for (int i = 0; i < m_gradients.size()-1; ++i) {
			if (m_gradients[i+1].first < v) continue;
			float b = linear_gradient(float(m_gradients[i].second.val[0]),
				float(m_gradients[i + 1].second.val[0]),
				v, m_gradients[i].first, m_gradients[i + 1].first);
			float g = linear_gradient(float(m_gradients[i].second.val[1]),
				float(m_gradients[i + 1].second.val[1]),
				v, m_gradients[i].first, m_gradients[i + 1].first);
			float r = linear_gradient(float(m_gradients[i].second.val[2]),
				float(m_gradients[i + 1].second.val[2]),
				v, m_gradients[i].first, m_gradients[i + 1].first);
			ret.val[0] = b;
			ret.val[1] = g;
			ret.val[2] = r;
			break;
		}
		return ret;
	}


};
