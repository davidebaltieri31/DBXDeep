#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <Eigen/Eigen>
#undef min

using net_datatype = double;
using myMat = Eigen::Matrix<net_datatype, -1, -1, Eigen::RowMajor>;

class DatasetLoader {
public:
	virtual bool load(std::vector<std::pair<myMat, myMat>>& training_data, 
		std::vector<std::pair<myMat, myMat>>& evaluation_data,
		std::vector<std::pair<myMat, myMat>>& test_data) = 0;

	virtual int get_size(int dim) = 0;

	virtual DatasetLoader* clone() = 0;

	//virtual myMat get_mean() = 0;

	//virtual myMat get_variance() = 0;
};

class MNISTLoaderMat : public DatasetLoader {
public:
	MNISTLoaderMat(int max_nun) {
		if (max_nun > 0)
			stop_at = max_nun;
	}
	int size_width;
	int size_height;
	int stop_at = INT_MAX;

	bool load(std::vector<std::pair<myMat, myMat>>& training_data,
		std::vector<std::pair<myMat, myMat>>& evaluation_data,
		std::vector<std::pair<myMat, myMat>>& test_data);

	int get_size(int dim) {
		if (dim == 2) return size_height;
		else if (dim == 3) return size_width;
		else return 1;
	}

	DatasetLoader* clone() {
		return new MNISTLoaderMat(*this);
	}
};

class TestDataset : public DatasetLoader {
public:
	TestDataset(int max_nun, int w, int h) {
		num_sample = max_nun;
		size_width = w;
		size_height = h;
	}
	int size_width;
	int size_height;
	int num_sample = INT_MAX;

	bool load(std::vector<std::pair<myMat, myMat>>& training_data,
		std::vector<std::pair<myMat, myMat>>& evaluation_data,
		std::vector<std::pair<myMat, myMat>>& test_data) {
		std::random_device rd;
		std::mt19937 gen(rd());
		std::normal_distribution<float> generate(0.0f, 0.8f);

		for (int i = 0; i < num_sample; ++i) {
			myMat sample;
			sample.resize(size_height, size_width);
			sample.setZero();
			myMat sample_label;
			sample_label.resize(1, 2);
			for (int j = 0; j < size_width*size_height / 10; ++j) {
				int ix = -1;
				int iy = -1;
				do {
					float x = std::abs(generate(gen));
					float y = std::abs(generate(gen));

					ix = int(x*size_width);
					iy = int(y*size_height);
				} while (ix<0 || ix>=size_width || iy<0 || iy>=size_height);

				if (i % 2 == 0) {
					sample(size_height - iy - 1, size_width - ix - 1) = 1.0;
					sample_label(0, 0) = 1.0;
					sample_label(0, 1) = 0.0;
				}
				else {
					sample(iy, ix) = 1.0;
					sample_label(0, 0) = 0.0;
					sample_label(0, 1) = 1.0;
				}
			}

			training_data.push_back(std::make_pair(sample, sample_label));
		}

		for (int i = 0; i < num_sample; ++i) {
			myMat sample;
			sample.resize(size_height, size_width);
			sample.setZero();
			myMat sample_label;
			sample_label.resize(1, 2);
			for (int j = 0; j < size_width*size_height / 10; ++j) {
				int ix = -1;
				int iy = -1;
				do {
					float x = std::abs(generate(gen));
					float y = std::abs(generate(gen));

					ix = int(x*size_width);
					iy = int(y*size_height);
				} while (ix<0 || ix >= size_width || iy<0 || iy >= size_height);

				if (i % 2 == 0) {
					sample(size_height - iy - 1, size_width - ix - 1) = 255;
					sample_label(0, 0) = 1.0;
					sample_label(0, 1) = 0.0;
				}
				else {
					sample(iy, ix) = 255;
					sample_label(0, 0) = 0.0;
					sample_label(0, 1) = 1.0;
				}
			}

			evaluation_data.push_back(std::make_pair(sample, sample_label));
		}

		for (int i = 0; i < num_sample; ++i) {
			myMat sample;
			sample.resize(size_height, size_width);
			sample.setZero();
			myMat sample_label;
			sample_label.resize(1, 2);
			for (int j = 0; j < size_width*size_height / 10; ++j) {
				int ix = -1;
				int iy = -1;
				do {
					float x = std::abs(generate(gen));
					float y = std::abs(generate(gen));

					ix = int(x*size_width);
					iy = int(y*size_height);
				} while (ix<0 || ix >= size_width || iy<0 || iy >= size_height);

				if (i % 2 == 0) {
					sample(size_height - iy - 1, size_width - ix - 1) = 255;
					sample_label(0, 0) = 1.0;
					sample_label(0, 1) = 0.0;
				}
				else {
					sample(iy, ix) = 255;
					sample_label(0, 0) = 0.0;
					sample_label(0, 1) = 1.0;
				}
			}

			test_data.push_back(std::make_pair(sample, sample_label));
		}
		return true;
	}

	int get_size(int dim) {
		if (dim == 2) return size_height;
		else if (dim == 3) return size_width;
		else return 1;
	}

	DatasetLoader* clone() {
		return new TestDataset(*this);
	}
};