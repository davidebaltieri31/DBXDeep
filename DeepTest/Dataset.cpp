#include "Dataset.h"

bool load_images_set_as_row(std::string file_name, std::vector<myMat>& images, int stop_at)
{
	std::ifstream file(file_name, std::ios::binary | std::ios::in);
	if (!file.is_open()) return false;
	int magic_number = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&magic_number) + 3 - j, 1);
	if (magic_number != 2051) return false;
	int num_images = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&num_images) + 3 - j, 1);
	int num_rows = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&num_rows) + 3 - j, 1);
	int num_cols = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&num_cols) + 3 - j, 1);
	num_images = std::min(stop_at, num_images);
	for (int i = 0; i < num_images; ++i) {
		static std::vector<uint8_t> data;
		data.resize(num_rows * num_cols);
		file.read((char*)(data.data()), num_rows*num_cols);

		myMat data_f = myMat::Zero(1, num_rows*num_cols);
		for (int k = 0; k < num_rows*num_cols; ++k)
			data_f(0, k) = double(data[k]) / 255.0;
		images.push_back(data_f);
	}
	return true;
}

bool load_images_set_as_mat(std::string file_name, std::vector<myMat>& images, int stop_at)
{
	std::ifstream file(file_name, std::ios::binary | std::ios::in);
	if (!file.is_open()) return false;
	int magic_number = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&magic_number) + 3 - j, 1);
	if (magic_number != 2051) return false;
	int num_images = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&num_images) + 3 - j, 1);
	int num_rows = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&num_rows) + 3 - j, 1);
	int num_cols = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&num_cols) + 3 - j, 1);
	num_images = std::min(stop_at, num_images);
	for (int i = 0; i < num_images; ++i) {
		static std::vector<uint8_t> data;
		data.resize(num_rows * num_cols);
		file.read((char*)(data.data()), num_rows*num_cols);

		double mean = 0.0;
		for (int k = 0; k < data.size(); ++k) {
			mean += data[k];
		}
		mean /= double(data.size());

		myMat data_f = myMat::Zero(num_rows, num_cols);
		for (int l = 0; l < num_rows; ++l) {
			for (int k = 0; k < num_cols; ++k) {
				data_f(l, k) = (double(data[l*num_cols + k])-mean) / 255.0;
			}
		}
		images.push_back(data_f);
	}
	return true;
}

bool load_labels_set(std::string file_name, std::vector<myMat>& labels, int stop_at)
{
	std::ifstream file(file_name, std::ios::binary);
	if (!file.is_open()) return false;
	int magic_number = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&magic_number) + 3 - j, 1);
	if (magic_number != 2049) return false;
	int num_images = 0;
	for (int j = 0; j < 4; ++j)
		file.read((char*)(&num_images) + 3 - j, 1);
	num_images = std::min(stop_at, num_images);
	for (int i = 0; i < num_images; ++i) {
		unsigned char label = 0;
		file.read((char*)(&label), 1);

		myMat label_f = myMat::Zero(1, 10);
		label_f(0, label) = 1.0;
		labels.push_back(label_f);
	}
	return true;
}

bool loadMNIST_as_row(std::vector<std::pair<myMat, myMat>>& training_data, std::vector<std::pair<myMat, myMat>>& test_data, int stop_at)
{
	std::vector<myMat> training_images;
	if (!load_images_set_as_row("MNIST/train-images.idx3-ubyte", training_images, stop_at)) return false;
	std::vector<myMat> test_images;
	if (!load_images_set_as_row("MNIST/t10k-images.idx3-ubyte", test_images, stop_at)) return false;
	std::vector<myMat> training_labels;
	if (!load_labels_set("MNIST/train-labels.idx1-ubyte", training_labels, stop_at)) return false;
	std::vector<myMat> test_labels;
	if (!load_labels_set("MNIST/t10k-labels.idx1-ubyte", test_labels, stop_at)) return false;

	for (int i = 0; i < training_images.size(); ++i) {
		training_data.push_back(std::pair<myMat, myMat>(training_images[i], training_labels[i]));
	}

	for (int i = 0; i < test_images.size(); ++i) {
		test_data.push_back(std::pair<myMat, myMat>(test_images[i], test_labels[i]));
	}

	return true;
}

bool loadMNIST_as_mat(std::vector<std::pair<myMat, myMat>>& training_data, std::vector<std::pair<myMat, myMat>>& test_data, int stop_at)
{
	std::vector<myMat> training_images;
	if (!load_images_set_as_mat("MNIST/train-images.idx3-ubyte", training_images, stop_at)) return false;
	std::vector<myMat> test_images;
	if (!load_images_set_as_mat("MNIST/t10k-images.idx3-ubyte", test_images, stop_at)) return false;
	std::vector<myMat> training_labels;
	if (!load_labels_set("MNIST/train-labels.idx1-ubyte", training_labels, stop_at)) return false;
	std::vector<myMat> test_labels;
	if (!load_labels_set("MNIST/t10k-labels.idx1-ubyte", test_labels, stop_at)) return false;

	for (int i = 0; i < training_images.size(); ++i) {
		training_data.push_back(std::pair<myMat, myMat>(training_images[i], training_labels[i]));
	}

	for (int i = 0; i < test_images.size(); ++i) {
		test_data.push_back(std::pair<myMat, myMat>(test_images[i], test_labels[i]));
	}

	return true;
}

bool MNISTLoaderMat::load(std::vector<std::pair<myMat, myMat>>& training_data,
	std::vector<std::pair<myMat, myMat>>& evaluation_data,
	std::vector<std::pair<myMat, myMat>>& test_data) {
	bool success = loadMNIST_as_mat(training_data, test_data, stop_at);
	size_height = training_data[0].first.rows();
	size_width = training_data[0].first.cols();
	evaluation_data = training_data;
	std::random_shuffle(evaluation_data.begin(), evaluation_data.end());
	evaluation_data.resize(evaluation_data.size() / 10);
	return success;
}