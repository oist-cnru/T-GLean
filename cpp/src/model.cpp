#include <fenv.h>

#include "model.h"


PvrnnModel::PvrnnModel(int seed): seed(seed) {
    // Seems to create problems, e.g. when computing the sqrt() of
    // an eigen array with only zero values.
    // #ifdef __linux__  // stops when a NaN is produced in the code
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
    // #endif
};

PvrnnModel::~PvrnnModel() {
    if (network != nullptr) { delete network; }
};

void PvrnnModel::set_dataset(bool enable_norm, double raw_min, double raw_max, double norm_min, double norm_max,
                             string output_layer, int sm_unit, vector<double> sm_sigma,
                             int n_seq, int seq_len, int output_size, int minibatch_size,
                             float* data_buffer)
{
    data = new Data(enable_norm, raw_min, raw_max, norm_min, norm_max,
                    output_layer, sm_unit, sm_sigma,
                    n_seq, seq_len, minibatch_size);
    data->setRawDataC(data_buffer, n_seq, seq_len, output_size);

    network = new PvrnnNetwork(data, seed);
}


void PvrnnModel::train(int n_epoch, int save_interval, string save_dir, int start_epoch) {
    network->nEpoch = n_epoch;
    network->saveInterval = save_interval;
    network->saveDirectory = save_dir;
    network->train();
}

void PvrnnModel::er(Map<VectorXf> &input, Map<VectorXf> &mask, Map<VectorXf> &output, string directory, bool verbose, bool saveIterPredictions) {
    if (mask.size() == 0) {
        VectorXf output_ = network->onlineErrorRegression(input, vector<VectorXf>(), directory, verbose, saveIterPredictions);
        output = output_;
    } else {
        vectorXf1DContainer seq_mask;
        for (int i = 0; i < data->seqLen; i++) { seq_mask.push_back(mask); }
        VectorXf output_ = network->onlineErrorRegression(input, seq_mask, directory, verbose, saveIterPredictions);
        output = output_;
    }
}
