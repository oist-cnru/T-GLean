/*
 * PvrnnModel class which takes care of instanciating and orchestrating other classes.
 */

#ifndef model_h
#define model_h

#include <string>
#include <vector>

#include "data.h"
#include "network.h"

class PvrnnModel {

public:
    PvrnnModel(int seed);
    ~PvrnnModel();

    Data* data = nullptr;
    PvrnnNetwork* network = nullptr;

    // Like this or snake through Cython???
    // /** getters for dynamic attributes **/
    // double raw_min() { return data->rawMin; };
    // double raw_max() { return data->rawMax; };

    const int seed;

    // Create data instance, setup dataset.
    void set_dataset(bool enable_norm, double raw_min, double raw_max, double norm_min, double norm_max,
                     string output_layer, int sm_unit, vector<double> sm_sigma,
                     int n_seq, int seq_len, int ouput_size, int minibatch_size,
                     float* data_buffer=nullptr);

    void train(int n_epoch, int save_interval, string save_dir, int start_epoch=0);
    // VectorXf er(Map<VectorXf> &input, Map<VectorXf> &mask, string directory, bool verbose=true, bool saveIterPredictions=false);
    void er(Map<VectorXf> &input, Map<VectorXf> &mask, Map<VectorXf> &output, string directory, bool verbose, bool saveIterPredictions);
};

#endif /* model_h */
