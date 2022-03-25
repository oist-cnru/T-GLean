#pragma once

#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <limits>

#include "../../toml/toml.h"
#include "INIReader.h"
#include <network.h>

static inline int countItemsInList(string list) noexcept {
    if (list.empty()) return 0;
    else return (int)std::count(list.begin(), list.end(), ',')+1;
}

static inline void parseStringIntList(string list, vector<int> &vlist, bool reverse=false, int pad_min=0) noexcept {
    stringstream slist(list);
    vlist.clear();
    while (slist.good()) {
        string item;
        getline(slist, item, ',');
        try {
            if (!reverse) vlist.push_back(stoi(item));
            else vlist.insert(vlist.begin(), stoi(item));
        } catch (const std::exception &e) {
            std::cerr << "warning: Failed to parse int list item " << item << std::endl;
        }
    }
    while (vlist.size() < pad_min) {
        if (!reverse) vlist.push_back(vlist.back());
        else vlist.insert(vlist.begin(), vlist.front());
    }
}

static inline void parseStringDblList(string list, vector<double> &vlist, bool reverse=false, int pad_min=0) noexcept {
    stringstream slist(list);
    vlist.clear();
    while (slist.good()) {
        string item;
        getline(slist, item, ',');
        try {
            if (!reverse) vlist.push_back(stod(item));
            else vlist.insert(vlist.begin(), stod(item));
        } catch (const std::exception &e) {
            std::cerr << "warning: Failed to parse double list item " << item << std::endl;
        }
    }
    while (vlist.size() < pad_min) {
        if (!reverse) vlist.push_back(vlist.back());
        else vlist.insert(vlist.begin(), vlist.front());
    }
}

/* GPvrnn: A GLean-like C++ interface for LibPvrnn */
class GPvrnn {
    static GPvrnn* pvrnn;
    private:
        PvrnnNetwork* model;
        struct pvrnnConfig pconfig;
        double last_err;
        float *mask = nullptr; // internal mask handling
        float *goal_dmask = nullptr;
        float *u_dmask = nullptr;
        int mask_drange[2] = {-1, -1}; // start and end of masked dimensions (before softmax)
        int _n_seq;
        int _dims;
        int _max_timesteps;
        int _softmax_quant;
        int _n_layers; // internal counter plus 1
        int *_d_neurons = nullptr;
        int *_z_units = nullptr;
        double *_meta_prior = nullptr;
        int _max_epochs;
        int _save_epochs;
        int _current_epoch;
        int _postdiction_window_length; // current postdiction (ER) window length
        int _planning_window_length;    // current planning window length (0 when doing normal ER)
        int _max_window_length;         // max window length including postdiction and planning
        string _training_path; // save directory when training
        string _output_dir;    // output for ER
        string _data_path;     // might not match internal datasetDirectory if loaded externally
        string _base_dir;      // path to config
    public:
        static GPvrnn* getInstance() noexcept {
            return pvrnn;
        }

        GPvrnn() noexcept {
            model = nullptr;
        }

        /**
        * Create an instance of LibPvrnn with the GPvrnn interface
        * @param config_file GLean INI or pvrnn TOML network configuration file
        * @param task sets up generation mode. can be training, testing, planning, online_error_regression
        * @param config_type optional specification of glean or toml config type. If left blank, detect from file extension
        * @param rng_seed optional fixed RNG seed. Defaults to random seed
        **/
        GPvrnn(string config_file, string task="training", int rng_seed=-1, string config_type="") noexcept {
            newModel(config_file, task, rng_seed, config_type);
        }

        ~GPvrnn() noexcept {
            if (model != nullptr) delete model;
            if (_meta_prior != nullptr) free(_meta_prior);
            if (_d_neurons != nullptr) free(_d_neurons);
            if (_z_units != nullptr) free(_z_units);
            if (mask != nullptr) free(mask);
            if (goal_dmask != nullptr) free(goal_dmask);
            if (u_dmask != nullptr) free(u_dmask);
        }

        inline void newModel(string config_file="", string task="training", int rng_seed=-1, string config_type="") noexcept {
            model = new PvrnnNetwork(true, rng_seed); // use weighted KLD
            if (model == nullptr) {
                std::cerr << "newModel: Failed to instantiate model" << std::endl;
                exit(EXIT_FAILURE);
            }

            if (config_file == "") {
                std::cout << "newModel: No config loaded" << std::endl;
                return;
            } else if (config_type == "" || (config_type != "glean" && config_type != "toml")) { // attempt to guess from file extension
                if (std::filesystem::path(config_file).extension() == ".cfg" || std::filesystem::path(config_file).extension() == ".ini") {
                    config_type = "glean";
                } else if (std::filesystem::path(config_file).extension() == ".toml") {
                    config_type = "toml";
                } else {
                    std::cerr << "newModel: Unsupported configuration file " << config_file << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            if (config_type == "glean" && !importGConfig(config_file, task)) {
                std::cerr << "newModel: Failed to load GLean config" << endl;
                exit(EXIT_FAILURE);
            } else if (config_type == "toml" && !importTOMLConfig(config_file, task)) {
                std::cerr << "newModel: Failed to load TOML config" << endl;
                exit(EXIT_FAILURE);
            }
            _base_dir = std::filesystem::path(config_file).parent_path(); // by convention the base directory is the config directory
            modelInitialize();
        }

        inline void modelInitialize() noexcept {
            try {
                model->initializeNetwork(true); // use zero initialization
            } catch (const std::exception& e) {
                std::cerr << "modelInitialize: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
            _n_seq = model->config->nSeq;
            _dims = model->config->outputSize;
            _max_timesteps = model->config->seqLen;
            _softmax_quant = (model->config->smUnit > 0 ? model->config->smUnit : 1); // for bc assume that in FC mode quant = 1
            _max_epochs = model->config->nEpoch;
            _save_epochs = model->config->saveInterval;
            _training_path = model->config->saveDirectory;
            _n_layers = model->nLayers + 1;
            _meta_prior = (double *)checkedMalloc(sizeof(double)*_n_layers-1);
            _d_neurons = (int *)checkedMalloc(sizeof(int)*_n_layers-1);
            _z_units = (int *)checkedMalloc(sizeof(int)*_n_layers-1);
            for (int l = 0; l < _n_layers-1; l++) {
                _meta_prior[l] = model->config->w[l];
                _d_neurons[l] = model->config->dSize[l];
                _z_units[l] = model->config->zSize[l];
            }
        }

        inline void trainInitialize(int startEpoch=0) noexcept {
            if (startEpoch == -1) startEpoch = model->config->epochToLoad(0); // load last saved model, with no fallback load
            try {
                model->initTraining(startEpoch);
            } catch (const std::exception& e) {
                std::cerr << "trainInitialize: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
            _current_epoch = startEpoch;
            last_err = std::numeric_limits<double>::max();
        }

        inline void testInitialize(int epoch=-1) noexcept {
            try {
                model->testInitialize(epoch);
            } catch (const std::exception& e) {
                std::cerr << "trainInitialize: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        inline void batchErInitialize(int epoch=-1) noexcept {
            try {
                model->erInitialize(epoch);
            } catch (const std::exception& e) {
                std::cerr << "batchErInitialize: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
            _max_timesteps = model->config->erSeqLen;
            _max_epochs = model->config->nItr;
            _postdiction_window_length = model->config->window;
            _planning_window_length = 0;
            _max_window_length = model->config->window;
            _output_dir = model->config->erSaveDirectory;
        }

        inline void onlineErInitialize(int epoch=-1) noexcept {
            try {
                model->onlineErInitialize(epoch);
            } catch (const std::exception& e) {
                std::cerr << "onlineErInitialize: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
            _max_timesteps = model->config->totalStep;
            _max_epochs = model->config->nItr;
            _postdiction_window_length = (model->config->gWindow ? 0 : model->config->window);
            _planning_window_length = 0;
            _max_window_length = model->config->window;
            _output_dir = model->config->erSaveDirectory;
        }

        inline void planInitialize(int epoch=-1) noexcept {
            try {
                // model->erInitialize(epoch);
                model->onlinePlanInitialize(epoch);
            } catch (const std::exception& e) {
                std::cerr << "planInitialize: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
            _max_timesteps = model->config->erSeqLen;
            _max_epochs = model->config->nItr;
            _current_epoch = 0;
            _postdiction_window_length = 1;
            _planning_window_length = _max_timesteps-1;
            _max_window_length = _max_timesteps;
            _output_dir = model->config->erSaveDirectory;
            if (mask_drange[0] == -1 || mask_drange[1] == -1) {
                std::cout << "planInitialize: Planning mask not set up by config" << std::endl;
            }
            mask = (float *)calloc(_max_window_length*_dims*_softmax_quant, sizeof(float)); if (mask == NULL) abort();
        }

        // Calls internal training routine
        inline void train() noexcept {
            try {
                model->train();
            } catch (const std::exception& e) {
                std::cerr << "train: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        /**
        * Execute a preset number of training epochs then return loss
        * Will block until background_epochs have passed. No error handling.
        * @param background_epochs number of epochs to run before returning
        * @param rec_loss current training reconstruction loss
        * @param reg_loss current training regularization loss (all layers)
        * @param greedy_train save only when current total loss is lower than previous save
        * */
        inline void train(int background_epochs, double &rec_loss, double &reg_loss, bool greedy_train=true) noexcept {
            int target_epoch = _current_epoch + background_epochs < _max_epochs ? _current_epoch + background_epochs : _max_epochs;
            for (int e = _current_epoch; e < target_epoch; e++) {
                rec_loss = 0.0;
                reg_loss = 0.0;
                model->trainOneEpoch(rec_loss, reg_loss);
                if (((e+1) % _save_epochs == 0 || (!greedy_train && e == _max_epochs-1)) && (!greedy_train || (greedy_train && rec_loss+reg_loss < last_err))) {
                    model->saveTraining();
                    last_err = rec_loss+reg_loss;
                }
                _current_epoch++;
            }
        }

        inline void postGenAndSave(string directory) noexcept {
            try {
                model->postGenAndSave(directory);
            } catch (const std::exception& e) {
                std::cerr << "postGenAndSave: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        inline void postGenAndSave() noexcept {
            postGenAndSave(std::filesystem::path(_training_path) / std::filesystem::path("posterior_generation"));
        }

        inline void priGenAndSave(int post_steps, int total_steps, string directory) noexcept {
            try {
                model->priGenAndSave(post_steps, total_steps, directory);
            } catch (const std::exception& e) {
                cerr << "priGenAndSave: " << e.what() << endl;
                exit(EXIT_FAILURE);
            }
        }

        inline void priGenAndSave(int post_steps) noexcept {
            priGenAndSave(post_steps, _max_timesteps, std::filesystem::path(_training_path) / std::filesystem::path("prior_generation_poststep" + to_string(post_steps)));
        }

        // GLean standard generation test (prior & posterior)
        inline void test() noexcept {
            postGenAndSave(std::filesystem::path(_training_path) / std::filesystem::path("posterior_generation"));
            priGenAndSave(0, _max_timesteps, std::filesystem::path(_training_path) / std::filesystem::path("prior_generation"));
        }

        /**
         * Executes one step of prior generation. No error handling.
         * @param output float buffer of size dims to hold output
         * */
        inline void priorGeneration(float* output) noexcept {
            model->priorGeneration(0, output);
        }

        inline void batchErrorRegression(double &rec_loss, double &reg_loss, string directory="", bool verbose=false, bool save_iter_predictions=true) noexcept {
            if (directory != "") directory = std::filesystem::path(_output_dir) / std::filesystem::path(directory);
            for (int i = 0; i < model->config->erStep; i++) model->errorRegressionStep(i, directory, verbose, save_iter_predictions);
            rec_loss = model->getRecErrFromLog(-1);
            reg_loss = 0.0;
            for (int l = 0; l < _n_layers-1; l++) reg_loss += model->getKldFromLog(l, -1);
            _current_epoch += _max_epochs;
        }

        inline void onlineErrorRegression(float *_input, float *output, double &rec_loss, double &reg_loss, string path="", bool verbose=false, bool save_iter_predictions=true) noexcept {
            if (path != "") path = std::filesystem::path(_output_dir) / std::filesystem::path(path);
            model->onlineErrorRegression(_input, output, path, verbose, save_iter_predictions);
            _postdiction_window_length += (_postdiction_window_length != model->config->window ? 1 : 0);
            rec_loss = model->getRecErrFromLog(-1);
            reg_loss = 0.0;
            for (int l = 0; l < _n_layers-1; l++) reg_loss += model->getKldFromLog(l, -1);
            _current_epoch += _max_epochs;
        }

        inline void onlineErrorRegression(float *_input, float *output, float *_mask, double &rec_loss, double &reg_loss, string path="", bool verbose=false, bool save_iter_predictions=true) noexcept {
            if (path != "") path = std::filesystem::path(_output_dir) / std::filesystem::path(path);
            model->onlineErrorRegression(_input, output, _mask, path, verbose, save_iter_predictions);
            _postdiction_window_length += (_postdiction_window_length != model->config->window ? 1 : 0);
            rec_loss = model->getRecErrFromLog(-1);
            reg_loss = 0.0;
            for (int l = 0; l < _n_layers-1; l++) reg_loss += model->getKldFromLog(l, -1);
            _current_epoch += _max_epochs;
        }

        /* Plan using externally passed mask */
        inline void plan(float *_mask, double &rec_loss, double &reg_loss, string path="", bool verbose=false, bool save_iter=true) noexcept {
            if (path != "") path = std::filesystem::path(_output_dir) / std::filesystem::path(path);
            model->errorRegressionStep(0, _mask, path, verbose, save_iter);
            rec_loss = model->getRecErrFromLog(-1);
            reg_loss = 0.0;
            for (int l = 0; l < _n_layers-1; l++) reg_loss += model->getKldFromLog(l, -1);
            _current_epoch += _max_epochs;
        }

        /* Plan using internal mask */
        inline void plan(double &rec_loss, double &reg_loss, string path="", bool verbose=false, bool save_iter=true) noexcept {
            if (path != "") path = std::filesystem::path(_output_dir) / std::filesystem::path(path);
            model->errorRegressionStep(0, mask, path, verbose, save_iter);
            rec_loss = model->getRecErrFromLog(-1);
            reg_loss = 0.0;
            for (int l = 0; l < _n_layers-1; l++) reg_loss += model->getKldFromLog(l, -1);
            _current_epoch += _max_epochs;
        }

        /* Dynamic length planner with external mask */
        inline void plan(float *_input, float *_mask, double &rec_loss, double &reg_loss, string path="", bool dynamic=true, bool save_iter=true) noexcept {
            if (path != "") path = std::filesystem::path(_output_dir) / std::filesystem::path(path);
            model->onlinePlanGeneration(_input, _mask, path, dynamic, save_iter);
            rec_loss = model->getRecErrFromLog(-1);
            reg_loss = 0.0;
            for (int l = 0; l < _n_layers-1; l++) reg_loss += model->getKldFromLog(l, -1);
            _current_epoch += _max_epochs;
        }

        inline void setData(float *data, int count, int length, int _dims) noexcept {
            try {
                model->data->setRawDataC(data, count, length, _dims); // GPvrnn only supports native-order data
            } catch (const std::exception& e) {
                std::cerr << "setData: " << e.what() << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        inline void setData(float *data) noexcept {
            setData(data, _n_seq, _max_timesteps, _dims);
        }

        /* Set a single sequence of data for ER */
        inline void setErData(float *data, int length, int _dims) noexcept {
            model->data->erSetRawData(data, length, _dims);
        }

        /* Set a single sequence of data for ER */
        inline void setErData(float *data) noexcept {
            model->data->erSetRawData(data, _max_timesteps, _dims);
        }

        /* Set a single sequence of data for planning (same as ER) */
        inline void setPlanInput(float *data) noexcept {
            model->data->erSetRawData(data, _max_timesteps, _dims);
        }

        /* Get generated output from ER */
        inline void getErOutput(float *output) noexcept {
            model->getErSequence(output);
        }

        /* Get generated output from planning (same as ER) */
        inline void getPlanOutput(float *output) noexcept {
            model->getErSequence(output);
        }

        inline void getPosteriorA(float *Amyu, float *Asigma) noexcept {
            model->getErASequence(Amyu, Asigma);
        }

        inline void getPriorMyuSigma(float *myu, float *sigma) noexcept {
            model->getErPriorMuSigmaSequence(myu, sigma);
        }

        inline void getPosteriorMyuSigma(float *myu, float *sigma) noexcept {
            model->getErPosteriorMuSigmaSequence(myu, sigma);
        }

        inline void getFullErRegErr(int layer, int epoch, double *kld) noexcept {
            model->getErKldFromLog(layer, epoch, kld);
        }

        inline void getFullErRecErr(int epoch, double *rec) noexcept {
            model->getErRecErrFromLog(epoch, rec);
        }

        /*
        * Set up the planning mask. Will expand masks to softmax dimensions
        * Mask must be managed by the interface (set by config)
        */
        inline void setMask() noexcept {
            for (int t = 0; t < _max_timesteps; t++) {
                if (t == 0) for (int d = 0; d < _dims; d++) std::fill(mask+(d*_softmax_quant), mask+((d+1)*_softmax_quant), u_dmask[d]);
                else for (int d = 0; d < _dims; d++) std::fill(mask+(t*_dims*_softmax_quant)+(d*_softmax_quant), mask+(t*_dims*_softmax_quant)+((d+1)*_softmax_quant), goal_dmask[d]);
            }
        }

        /**
        * Set up the planning mask. Will expand masks to softmax dimensions
        *
        * @param imask mask applied to first timestep
        * @param fmask mask applied to subsequent timesteps
        **/
        inline void setMask(float *imask, float *fmask) noexcept {
            for (int t = 0; t < _max_timesteps; t++) {
                if (t == 0) for (int d = 0; d < _dims; d++) std::fill(mask+(d*_softmax_quant), mask+((d+1)*_softmax_quant), imask[d]);
                else for (int d = 0; d < _dims; d++) std::fill(mask+(t*_dims*_softmax_quant)+(d*_softmax_quant), mask+(t*_dims*_softmax_quant)+((d+1)*_softmax_quant), fmask[d]);
            }
        }

        /**
        * Update the planning mask (unmask) at timestep t. Will expand masks to softmax dimensions
        * Mask must be managed by the interface (set by config)
        * 
        * @param t timestep this mask applies to
        **/
        inline void unmask(int t) noexcept {
            for (int d = 0; d < _dims; d++) std::fill(mask+(t*_dims*_softmax_quant)+(d*_softmax_quant), mask+(t*_dims*_softmax_quant)+((d+1)*_softmax_quant), u_dmask[d]);
        }
        /**
        * Update the planning mask at timestep t. Will expand masks to softmax dimensions
        *
        * @param t timestep this mask applies to
        * @param dmask mask (mask can contain any real value, usually 0.0 or 1.0)
        **/
        inline void setMask(int t, float *dmask) noexcept {
            for (int d = 0; d < _dims; d++) std::fill(mask+(t*_dims*_softmax_quant)+(d*_softmax_quant), mask+(t*_dims*_softmax_quant)+((d+1)*_softmax_quant), dmask[d]);
        }

        /* Config loaders: GLean INI and pvrnn TOML are supported */

        /**
        * Parse GLean configs for LibPvrnn. Only a subset of GLean/TF features are supported
        * 
        * @param cfg_path path to TOML configuration path
        * @param task string representing current active task. See below for valid tasks
        * 
        * Task can be:
        * training: self explanatory
        * testing: don't initialize variables for training
        * planning: sets up static planner using batch ER (includes online planning)
        * online_error_regression: sets up generation using online ER using planner configuration (not recommended for use)
        *
        * Notes:
        * error regression is not supported -- planning and ER configurations overlap
        **/
        inline bool importGConfig(string cfg_path, string task="training") noexcept {
            if (task != "training" && task != "testing" && task != "planning" && task != "online_error_regression") {
                std::cerr << "importGConfig: Unknown task configuration " << task << std::endl;
                return false;
            }
            // Parse config
            INIReader config(cfg_path);

            if (config.ParseError() != 0) {
                std::cerr << "importGConfig: Failed to parse config file " << cfg_path << std::endl;
                return false;
            } else {
                std::cout << "importGConfig: Loaded config " << cfg_path << " (" << task << ")" << std::endl;
            }

            string modality = config.Get("network", "modalities", "motor");
            if (modality.find(',') != string::npos) {
                modality = modality.substr(0, modality.find(','));
                std::cerr << "importGConfig: GLean/pvrnn-cpp does not support multiple modalities, using first modality " << modality << " only" << std::endl;
            }
            string _layers_neurons = config.Get("network", modality + "_layers_neurons", "missing layers_neurons");
            pconfig.nLayers = countItemsInList(_layers_neurons);
            parseStringIntList(_layers_neurons, pconfig.dSize, true);
            string _layers_z_units = config.Get("network", modality + "_layers_z_units", "missing layers_z_units");
            parseStringIntList(_layers_z_units, pconfig.zSize, true);
            string _layers_param = config.Get("network", modality + "_layers_param", "missing layers_param");
            parseStringIntList(_layers_param, pconfig.tau, true);
            string _meta_prior = config.Get("network", modality + "_meta_prior", "missing meta_prior");
            parseStringDblList(_meta_prior, pconfig.w, true, pconfig.nLayers);
            string _ugaussian_weight = config.Get("network", "ugaussian_weight", "missing ugaussian_weight");
            parseStringDblList(_ugaussian_weight, pconfig.beta, true, pconfig.nLayers);
            /* Fall back to training settings if not explicitly set */
            string _er_meta_prior = config.Get("planning", modality + "_meta_prior", "");
            parseStringDblList((_er_meta_prior == "" ? _meta_prior : _er_meta_prior), pconfig.erW, true, pconfig.nLayers);
            string _er_ugaussian_weight = config.Get("planning", "ugaussian_weight", "");
            parseStringDblList((_er_ugaussian_weight == "" ? _ugaussian_weight : _er_ugaussian_weight), pconfig.erBeta, true, pconfig.nLayers);

            string _softmax_sigma = config.Get("data", modality + "_softmax_sigma", "0.05");
            parseStringDblList(_softmax_sigma, pconfig.smSigma);
            pconfig.nSeq = config.GetInteger("data", "sequences", -1);
            pconfig.seqLen = config.GetInteger("data", "max_timesteps", -1);
            pconfig.outputSize = config.GetInteger("data", modality + "_dims", -1);
            pconfig.dataMin = config.GetReal("data", modality + "_softmax_min", -1.0);
            pconfig.dataMax = config.GetReal("data", modality + "_softmax_max", -1.0);
            pconfig.smUnit = config.GetInteger("data", "softmax_quant", 10);
            pconfig.outputLayer = (pconfig.smUnit <= 1 ? "fc" : "sm"); // if quant is zero or one, use FC output (not recommended)
            pconfig.minibatchSize = (task == "planning" ? 1 : pconfig.nSeq); // by default GLean did batched training (manual setting currently not supported)
            pconfig.nEpoch = config.GetInteger("training", "max_epochs", 1000);
            pconfig.alpha = config.GetReal("training", "learning_rate", -1.0); // task learning rate takes priority
            pconfig.alpha = (pconfig.alpha == -1.0 ? config.GetReal("network", "learning_rate", 0.001) : pconfig.alpha);
            pconfig.beta1 = 0.9;
            pconfig.beta2 = 0.999;
            pconfig.a_eps = config.GetReal(task, "opt_epsilon", -1.0);
            pconfig.a_eps = (pconfig.a_eps == -1.0 ? (task == "planning" ? pconfig.erAlpha / 10.0 : pconfig.alpha / 10.0) : pconfig.a_eps);
            pconfig.sigmaMaxVal = 0.0; // disable sigma clipping
            pconfig.sigmaMinVal = 0.0; // disable sigma clipping
            pconfig.saveInterval = config.GetInteger(task, "save_interval", (task == "training" ? 1000 : 100));
            pconfig.optimizerAlgorithm = config.Get("network", "optimizer", "adam");
            
            char* _save_dir_prefix = std::getenv("PVRNN_SAVE_DIR");
            string save_dir_prefix;
            if (_save_dir_prefix != nullptr) save_dir_prefix = _save_dir_prefix;
            else save_dir_prefix = "./"; // fallback
            pconfig.saveDirectory = std::filesystem::path(save_dir_prefix) / std::filesystem::path("results/") / ::filesystem::path(config.Get("data", "training_path", ""));
            _data_path = config.Get("data", modality + "_path", "");
            /* Load metadata instead of reloading data if possible */
            if (task != "training")
                pconfig.datasetDirectory = "";
            else
                pconfig.datasetDirectory = _data_path;
            pconfig.erSaveDirectory = config.Get("planning", "planning_path", "");
            pconfig.erSaveDirectory = (pconfig.erSaveDirectory == "" ? std::filesystem::path(pconfig.saveDirectory) / std::filesystem::path("planning_output") : std::filesystem::path(save_dir_prefix) / std::filesystem::path("results/") / std::filesystem::path(pconfig.erSaveDirectory));
            pconfig.baseDirectory = cfg_path;
            pconfig.epochToLoad = -1; // load last saved epoch

            /* General ER/planning settings */
            pconfig.erOptimizerAlgorithm = config.Get("planning", "optimizer", pconfig.optimizerAlgorithm);
            pconfig.erAlpha = config.GetReal("planning", "learning_rate", -1.0);
            pconfig.erAlpha = (pconfig.erAlpha == -1.0 ? pconfig.alpha*100.0 : pconfig.erAlpha);
            pconfig.erBeta1 = 0.9;
            pconfig.erBeta2 = 0.999;
            pconfig.nItr = config.GetInteger("planning", "max_epochs", 100);
            pconfig.nInitItr = config.GetInteger("planning", "init_epochs", pconfig.nItr); // Warmup epochs. Only applies to batched ER

            /* The meaning of these settings will change if we are in planning mode or ER mode */
            pconfig.erStep = 1; // Number of steps batched ER makes. With the static planner this is always 1
            pconfig.totalStep = config.GetInteger("planning", "prediction_steps", 1); // Number of steps online ER makes. This controls the length of the output
            pconfig.gWindow = (task == "online_error_regression" ? 1 : 0);
            pconfig.erSeqLen = pconfig.seqLen; // Length of the output from batched ER. With the static planner this is always seqLen
            pconfig.window = (task == "online_error_regression" ? config.GetInteger("planning", "postdiction_horizon", pconfig.seqLen / 4) : pconfig.seqLen); // Static planner always has maxed out window
            pconfig.predStep = (task == "online_error_regression" ? config.GetInteger("planning", "prediction_horizon", 1) : 0); // How many steps of prior generation to use. Doesn't apply to static planner
            pconfig.erDataDirectory = config.Get("planning", modality + "_path", "");
            if (task != "training" && task != "testing" && pconfig.erDataDirectory != "")
                _data_path = pconfig.erDataDirectory;
            pconfig.erBaseDirectory = cfg_path;

            string _goal_mask = config.Get("planning", "goal_modalities_mask", "");
            if (_goal_mask != "") {
                std::vector<int> _mrange;
                parseStringIntList(_goal_mask, _mrange);
                if (_mrange.size() == 2) { // manage goal mask
                    mask_drange[0] = std::max(_mrange[0], 0); // min
                    mask_drange[1] = std::min(_mrange[1], pconfig.outputSize-1); // max
                    goal_dmask = (float *)calloc(pconfig.outputSize, sizeof(float)); if (goal_dmask == NULL) abort();
                    for (int d = mask_drange[0]; d <= mask_drange[1]; d++) goal_dmask[d] = 1.0f;
                    u_dmask = (float *)checkedMalloc(sizeof(float)*pconfig.outputSize);
                    for (int d = 0; d < pconfig.outputSize; d++) u_dmask[d] = 1.0f;
                } else if (_mrange.size() > 0) std::cerr << "importGConfig: Goal mask invalid" << std::endl;
            }

            model->config->importConfig(pconfig);
            return true;
        }

        /**
        * Parse TOML configs for pvrnn-cpp, with GLean planner extensions
        * 
        * @param cfg_path path to TOML configuration path
        * @param task string representing current active task. See below for valid tasks
        * @param cfg_str string containing TOML to be parsed. Overrides cfg_path and load, use for changing settings after loading base config
        * @param load boolean indicating whether this is the final configuration to be loaded. Used recursively, don't set manually
        *
        * Task can be:
        * training: self explanatory
        * testing: don't initialize variables for training
        * planning: load ER settings for planning
        * error_regression: load ER settings for ER
        * online_error_regression: as above (no online_er section?)
        *
        * Notes:
        * Other extensions such as branch network and developmental learning are not supported
        * The following options are ignored: norm.enable, norm.min, norm.max, zero_init, backend, wKLD, erRandomInitA, rng_seed, layer type
        **/
        inline bool importTOMLConfig(string cfg_path, string task="training", string cfg_str="", bool load=true) noexcept {
            if (task != "training" && task != "testing" && task != "planning" && task != "error_regression" && task != "online_error_regression") {
                std::cerr << "importTOMLConfig: Unknown task configuration " << task << std::endl;
                return false;
            }
            toml::table config;
            if (cfg_str.empty()) {
                try {
                    config = toml::parse_file(cfg_path);
                } catch (const toml::parse_error& err) {
                    std::cerr << "importTOMLConfig: Failed to parse config file " << cfg_path << std::endl;
                    return false;
                }
                std::cout << "importTOMLConfig: Loaded config " << cfg_path  << " (" << task << ")" << std::endl;
            } else {
                try {
                    config = toml::parse(cfg_str);
                } catch (const toml::parse_error& err) {
                    std::cerr << "importTOMLConfig: Failed to parse TOML " << cfg_str << std::endl;
                    return false;
                }
                load = false; // bypass normal load checks
            }

            if (config["base"]) { // load base config first
                if (!importTOMLConfig(std::filesystem::path(cfg_path).parent_path() / std::filesystem::path(config["base"].value_or("")), task, cfg_str, false)) return false;
            }

            auto layers = config["network"]["layers"];
            if (layers && layers.as_array()->size() > 1) {
                pconfig.nLayers = layers.as_array()->size()-1; // not including output layer
                pconfig.dSize.clear();
                pconfig.zSize.clear();
                pconfig.tau.clear();
                pconfig.w.clear();
                pconfig.beta.clear();

                for (int l = pconfig.nLayers; l > 0; l--) {
                    if (!layers[l]["d"]) { std::cerr << "importTOMLConfig: Missing required 'd' for layer " << l << std::endl; return false; }
                    pconfig.dSize.push_back(layers[l]["d"].value_or<int>(-1));
                    if (!layers[l]["z"]) { std::cerr << "importTOMLConfig: Missing required 'z' for layer " << l << std::endl; return false; }
                    pconfig.zSize.push_back(layers[l]["z"].value_or<int>(-1));
                    if (!layers[l]["tau"]) { std::cerr << "importTOMLConfig: Missing required 'tau' for layer " << l << std::endl; return false; }
                    pconfig.tau.push_back(layers[l]["tau"].value_or<int>(-1));
                    if (!layers[l]["w"]) { std::cerr << "importTOMLConfig: Missing required 'w' for layer " << l << std::endl; return false; }
                    pconfig.w.push_back(layers[l]["w"].value_or<double>(-1.0));
                    if (!layers[l]["beta"]) { std::cerr << "importTOMLConfig: Missing required 'beta' for layer " << l << std::endl; return false; }
                    pconfig.beta.push_back(layers[l]["beta"].value_or<double>(-1.0));
                }
            }

            if (load && pconfig.nLayers == 0) { std::cerr << "importTOMLConfig: Missing required 'layers'" << std::endl; return false; }

            string ertask = "er";
            if (config["planning"] && task != "error_regression" && task != "online_error_regression") ertask = "planning";  // default to planning if not explicitly doing ER
            /* Unfortunately there's no way to know where the previous values came from, so just emit a warning */
            if (load && task == "planning" && ertask == "er" && config["er"]) std:cerr << "importTOMLConfig: Missing [planning] section in top level config, some values may be overridden by [er]" << std::endl;

            /* ER settings: Fall back to training settings if not explicitly set */
            auto er = config[ertask];
            if (er) {
                if (er["w"]) {
                    pconfig.erW.clear();
                    if (er["w"].as_array()->size() == pconfig.nLayers) {
                        for (int l = 0; l < pconfig.nLayers; l++)
                            pconfig.erW.push_back(er["w"][l].value_or<double>((double)pconfig.w[l]));
                    } else {
                        for (int l = 0; l < pconfig.nLayers; l++)
                            pconfig.erW.push_back(er["w"][0].value_or<double>((double)pconfig.w[l]));
                    }
                }
                if (er["beta"]) {
                    pconfig.erBeta.clear();
                    if (er["beta"].as_array()->size() == pconfig.nLayers) {
                        for (int l = 0; l < pconfig.nLayers; l++)
                            pconfig.erBeta.push_back(er["beta"][l].value_or<double>((double)pconfig.beta[l]));
                    } else {
                        for (int l = 0; l < pconfig.nLayers; l++)
                            pconfig.erBeta.push_back(er["beta"][0].value_or<double>((double)pconfig.beta[l]));
                    }
                }
            } else if (load) {
                if (pconfig.erW.size() == 0) {
                    for (int l = 0; l < pconfig.w.size(); l++)
                        pconfig.erW.push_back(pconfig.w[l]);
                }
                if (pconfig.erBeta.size() == 0) {
                    for (int l = 1; l <= pconfig.beta.size(); l++)
                        pconfig.erBeta.push_back(pconfig.beta[l]);
                }
            }

            /* Data */
            pconfig.nSeq = config["dataset"]["n_seq"].value_or<int>((int)pconfig.nSeq);
            pconfig.seqLen = config["dataset"]["seq_len"].value_or<int>((int)pconfig.seqLen);
            pconfig.outputSize = config["dataset"]["output_size"].value_or<int>((int)pconfig.outputSize);
            pconfig.dataMin = config["dataset"]["norm"]["raw_min"].value_or<double>((double)pconfig.dataMin);
            pconfig.dataMax = config["dataset"]["norm"]["raw_max"].value_or<double>((double)pconfig.dataMax);
            // Datapath is set later

            /* Output layer */
            pconfig.outputLayer = layers[0]["type"].value_or<string>((string)pconfig.outputLayer);
            auto softmax_sigma = layers[0]["sm_sigma"];
            if (softmax_sigma) {
                pconfig.smUnit = layers[0]["sm_unit"].value_or<int>(10);
                pconfig.smSigma.clear();
                if (!softmax_sigma || softmax_sigma.as_array()->size() == 0) {
                    pconfig.smSigma.push_back(0.05);
                } else if (softmax_sigma.as_array()->size() == 1 || (softmax_sigma.as_array()->size() != 1 && softmax_sigma.as_array()->size() != pconfig.outputSize)) {
                    pconfig.smSigma.push_back(softmax_sigma[0].value_or<double>(0.05));
                } else {
                    for (int d = 0; d < softmax_sigma.as_array()->size(); d++)
                        pconfig.smSigma.push_back(softmax_sigma[d].value_or<double>(0.05));
                }
            }

            pconfig.minibatchSize = (task == "planning") ? 1 : config["dataset"]["minibatch_size"].value_or<int>((int)pconfig.minibatchSize); // TODO: batch planning
            pconfig.nEpoch = config["training"]["n_epoch"].value_or<int>((int)pconfig.nEpoch);
            pconfig.optimizerAlgorithm = config["training"]["optimizer"]["name"].value_or<string>((string)pconfig.optimizerAlgorithm);
            if (pconfig.optimizerAlgorithm == "adam") {
                pconfig.alpha = config["training"]["optimizer"]["adam"]["alpha"].value_or<double>((double)pconfig.alpha);
                pconfig.beta1 = config["training"]["optimizer"]["adam"]["beta1"].value_or<double>((double)pconfig.beta1);
                pconfig.beta2 = config["training"]["optimizer"]["adam"]["beta2"].value_or<double>((double)pconfig.beta2);
                pconfig.a_eps = config["training"]["optimizer"]["adam"]["eps"].value_or<double>((double)pconfig.a_eps);
            }

            /* Sigma clipping */
            pconfig.sigmaMaxVal = config["network"]["sigma_max"].value_or<double>((double)pconfig.sigmaMaxVal);
            pconfig.sigmaMinVal = config["network"]["sigma_min"].value_or<double>((double)pconfig.sigmaMinVal);

            /* Saved model */
            pconfig.saveInterval = config["training"]["save_interval"].value_or<int>((int)pconfig.saveInterval);
            char* _save_dir_prefix = std::getenv("PVRNN_SAVE_DIR");
            string save_dir_prefix;
            if (_save_dir_prefix != nullptr) save_dir_prefix = _save_dir_prefix;
            else save_dir_prefix = "./"; // fallback
            if (config["training"]["save_directory"])
                pconfig.saveDirectory = std::filesystem::path(save_dir_prefix) / std::filesystem::path("results/") / std::filesystem::path(config["training"]["save_directory"].value_or<string>(""));
            _data_path = config["dataset"]["dataset_path"].value_or<string>((string)_data_path);
            /* Load metadata instead of reloading data if possible */
            if (task != "training")
                pconfig.datasetDirectory = "";
            else
                pconfig.datasetDirectory = _data_path;
            pconfig.datasetDirectory = _data_path;
            pconfig.baseDirectory = cfg_path;
            pconfig.epochToLoad = config["training"]["epoch_to_load"].value_or<int>((int)pconfig.epochToLoad);

            /* General ER/planning settings */
            if (config[ertask]["save_directory"]) pconfig.erSaveDirectory = std::filesystem::path(save_dir_prefix) / std::filesystem::path("results/") / std::filesystem::path(config[ertask]["save_directory"].value_or<string>((string)pconfig.erSaveDirectory));

            pconfig.erOptimizerAlgorithm = config[ertask]["optimizer"]["name"].value_or<string>("adam");
            if (pconfig.erOptimizerAlgorithm == "adam") {
                pconfig.erAlpha = config[ertask]["optimizer"]["adam"]["alpha"].value_or<double>((double)pconfig.erAlpha);
                pconfig.erBeta1 = config[ertask]["optimizer"]["adam"]["beta1"].value_or<double>((double)pconfig.erBeta1);
                pconfig.erBeta2 = config[ertask]["optimizer"]["adam"]["beta2"].value_or<double>((double)pconfig.erBeta2);
                if (task != "training") pconfig.a_eps = config[ertask]["optimizer"]["adam"]["eps"].value_or<double>((double)pconfig.a_eps); // FIXME: ER doesn't have its own epsilon?
            }
            pconfig.nItr = config[ertask]["n_itr"].value_or<int>((int)pconfig.nItr);
            pconfig.nInitItr = config[ertask]["n_init_itr"].value_or<int>((int)pconfig.nInitItr); // Warmup epochs. Only applies to batched ER

            /* Planning overrides ER settings TODO: verify functionality with er vs planning */
            pconfig.erStep = 1; // FIXME: With the static planner this is always 1, but what should it be for other ER??
            pconfig.totalStep = config[ertask]["total_step"].value_or<int>((int)pconfig.totalStep); // Number of steps online ER makes. This controls the length of the output
            pconfig.gWindow = (ertask == "planning") ? 0 : (config[ertask]["grow_window"].value_or<bool>(pconfig.gWindow == 1 ? 1 : 0) ? 1 : 0);
            pconfig.erSeqLen = (ertask == "planning") ? pconfig.seqLen : config[ertask]["seq_len"].value_or<int>((int)pconfig.erSeqLen); // Length of the output from ER. With the static planner this is always seqLen
            pconfig.window = (ertask == "planning") ? pconfig.seqLen : config[ertask]["window_size"].value_or<int>((int)pconfig.window); // Static planner always has maxed out window
            pconfig.predStep = (ertask == "planning") ? 0 : config[ertask]["pred_step"].value_or<int>((int)pconfig.predStep); // How many steps of prior generation to use. Doesn't apply to static planner

            pconfig.erDataDirectory = config[ertask]["dataset_path"].value_or<string>((string)pconfig.erDataDirectory);
            if (task != "training" && task != "testing" && pconfig.erDataDirectory != "")
                _data_path = pconfig.erDataDirectory;
            pconfig.erBaseDirectory = cfg_path;

            mask_drange[0] = config["planning"]["goal_mask"]["start"].value_or<int>((int)mask_drange[0]);
            mask_drange[1] = config["planning"]["goal_mask"]["end"].value_or<int>((int)mask_drange[1]);

            if (load) {
                /* Fallback defaults */
                if (pconfig.a_eps == -1.0) pconfig.a_eps = (task == "training") ? pconfig.alpha / 10.0 : pconfig.erAlpha / 10.0;
                if (pconfig.nInitItr == -1) pconfig.nInitItr = pconfig.nItr;
                if (pconfig.beta1 == -1.0) pconfig.beta1 = 0.9;
                if (pconfig.beta2 == -1.0) pconfig.beta1 = 0.999;
                if (pconfig.erBeta1 == -1.0) pconfig.erBeta1 = 0.9;
                if (pconfig.erBeta2 == -1.0) pconfig.erBeta1 = 0.999;

                if (mask_drange[0] != -1 && mask_drange[1] != -1) {
                    /* Prepare mask */
                    mask_drange[0] = std::max(mask_drange[0], 0);
                    mask_drange[1] = std::min(mask_drange[1], pconfig.outputSize);
                    goal_dmask = (float *)calloc(pconfig.outputSize, sizeof(float)); if (goal_dmask == NULL) abort();
                    for (int d = mask_drange[0]; d <= mask_drange[1]; d++) goal_dmask[d] = 1.0f;
                    u_dmask = (float *)checkedMalloc(sizeof(float)*pconfig.outputSize);
                    for (int d = 0; d < pconfig.outputSize; d++) u_dmask[d] = 1.0f;
                }
                model->config->importConfig(pconfig);
            } else if (!cfg_str.empty()) {
                cout << "importTOMLConfig: Loading parsed TOML string with " << config.size() << " parameters" << endl;
                model->config->importConfig(pconfig); // load parsed partial config
            }
            return true;
        }

        inline int n_seq() noexcept { return _n_seq; }
        inline int dims() noexcept { return _dims; }
        inline int max_timesteps() noexcept { return _max_timesteps; }
        inline int softmax_quant() noexcept { return _softmax_quant; }
        inline int n_layers() noexcept { return _n_layers; }
        inline int max_epochs() noexcept { return _max_epochs; }
        inline int save_epochs() noexcept { return _save_epochs; }
        inline int current_epoch() noexcept { return _current_epoch; }
        inline int postdiction_window_length() noexcept { return _postdiction_window_length; }
        inline int planning_window_length() noexcept { return _planning_window_length; }
        inline int window_length() noexcept { return _postdiction_window_length + _planning_window_length; }
        inline int max_window_length() noexcept { return _max_window_length; }
        inline int* d_neurons() noexcept { return _d_neurons; }
        inline int d_neurons(int l) noexcept { return _d_neurons[l]; }
        inline int* z_units() noexcept { return _z_units; }
        inline int z_units(int l) noexcept { return _z_units[l]; }
        inline double* meta_prior() noexcept { return _meta_prior; }
        inline double meta_prior(int l) noexcept { return _meta_prior[l]; }
        inline string training_path() noexcept { return _training_path; }
        inline string output_dir() noexcept { return _output_dir; }
        inline string data_path() noexcept { return _data_path; }
        inline string base_dir() noexcept { return _base_dir; }
};
