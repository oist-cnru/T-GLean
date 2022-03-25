/*
 * This is for the GLean ctypes interface
 */

#include "glean.h"

GPvrnn* GPvrnn::pvrnn = new GPvrnn();

extern "C" {

    GPvrnn* getInstance() {
        return GPvrnn::getInstance();
    }

    void newModel(GPvrnn* instance, const char *config_file, const char *task, const char *config_type, int rng_seed) {
        instance->newModel(config_file, task, rng_seed, config_type);
    }

    void importGConfig(GPvrnn* instance, const char* cfg_path, const char* task) {
        instance->importGConfig(cfg_path, task);
    }

    void importTOMLConfig(GPvrnn* instance, const char* cfg_path, const char* task) {
        instance->importTOMLConfig(cfg_path, task);
    }

    void reconfigureTOML(GPvrnn* instance, const char* cfg_str, const char* task) {
        instance->importTOMLConfig(std::string(), task, cfg_str, false);
    }

    void modelInitialize(GPvrnn* instance) {
        instance->modelInitialize();
    }

    void trainInitialize(GPvrnn* instance, int start_epoch) {
        instance->trainInitialize(start_epoch);
    }

    void train(GPvrnn* instance) {
        instance->train();
    }

    void trainBackground(GPvrnn* instance, int background_epochs, double &rec_loss, double &reg_loss, bool greedy_train) {
        instance->train(background_epochs, rec_loss, reg_loss, greedy_train);
    }

    void testInitialize(GPvrnn* instance, int epoch) {
        instance->testInitialize(epoch);
    }

    /* Run and save prior and posterior generation */
    void test(GPvrnn* instance) {
        instance->test();
    }

    /* For unit testing */
    void postGenAndSave(GPvrnn* instance) {
        instance->postGenAndSave();
    }

    /* For unit testing */
    void priorGenAndSave(GPvrnn* instance, int post_steps) {
        instance->priGenAndSave(post_steps);
    }

    void batchErInitialize(GPvrnn* instance, int epoch) {
        instance->batchErInitialize(epoch);
    }

    void onlineErInitialize(GPvrnn* instance, int epoch) {
        instance->onlineErInitialize(epoch);
    }

    void planInitialize(GPvrnn* instance, int epoch) {
        instance->planInitialize(epoch);
    }

    void batchErrorRegression(GPvrnn* instance, double &rec_loss, double &reg_loss, const char *sub_dir, const char *output_file) {
        if (sub_dir[0] == '\0') instance->batchErrorRegression(rec_loss, reg_loss);
        else instance->batchErrorRegression(rec_loss, reg_loss, std::filesystem::path(sub_dir) / std::filesystem::path(output_file));
    }

    void onlineErrorRegression(GPvrnn* instance, float *input, float *output, double &rec_loss, double &reg_loss, const char *sub_dir, const char *output_file) {
        if (sub_dir[0] == '\0') instance->onlineErrorRegression(input, output, NULL, rec_loss, reg_loss);
        else instance->onlineErrorRegression(input, output, NULL, rec_loss, reg_loss, std::filesystem::path(sub_dir) / std::filesystem::path(output_file));
    }

    void maskedOnlineErrorRegression(GPvrnn* instance, float *input, float *output, float *mask, double &rec_loss, double &reg_loss, const char *sub_dir, const char *output_file) {
        if (sub_dir[0] == '\0') instance->onlineErrorRegression(input, output, mask, rec_loss, reg_loss);
        else instance->onlineErrorRegression(input, output, mask, rec_loss, reg_loss, std::filesystem::path(sub_dir) / std::filesystem::path(output_file));
    }

    void plan(GPvrnn* instance, float *mask, double &rec_loss, double &reg_loss, const char *sub_dir, const char *output_file) {
        if (sub_dir[0] == '\0') instance->plan(mask, rec_loss, reg_loss);
        else instance->plan(mask, rec_loss, reg_loss, std::filesystem::path(sub_dir) / std::filesystem::path(output_file));
    }

    void dynamicPlan(GPvrnn* instance, float *input, float *mask, bool dynamic, double &rec_loss, double &reg_loss, const char *sub_dir, const char *output_file) {
        if (sub_dir[0] == '\0') instance->plan(input, mask, rec_loss, reg_loss, "", dynamic);
        else instance->plan(input, mask, rec_loss, reg_loss, std::filesystem::path(sub_dir) / std::filesystem::path(output_file), dynamic);
    }

    void priorGeneration(GPvrnn* instance, float *output) {
        instance->priorGeneration(output);
    }

    int n_seq(GPvrnn *instance) {
        return instance->n_seq();
    }

    int dims(GPvrnn *instance) {
        return instance->dims();
    }

    int softmax_quant(GPvrnn *instance) {
        return instance->softmax_quant();
    }

    int max_timesteps(GPvrnn *instance) {
        return instance->max_timesteps();
    }

    int n_layers(GPvrnn *instance) {
        return instance->n_layers();
    }

    int window_length(GPvrnn *instance) {
        return instance->window_length();
    }

    int postdiction_window_length(GPvrnn *instance) {
        return instance->postdiction_window_length();
    }

    int planning_window_length(GPvrnn *instance) {
        return instance->planning_window_length();
    }

    int save_epochs(GPvrnn *instance) {
        return instance->save_epochs();
    }

    int max_epochs(GPvrnn *instance) {
        return instance->max_epochs();
    }

    int current_epoch(GPvrnn *instance) {
        return instance->current_epoch();
    }

    void getDataPath(GPvrnn* instance, char *path) {
        instance->data_path().copy(path, PATH_MAX);
    }

    void getOutputDir(GPvrnn* instance, char *path) {
        instance->output_dir().copy(path, PATH_MAX);
    }

    void getTrainingPath(GPvrnn* instance, char *path) {
        instance->training_path().copy(path, PATH_MAX);
    }

    void getBaseDir(GPvrnn* instance, char *path) {
        instance->base_dir().copy(path, PATH_MAX);
    }

    void getMetaPrior(GPvrnn *instance, double* meta_prior) {
        memcpy(meta_prior, instance->meta_prior(), sizeof(double)*(n_layers(instance)-1));
    }

    void getDNeurons(GPvrnn *instance, int* d_neurons) {
        memcpy(d_neurons, instance->d_neurons(), sizeof(int)*(n_layers(instance)-1));
    }

    void getZUnits(GPvrnn *instance, int* z_units) {
        memcpy(z_units, instance->z_units(), sizeof(int)*(n_layers(instance)-1));
    }

    void getFullErRegErr(GPvrnn *instance, int layer, int step, double *kld) {
        instance->getFullErRegErr(layer, step, kld);
    }

    void getFullErRecErr(GPvrnn *instance, int step, double *rec) {
        instance->getFullErRecErr(step, rec);
    }

    void getErOutput(GPvrnn* instance, float *output) {
        instance->getErOutput(output);
    }

    void getPosteriorA(GPvrnn* instance, float *Amyu, float *Asigma) {
        instance->getPosteriorA(Amyu, Asigma);
    }

    void getPriorMyuSigma(GPvrnn* instance, float *myu, float *sigma) {
        instance->getPriorMyuSigma(myu, sigma);
    }

    void getPosteriorMyuSigma(GPvrnn* instance, float *myu, float *sigma) {
        instance->getPosteriorMyuSigma(myu, sigma);
    }

    void setErData(GPvrnn* instance, float *input) {
        instance->setErData(input, instance->max_timesteps(), instance->dims());
    }

    void setData(GPvrnn* instance, float *input) {
        instance->setData(input);
    }
}
