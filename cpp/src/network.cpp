#include <iostream>
#include <cmath>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <iterator>

#include "output_layer/sm_layer.h"
#include "output_layer/fc_layer.h"
#include "network.h"


PvrnnNetwork::PvrnnNetwork(bool wKLD, int rng_seed) : wKLD(wKLD) {  // FIXME: refactor so const member and config are compatible
    #ifdef NDEBUG
    set_terminate(catchall_exceptions);
    #endif
    config = new networkConfig();
    this->engine = (rng_seed >= 0) ? default_random_engine(rng_seed) : default_random_engine(random_device()());
    if (rng_seed >= 0) { srand(rng_seed+1); }  // set the random generator for Eigen
}

PvrnnNetwork::~PvrnnNetwork() {
    delete config;
    if (optimizer != nullptr) delete optimizer;
    if (outputLayer != nullptr) delete outputLayer;
    if (data != nullptr) delete data;
    for (size_t i = 0; i < layers.size(); i++) {
        delete layers[i];
    }
}

void PvrnnNetwork::initializeNetwork(bool zeroInit) {
    string datasetDirectory = (!config->datasetDirectory.empty()) ? config->trainConfigPath.parent_path() / config->datasetDirectory : "";
    data = new Data(true, config->dataMin, config->dataMax, config->normMin, config->normMax,
                    config->outputLayer, config->smUnit, config->smSigma,
                    config->nSeq, config->seqLen, config->minibatchSize, datasetDirectory);

    this->dSize            = config->dSize;
    this->zSize            = config->zSize;
    this->nLayers          = dSize.size();
    this->outputSize       = config->outputSize;
    this->tau              = config->tau;
    this->w                = config->w;
    this->beta             = config->beta;
    this->seqLen           = config->seqLen;        //FIXME: not needed, access data->seqLen
    this->nSeq             = config->nSeq;          //FIXME: not needed, access data->nSeq
    this->minibatchSize    = config->minibatchSize; //FIXME: not needed, no access
    this->smUnit           = config->smUnit;        //FIXME: not needed, no access
    this->smSigma          = config->smSigma;       //FIXME: not needed, no access
    this->nEpoch           = config->nEpoch;
    this->saveDirectory    = config->saveDirectory;
    this->saveInterval     = config->saveInterval;
    this->epoch            = 0;

    this->nMinibatch       = nSeq / minibatchSize;

    if (this->nLayers != (int)zSize.size()) { throw invalid_argument("zSize is not correct"); }
    if (this->nLayers != (int) beta.size()) { throw invalid_argument("beta is not correct"); }
    if (this->nLayers != (int)    w.size()) { throw invalid_argument("w is not correct"); }
    if (this->nLayers != (int)  tau.size()) { throw invalid_argument("tau is not correct"); }
    if (this->minibatchSize > this->nSeq)   { throw invalid_argument("minibatchSize is larger than the training dataset size"); }
    if (this->nSeq % this->minibatchSize != 0)   { throw invalid_argument("minibatchSize is not a divisor of nSeq"); }  // will create segfaults if not a divisor.

    // initialize network layers
    for ( int i = 0; i < nLayers; i++ ){
        int inputSize = i == 0 ? 0 : dSize[i-1];
        PvrnnLayer *layer = new PvrnnLayer(dSize[i], zSize[i], inputSize, tau[i], w[i], beta[i],
                                           seqLen, nSeq, minibatchSize, sigmaMinVal, sigmaMaxVal,
                                           &engine, zeroInit);
        layers.push_back( layer );
    }
    if ( data->outputLayer == "sm"  ){
        outputLayer = new SoftmaxLayer(config->dSize.back(), config->outputSize, config->smUnit,
                                        config->seqLen, config->nSeq, config->minibatchSize, &(data->erxSeq), &engine, zeroInit);
    }
    else if ( data->outputLayer == "fc" ){
        outputLayer = new FCLayer(config->dSize.back(), config->outputSize, config->seqLen, config->nSeq,
                                   config->minibatchSize, &engine, zeroInit);
    } else {
        throw invalid_argument("Unknown outputLayer type '" + data->outputLayer + "'");
    }
    outputLayer->data = data; // FIXME: is this really necessary??
    // collect pointers of the parameters
    registerParameters();
    // instantiate optimizer here
    optimizer = new Optimizer(config->optimizerAlgorithm, config->alpha, config->beta1, config->beta2);
    optimizer->registerParameters(weight, weightGrad, bias, biasGrad, APtr, AGradPtr, nMinibatch); // APtr
}


PvrnnNetwork::PvrnnNetwork(Data* data_, int rng_seed, bool wKLD):
    nSeq(data_->nSeq), seqLen(data_->seqLen), outputSize(data_->rawDim), minibatchSize(data_->minibatchSize),
    nLayers(0),
    smUnit(data_->smUnit), smSigma(data_->smSigma), //FIXME: none of this should be needed
    epoch(0), data(data_), wKLD(wKLD) {
    // config = new networkConfig();
    engine = (rng_seed >= 0) ? default_random_engine(rng_seed) : default_random_engine(random_device()());
    if (rng_seed >= 0) { srand(rng_seed+1); }  // set the random generator for Eigen (not working, should be effective in Eigen 3.4)

}

void PvrnnNetwork::add_layer(int d_size, int z_size, int input_size, double tau, double w, double beta,
                             double sigma_min, double sigma_max, bool zero_init) {
    PvrnnLayer *layer = new PvrnnLayer(d_size, z_size, input_size, tau, w, beta,
                                       data->seqLen, data->nSeq, data->minibatchSize, sigma_min, sigma_max,
                                       &engine, zero_init);
    layers.push_back(layer);
    nLayers += 1;

    // necessary, but don't know exactly why yet # FIXME: if not there, produces segfaults
    dSize.push_back(d_size);
    zSize.push_back(z_size);
    this->tau.push_back(tau);
    this->w.push_back(w);
    this->beta.push_back(beta);
}

// To do after the all other layers have been created.
void PvrnnNetwork::add_output_layer(bool zero_init) {
    int inputSize = layers.back()->dSize;
    if ( data->outputLayer == "sm" ){
        outputLayer = new SoftmaxLayer(inputSize, data->rawDim, data->smUnit,
                                        data->seqLen, data->nSeq, data->minibatchSize, &(data->erxSeq), &engine, zero_init);
    }
    else if ( data->outputLayer == "fc" ){
        outputLayer = new FCLayer(inputSize, data->rawDim, data->seqLen, data->nSeq, data->minibatchSize, &engine, zero_init);
    } else {
        throw invalid_argument("unknown output layer type \"" + data->outputLayer + "\".");
    }
}

void PvrnnNetwork::add_optimizer(string algorithm, double alpha, double beta1, double beta2, int n_minibatch) {
    registerParameters();  // collect pointers of the parameters
    optimizer = new Optimizer(algorithm, alpha, beta1, beta2);
    optimizer->registerParameters(weight, weightGrad, bias, biasGrad, APtr, AGradPtr, n_minibatch);
}

void PvrnnNetwork::set_layer_w(vector<double> ws) {
    if (ws.size() != layers.size()) { throw invalid_argument("The number of w values (" + to_string(ws.size()) + ") is not the " +
                                                             "same as the number of layers (" + to_string(layers.size()) + ")"); }
    for (size_t i = 0; i < ws.size(); i++) { layers[i]->w = ws[i]; }
}

void PvrnnNetwork::randomizeA(){
    for (const auto& layer: layers)
        layer->randomizeA();
}

void PvrnnNetwork::initializeSequence(){
    for ( const auto& layer: layers ){
        layer->initializeSequence();
    }
    outputLayer->initializeSequence();
}

void PvrnnNetwork::initializeState(){
    for ( const auto& layer: layers ){
        layer->initializeState();
    }
}

void PvrnnNetwork::forwardStep(int seqInd, int steps){
    for (int s = 0; s < steps; s++) {
        layers[0]->topLayerForwardStep(seqInd);
        for ( int i = 1; i < nLayers; i++ ){
            layers[i]->layerForwardStep(layers[i-1]->d, seqInd);
        }
        outputLayer->computeOutput(layers[nLayers-1]->d, seqInd);
    }
}

void PvrnnNetwork::priorStep(int seqInd, int steps){
    for (int s = 0; s < steps; s++) {
        layers[0]->topLayerPriorStep(seqInd);
        for ( int i = 1; i < nLayers; i++ ){
            layers[i]->layerPriorStep(layers[i-1]->d, seqInd);
        }
        outputLayer->computeOutput(layers[nLayers-1]->d, seqInd);
    }
}

// note: minibatch might not be of minibatchSize (e.g. when called from postGenAndSave())
void PvrnnNetwork::forwardSequence(const vector<int>& minibatch){
    for ( int seqInd: minibatch ){
        initializeState();
        forwardStep(seqInd, data->seqLen);
    }
}

void PvrnnNetwork::postGenAndSave(string directory){
    /*
    This function generates posterior reconstruction for the training sequences
    and save the generated sequences.
    This function should be used after loading a parameter/
    */
    initializeSequence();
    vector<int> whole_batch(data->nSeq);
    iota(whole_batch.begin(), whole_batch.end(), 0);  // fill with range of increasing values
    forwardSequence( whole_batch );
    saveSequences( directory , true );
}

void PvrnnNetwork::priGenAndSave(int postStep, int totalStep, string directory){
    /*
    This function generates with prior after generating posterior reconstruction for the time steps
    specified by "postStep". The total length of the generated sequence is speficied by "totalStep".
    */
    initializeSequence();
    for ( int seqInd = 0; seqInd < data->nSeq; seqInd++ ){
        initializeState();
        forwardStep(seqInd, postStep);
        priorStep(seqInd, totalStep-postStep);
    }
    saveSequences( directory , true );
}

double PvrnnNetwork::computeKldSeq(const vector<int>& minibatch){
    double _kld = 0.0;
    double *_w = &w[0];
    for ( const auto& layer: layers ){
        if (wKLD) {
            _kld += *(_w) * layer->computeKldSeq( minibatch );
            _w++;
        } else {
            _kld += layer->computeKldSeq( minibatch );
        }
    }
    return _kld;
}

void PvrnnNetwork::zeroGradient(){
    for ( const auto& layer: layers ){
        layer->zeroGradient();
    }
    outputLayer->zeroGradient();
}

void PvrnnNetwork::initBPTT(){
    for ( const auto& layer: layers ){
        layer->initBPTT();
    }
}

void PvrnnNetwork::computeGradient(int seqInd, int _t){
    // compute the gradients in the output layer
    outputLayer->computeGradient(data->trainData, layers.back()->dSeq, seqInd, _t);

    // compute the gradients in the rnn layers
    if ( nLayers == 1 ){
        layers[0]->topLayerComputeGradient(outputLayer->wdLdo, seqInd, _t);
    }
    else {
        layers.back()->layerComputeGradient(layers[nLayers-2]->dSeq, outputLayer->wdLdo, seqInd, _t);
        for ( int i = nLayers - 2; i > 0; i-- ){
            layers[i]->layerComputeGradient(layers[i-1]->dSeq, layers[i+1]->wdLdh, seqInd, _t);
        }
        layers[0]->topLayerComputeGradient(layers[1]->wdLdh, seqInd, _t);
    }
}

void PvrnnNetwork::BPTT(const vector<int>& minibatch){
    zeroGradient();
    for ( int seqInd: minibatch ){
        initBPTT();
        for ( int _t = data->seqLen - 1; _t > -1; _t-- ){
            computeGradient(seqInd, _t);
        }
    }
}

void PvrnnNetwork::registerParameters(){
    /*
    This function creates containers holding pointers of all network paraemters, which are used in the optimizer class
    */
    weight.clear();
    weightGrad.clear();
    bias.clear();
    biasGrad.clear();
    APtr.clear();
    AGradPtr.clear();

    // collect all pointers of layer parameters
    for ( int i = 0; i < nLayers; i++ ){
        weight.push_back( &layers[i]->Wdh );
        weightGrad.push_back( &layers[i]->dLdWdh );
        weight.push_back( &layers[i]->Wzh );
        weightGrad.push_back( &layers[i]->dLdWzh );
        weight.push_back( &layers[i]->Wdmp );
        weightGrad.push_back( &layers[i]->dLdWdmp );
        weight.push_back( &layers[i]->Wdsp );
        weightGrad.push_back( &layers[i]->dLdWdsp );

        if ( i > 1){
            weight.push_back( &layers[i]->Whdh );
            weightGrad.push_back( &layers[i]->dLdWhdh );
        }

        bias.push_back( &layers[i]->Bh );
        biasGrad.push_back( &layers[i]->dLdBh );
        bias.push_back( &layers[i]->Bmp );
        biasGrad.push_back( &layers[i]->dLdBmp );
        bias.push_back( &layers[i]->Bsp );
        biasGrad.push_back( &layers[i]->dLdBsp );

        APtr.push_back( &layers[i]->AMuSeq );
        APtr.push_back( &layers[i]->ASigmaSeq );
        AGradPtr.push_back( &layers[i]->dLdAMuSeq );
        AGradPtr.push_back( &layers[i]->dLdASigmaSeq );
    }

    // add pointers of the output layer parameters
    weight.push_back( &outputLayer->W );
    weightGrad.push_back( &outputLayer->dLdW );
    bias.push_back( &outputLayer->b );
    biasGrad.push_back( &outputLayer->dLdb );
}

void PvrnnNetwork::initTraining(int startEpoch) {
    epoch = startEpoch;
    if (epoch == 0) {
        std::filesystem::create_directories(saveDirectory + "/parameters");
        std::filesystem::create_directories(saveDirectory + "/sequences");
        std::filesystem::create_directories(saveDirectory + "/learning");
        // clearing log files
        ofstream _recErr_stream( saveDirectory + "/learning/recErr.txt", ios::trunc );
        _recErr_stream.close();
        for ( int i = 0; i < nLayers; i++ ){
            ofstream _kld_stream(saveDirectory + "/learning/layer" + to_string(nLayers-i) + "_kld.txt", ios::trunc);
            _kld_stream.close();
        }
        // write metadata about the loaded dataset
        ofstream _metadata_stream(saveDirectory + "/learning/metadata.toml", ios::trunc);
        _metadata_stream << toml::table{{
                            {"data", toml::table{{
                                {"nSeq", data->nSeq}, {"seqLen", data->seqLen}, {"rawDim", data->rawDim},
                                {"rawMin", data->rawMin}, {"rawMax", data->rawMax},
                                {"normMin", data->normMin}, {"normMax", data->normMax}
                                }}},
                            {"save", toml::table{{
                                {"rowMajorW", saveRowMajorW}
                            }}}
                            }};
        _metadata_stream.close();
    } else { // load previous parameters
        try { // check the loaded data against the saved metadata if available, warn if mismatch
            metadata = toml::parse_file(saveDirectory + "/learning/metadata.toml");
            if (data->rawMin != metadata["data"]["rawMin"].value_or<double>(-1.0) ||
                data->rawMax != metadata["data"]["rawMax"].value_or<double>(-1.0) ||
                data->normMin != metadata["data"]["normMin"].value_or<double>(-1.0) ||
                data->normMax != metadata["data"]["normMax"].value_or<double>(-1.0)) {
                    cerr << "initTraining: Warning - loaded data doesn't match saved metadata" << endl;
                    cerr << "              data->rawMin  = " << data->rawMin  << "  metadata->rawMin  = " << metadata["data"]["rawMin"].value_or<double>(-1.0) << endl;
                    cerr << "              data->rawMax  = " << data->rawMax  << "  metadata->rawMax  = " << metadata["data"]["rawMax"].value_or<double>(-1.0) << endl;
                    cerr << "              data->normMin = " << data->normMin << "  metadata->normMin = " << metadata["data"]["normMin"].value_or<double>(-1.0) << endl;
                    cerr << "              data->normMax = " << data->normMax << "  metadata->normMax = " << metadata["data"]["normMax"].value_or<double>(-1.0) << endl;
            }
        } catch (const toml::parse_error &e) {
            cout << "initTraining: Training metadata couldn't be parsed, skipping check" << endl;
        }
        string parameterDirectory = saveDirectory + "/parameters/epo_" + to_string(epoch);
        saveRowMajorW = metadata["save"]["rowMajorW"].value_or<bool>(false);
        loadParameters(parameterDirectory, saveRowMajorW);
    }
    // clearing log data structures
    _recErrLog.clear();
    _kldLog.clear();
    _kldLog.resize(nLayers);
}

void PvrnnNetwork::trainOneEpoch(double &epochErr, double &epochKld, bool optimize) {
    initializeSequence();
    vector<vector<int>> minibatches = data->minibatches();
    for ( const auto& minibatch: minibatches ){
        forwardSequence( minibatch );
        double minibatchErr = outputLayer->computeErr(data->trainData, minibatch);
        double minibatchKld = computeKldSeq(minibatch);
        epochErr += minibatchErr;
        epochKld += minibatchKld;
        if (optimize) {
            BPTT( minibatch );
            optimizer->computeDelta( minibatch );
            optimizer->updateParameters( minibatch );
        }
        // cout << "   minibatch: " << minibatchInd << " RecErr: " << minibatchErr <<  " KLD: " << minibatchKld << "\n";
    }

    for ( int i = 0; i < nLayers; i++ ){
        double _kld = 0.0;
        for ( const auto& minibatch: minibatches ){
            if (wKLD)
                _kld += w[i] * layers[i]->computeKldSeq( minibatch );
            else
                _kld += layers[i]->computeKldSeq( minibatch );
        }
        _kldLog[i].push_back( _kld );
    }
    _recErrLog.push_back( epochErr );
    epoch++;

    // synchronizing the epochs of the layers
    for ( const auto& layer: layers){ layer->epoch = epoch; };
}

void PvrnnNetwork::trainOneEpoch() {
    chrono::system_clock::time_point start, end;
    start = chrono::system_clock::now();

    double epochErr = 0.0;
    double epochKld = 0.0;

    trainOneEpoch(epochErr, epochKld);

    end = chrono::system_clock::now();

    // cout << left;  // left-align values
    cout << right << "epoch#" << setw(6) << epoch << left
         << "  recErr: " << setw(11) << epochErr
         << "  kld: "    << setw(11) << epochKld
         << "  microsec/epoch: " << chrono::duration_cast<chrono::microseconds>( end - start ).count() << "\n";

    // save learning curve, generated sequences, and parameters
    if ((saveInterval > 0) && (epoch % saveInterval == 0)) { saveTraining(); }

    #if !defined(NDEBUG) || defined(NAN_INF_STRICT)
    if (isinf(epochErr)) { throw runtime_error("reconstruction error is infinite"); }
    #endif
}

double PvrnnNetwork::getKldFromLog(int layer, int step) {
    if (step == -1) return _kldLog[layer].back();
    else return _kldLog[layer][step];
}

double PvrnnNetwork::getRecErrFromLog(int step) {
    if (step == -1) return _recErrLog.back();
    else return _recErrLog[step];
}

void PvrnnNetwork::getErKldFromLog(int layer, int epoch, double *kld) {
    // Returns full KLD values without length normalization or weighting
    if (epoch == -1) copy(_kldFullLog[layer].back().begin(), _kldFullLog[layer].back().end(), kld);
    else copy(_kldFullLog[layer][epoch].begin(), _kldFullLog[layer][epoch].end(), kld);
}

void PvrnnNetwork::getErRecErrFromLog(int epoch, double *rec_err) {
    // Returns full rec err values without length normalization
    if (epoch == -1) copy(_recErrFullLog.back().begin(), _recErrFullLog.back().end(), rec_err);
    else copy(_recErrFullLog[epoch].begin(), _recErrFullLog[epoch].end(), rec_err);
}

void PvrnnNetwork::getErSequence(float *seq) {
    size_t winlen = gWindow ? min(onlineErStep, window) : window;
    for (size_t i = 0; i < winlen; i++)
        Map<VectorXf>(i*outputSize+seq, (*(outputLayer->erOutSeq))[i].size()) = (*(outputLayer->erOutSeq))[i];
}

void PvrnnNetwork::getErASequence(float *AMu, float *ASigma) {
    size_t winlen = gWindow ? min(onlineErStep, window) : window;
    size_t offset = 0;
    for (size_t t = 0; t < winlen; t++) {
        for (const auto& layer: layers) {
            Map<VectorXf>(AMu+offset, layer->erAMuSeq[t].size()) = layer->erAMuSeq[t];
            Map<VectorXf>(ASigma+offset, layer->erASigmaSeq[t].size()) = layer->erASigmaSeq[t];
            offset += layer->zSize;
        }
    }
}

void PvrnnNetwork::getErPriorMuSigmaSequence(float *mu, float *sigma) {
    size_t offset = 0;
    for (size_t t = 0; t < layers[0]->erMupSeq.size(); t++) {
        for (const auto& layer: layers) {
            Map<VectorXf>(mu+offset, layer->zSize) = layer->erMupSeq[t];
            Map<VectorXf>(sigma+offset, layer->zSize) = layer->erSigmapSeq[t];
            offset += layer->zSize;
        }
    }
}

void PvrnnNetwork::getErPosteriorMuSigmaSequence(float *mu, float *sigma) {
    size_t offset = 0;
    for (size_t t = 0; t < layers[0]->erMuqSeq.size(); t++) {
        for (const auto& layer: layers) {
            Map<VectorXf>(mu+offset, layer->zSize) = layer->erMuqSeq[t];
            Map<VectorXf>(sigma+offset, layer->zSize) = layer->erSigmaqSeq[t];
            offset += layer->zSize;
        }
    }
}

void PvrnnNetwork::train(){
    cout << "\n START TRAINING \n\n";
    cout << "results will be saved in "
         << filesystem::  relative(saveDirectory)
         << endl << endl;

    initTraining();

    for (int epo = 0; epo < nEpoch; epo++){
        trainOneEpoch();
    }
 }

/** save the training information (parameters, sequences, error and kld logs).
 *  Remark: clears the error and kld logs.
 */
void PvrnnNetwork::saveTraining() {
    if (epoch != last_saved_epoch) {  // do not save twice the same epoch, especially with empty logs.
        last_saved_epoch = epoch;
        saveParameters(saveDirectory + "/parameters/epo_" + to_string(epoch));
        saveSequences (saveDirectory + "/sequences/epo_"  + to_string(epoch));
        ofstream _epoIdx_stream(saveDirectory + "/learning/lastEpo.txt", ios::trunc);
        _epoIdx_stream << to_string(epoch);
        _epoIdx_stream.close();

        // save rec. err. and kld
        if (_recErrLog.size() > 0) {
            ofstream _recErr_stream(saveDirectory + "/learning/recErr.txt", ios::app);
            copy(_recErrLog.begin(), _recErrLog.end(), std::ostream_iterator<double>(_recErr_stream, "\n"));
            _recErr_stream.close();
            _recErrLog.clear();
        }

        if (_kldLog.size() > 0) {
            for (int i = 0; i < nLayers; i++) {
                if (_kldLog[i].size() > 0) {
                    ofstream _kld_stream(saveDirectory + "/learning/layer" + to_string(nLayers - i) + "_kld.txt", ios::app);
                    copy(_kldLog[i].begin(), _kldLog[i].end(), std::ostream_iterator<double>(_kld_stream, "\n"));
                    _kld_stream.close();
                    _kldLog[i].clear();
                }
            }
        }
    }
}

void PvrnnNetwork::testInitialize(int epoch) {
    this->epoch = (epoch == -1) ? config->epochToLoad() : epoch;
    string parameterDirectory = saveDirectory + "/parameters/epo_" + to_string(this->epoch);
    loadMetadata(saveDirectory + "/learning"); // get saved data information, fallback to data if available
    loadParameters(parameterDirectory, saveRowMajorW);
}



    /* Error Regression */

void PvrnnNetwork::_erInitialize(int epoch) {
    // General ER initialization
    this->window   = config->window;
    this->erSeqLen = config->erSeqLen;
    this->nInitItr = config->nInitItr;
    this->nItr     = config->nItr;
    if (epoch == -1) epoch = config->epochToLoad();

    string saveDirectory = config->saveDirectory;
    string parameterDirectory = saveDirectory + "/parameters/epo_" + to_string(epoch);
    loadMetadata(saveDirectory + "/learning"); // get saved data information, fallback to data if available
    loadParameters(parameterDirectory, saveRowMajorW);

    if (!config->erSaveDirectory.empty())
        std::filesystem::create_directories(config->erSaveDirectory);

    for ( int i = 0; i < nLayers; i++ ){
        layers[i]->erInitialize(config->window, config->erW[i], config->erBeta[i], config->gWindow, config->erRandomInitA);
    }
    outputLayer->erInitialize(config->window);

    optimizer->erInitialize( config->erOptimizerAlgorithm, config->erAlpha, config->erBeta1, config->erBeta2 );

    _recErrLog.clear();
    _kldLog.clear();
    _kldLog.resize(nLayers);
    _recErrFullLog.clear();
    _kldFullLog.clear();
    _kldFullLog.resize(nLayers);
}

void PvrnnNetwork::erInitialize(int epoch) {
    this->gWindow = false; // Disable growing window for batch ER
    _erInitialize(epoch);
}

void PvrnnNetwork::onlinePlanInitialize(int epoch) {
    this->gWindow = false; // Planner handles window
    _erInitialize(epoch);
    data->onlineErInitialize( config->window );
    onlineErStep = 0;
}


void PvrnnNetwork::erSlideWindow(){
    for ( const auto& layer: layers){
        layer->erSlideWindow();
    }
}

void PvrnnNetwork::erForwardStep(){
    layers[0]->erTopLayerForwardStep();
    for ( int i = 1; i < nLayers; i++ ){
        layers[i]->erLayerForwardStep( layers[i-1]->d );
    }
    outputLayer->erComputeOutput( layers.back()->d );

    if ( outputLayer->model == "sm" ){
        data->erInvTransform( outputLayer->x );
    }
}

void PvrnnNetwork::erPriorStep(){
    layers[0]->erTopLayerPriorStep();
    for ( int i = 1; i < nLayers; i++ ){
        layers[i]->erLayerPriorStep( layers[i-1]->d );
    }
    outputLayer->erComputeOutput( layers.back()->d );
    if ( outputLayer->model == "sm" ){
        data->erInvTransform( outputLayer->x );
    }
}

void PvrnnNetwork::erForwardSequence(bool updateLastState) {
    erReset();
    int max_t = gWindow ? min(onlineErStep, window) : window;
    for ( int _t = 0; _t < max_t; _t++ ){
        erForwardStep();
        if ( _t == 0 && updateLastState ){
            erKeepLastState();
        }
    }
}

void PvrnnNetwork::erPrediction(int predSteps, string saveFilepath, string prefix, bool updateLastState){
    if (predSteps > 0) {
        erReset();
        for ( int _t = 0; _t < window; _t++ ){
            erForwardStep();
            if ( _t == 0 && updateLastState ){
                erKeepLastState();
            }
        }
        for ( int _t = 0; _t < predSteps; _t++ ){
            erPriorStep();
        }
        if (!prefix.empty()) erSaveSequences(saveFilepath, prefix, true);
    }
}

void PvrnnNetwork::erKeepLastState(){
    for ( const auto& layer: layers){
        layer->erKeepLastState();
    }
}

void PvrnnNetwork::erReset(){
    for ( const auto& layer: layers){
        layer->erReset();
    }
    outputLayer->erReset();
    data->erReset();
}

void PvrnnNetwork::erSeqReset(){
    for ( const auto& layer: layers){
        layer->erSeqReset();
    }
    outputLayer->erReset();
    data->erReset();
}

void PvrnnNetwork::erInitializeGradient(){
    for ( const auto& layer: layers){
        layer->erInitializeGradient();
    }
}

void PvrnnNetwork::erRegisterParameters(){
    erAPtr.clear();
    erAGradPtr.clear();

    // collect all pointers of layer parameters
    for ( const auto& layer: layers ){
      for ( size_t _t = 0; _t < layer->erAMuSeq.size(); _t++ ){
        erAPtr    .push_back(&layer->erAMuSeq      [_t]);
        erAPtr    .push_back(&layer->erASigmaSeq   [_t]);
        erAGradPtr.push_back(&layer->erdLdAMuSeq   [_t]);
        erAGradPtr.push_back(&layer->erdLdASigmaSeq[_t]);
      }
    }
}

void PvrnnNetwork::erIteration(vectorXf1DContainer erTarget, vectorXf1DContainer mask, string saveFilepath, string prefix) {
    erReset();
    erForwardSequence();
    outputLayer->erComputeErr(erTarget, mask);

    /* BPTT */
    erInitializeGradient();
    for ( int _t = window - 1; _t > -1; _t-- ) {
        /* Compute gradient */
        outputLayer->erComputeGradient(erTarget, mask, _t);
        layers.back()->erComputeGradient(outputLayer->wdLdo, _t);
        for ( int i = nLayers - 2; i > -1; i-- ){
            layers[i]->erComputeGradient(layers[i+1]->wdLdh, _t);
        }
    }
    optimizer->erComputeDelta();
    optimizer->erUpdateParameters();
    if (!prefix.empty()) erSaveSequences(saveFilepath, prefix, false);
}

void PvrnnNetwork::errorRegressionStep(int _t, vectorXf1DContainer mask, string savePath, bool verbose, bool saveIterPredictions) {
    int itrLimit = nItr;
    if (_t == 0) {
        if (verbose) cout << "Initial regression... \n";
        itrLimit = nInitItr;
    } else erSlideWindow();
    erTarget = vectorXf1DContainer{ data->erData->begin() + _t, data->erData->begin() + window + _t };
    if (!savePath.empty()) {
        std::filesystem::create_directories(std::filesystem::path(savePath).parent_path());
        savePath += ".npz";
        std::filesystem::remove(savePath); // remove any previous save
    }
    double _kld;
    string itrSave = "";
    erRegisterParameters();
    optimizer->erRegisterParameters(erAPtr, erAGradPtr);
    for (int itr = 0; itr < itrLimit; itr++) {
        if (!savePath.empty() && (saveIterPredictions || itr==nItr-1)) itrSave = "itr" + to_string( itr );
        erIteration(erTarget, mask, savePath, itrSave);
        _kld = 0.0;
        for (int i = 0; i < nLayers; i++) {
            if (wKLD)
                _kld += w[i] * accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / window;
            else
                _kld += accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / window;
        }
        if (saveIterPredictions || itr == itrLimit-1) {
            erPrediction(erSeqLen - window, savePath, itrSave, (itr == itrLimit-1 ? true : false));
            for (int i = 0; i < nLayers; i++) {
                _kldLog[i].push_back(accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / window);
                if (wKLD) _kldLog[i].back() *= w[i];
                _kldFullLog[i].push_back(layers[i]->erKldSeq);
            }
            _recErrLog.push_back(accumulate(outputLayer->erErrSeq.begin(), outputLayer->erErrSeq.end(), 0.0) / window);
            _recErrFullLog.push_back(outputLayer->erErrSeq);
        }
        if (verbose) {
            cout << "Step: " << _t
            << "  Iteration: " << itr
            << "  RecErr: " << accumulate(outputLayer->erErrSeq.begin(), outputLayer->erErrSeq.end(), 0.0) / window
            << "  KLD: " << _kld << "\n";
        }
    }
}

void PvrnnNetwork::errorRegressionStep(int _t, float *_mask, string savePath, bool verbose, bool saveIterPredictions) {
    vectorXf1DContainer mask;
    if (_mask != nullptr) {
        for (int i = 0; i < erSeqLen; i++) {
            Map<VectorXf> dmask(_mask+(i*outputLayer->outputSize), outputLayer->outputSize);
            mask.push_back(dmask);
        }
    }
    errorRegressionStep(_t, mask, savePath, verbose, saveIterPredictions);
}

void PvrnnNetwork::errorRegression(){
    cout << "\n START ERROR REGRESSION \n\n";

    string erSaveDirectory = "";
    if (!config->erSaveDirectory.empty())
        erSaveDirectory = config->erSaveDirectory;

    for ( int _t = 0; _t < config->erStep; _t++ ){
        cout << "\n STEP " << _t << "\n\n";
        errorRegressionStep( _t,  erSaveDirectory + "/step_" + to_string(_t) );
    }
    cout << "Error regression done \n";
}


    /** Online Error Regression **/

void PvrnnNetwork::online_er_init(int epoch_to_load, int er_window, bool growing_window, int n_itr,
                                  int pred_step, int total_step, string er_save_directory,
                                  vector<double> er_w, vector<double> er_beta, bool er_random_init_A) {
    window            = er_window;
    nItr              = n_itr;
    onlineErPredStep  = pred_step;
    onlineErTotalStep = total_step;
    gWindow           = growing_window;

    // loading learning
    loadMetadata(saveDirectory + "/learning"); // get saved data information, fallback to data if available
    loadParameters(saveDirectory + "/parameters/epo_" + to_string(epoch_to_load), saveRowMajorW);
    // creating save dir for er
    if (!er_save_directory.empty()) { filesystem::create_directories(er_save_directory); }

    for ( int i = 0; i < nLayers; i++ ){
        layers[i]->erInitialize(er_window, er_w[i], er_beta[i], er_window, er_random_init_A);
    }
    outputLayer->erInitialize(er_window);
    data->onlineErInitialize(er_window);
}

void PvrnnNetwork::online_er_init_optimizer(string name, double alpha, double beta1, double beta2) {
    optimizer->erInitialize(name, alpha, beta1, beta2);
    onlineErStep = 0;

    _recErrLog.clear();
    _kldLog.clear();
    _kldLog.resize(nLayers);
    _recErrFullLog.clear();
    _kldFullLog.clear();
    _kldFullLog.resize(nLayers);
}

void PvrnnNetwork::onlineErInitialize(int epoch) {
    this->onlineErPredStep  = config->predStep;
    this->onlineErTotalStep = config->totalStep;
    this->gWindow           = config->gWindow;
    _erInitialize(epoch);
    data->onlineErInitialize( config->window );
    onlineErStep = 0;
}

void PvrnnNetwork::onlineErIteration(vectorXf1DContainer onlineErTarget, vectorXf1DContainer mask, string saveFilepath, string prefix, bool updateLastState) {
    /*
    This function computes one iteration of forward computation and backward computation (with error partially masked) including the posteior update
    */
    erReset();

    erForwardSequence(updateLastState);
    outputLayer->erComputeErr(onlineErTarget, mask);

    /* BPTT */
    erInitializeGradient();
    int max_t = gWindow ? min(onlineErStep, window) : window;
    for ( int _t = max_t - 1; _t > -1; _t-- ){
        /* Gradients at each timestep */
        outputLayer->erComputeGradient(onlineErTarget, mask, _t);
        layers.back()->erComputeGradient(outputLayer->wdLdo, _t);
        for ( int i = nLayers - 2; i > -1; i-- ){
            layers[i]->erComputeGradient(layers[i+1]->wdLdh, _t);
        }
    }
    optimizer->erComputeDelta();
    optimizer->erUpdateParameters();
    if (!prefix.empty()) onlineErSaveSequences(saveFilepath, prefix, false);
}

void PvrnnNetwork::onlineErPrediction(int predSteps, string saveFilepath, string prefix, bool updateLastState){  //FIXME: should not exist.
    /*
    This function generates the prediction after all the iterations.
    First this generates with posterior inside the window.
    Then, it generates with prior for the time steps specified by "predSteps"
    */
    erReset();

    int max_t = gWindow ? min(onlineErStep, window) : window;
    for ( int _t = 0; _t < max_t; _t++ ){
        erForwardStep();
        if ( _t == 0 && updateLastState ){ erKeepLastState(); }
    }

    for ( int _t = 0; _t < predSteps; _t++ ){
        erPriorStep();
    }

    outputLayer->erComputeErr(*(data->onlineErTarget));
    if (!prefix.empty()) onlineErSaveSequences(saveFilepath, prefix, true);
}

void PvrnnNetwork::onlinePlanGeneration(Map<VectorXf> &input, vectorXf1DContainer mask, string savePath, bool dynamicStep, bool saveIter) {
    if (dynamicStep && onlineErStep > 0) {
        // Manually manage target
        data->onlineErAnalogTarget.erase( data->onlineErAnalogTarget.begin() );
        data->onlineErTarget->erase( data->onlineErTarget->begin() );
    }
    onlineErStep++;
    data->onlineErKeepTarget(input); // current sensory info goes into history
    int itrLimit = (onlineErStep > 1) ? nItr : nInitItr;

    if (!savePath.empty()) {
        std::filesystem::create_directories(std::filesystem::path(savePath).parent_path());
        savePath += ".npz";
        std::filesystem::remove(savePath); // remove any previous save
        data->onlineErSaveTarget(savePath); // save the current target (in analog dim)
    }

    erRegisterParameters();
    optimizer->erRegisterParameters(erAPtr, erAGradPtr);

    erTarget = vectorXf1DContainer{ begin(*(data->onlineErTarget)), end(*(data->onlineErTarget)) }; // copy past history
    erTarget.insert(erTarget.end(), data->erData->begin() + (*(data->onlineErTarget)).size(), data->erData->end()); // copy goal signal

    double _kld;
    string itrSave = "";
    for (int itr = 0; itr < itrLimit; itr++) {
        if (!savePath.empty() && (saveIter || itr==nItr-1)) itrSave = "itr" + to_string( itr );
        onlineErIteration(erTarget, mask, savePath, itrSave, (dynamicStep && itr == 0));
        _kld = 0.0;
        for (int i = 0; i < nLayers; i++) {
            if (wKLD)
                _kld += w[i] * accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / layers[i]->currentWindowSize;
            else
                _kld += accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / layers[i]->currentWindowSize;
        }

        // Save KLD and rec. error
        for (int i = 0; i < nLayers; i++) {
            _kldLog[i].push_back(accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / layers[i]->currentWindowSize);
            if (wKLD) _kldLog[i].back() *= w[i];
            _kldFullLog[i].push_back(layers[i]->erKldSeq);
        }
        _recErrLog.push_back(accumulate(outputLayer->erErrSeq.begin(), outputLayer->erErrSeq.end(), 0.0) / outputLayer->erErrSeq.size());
        _recErrFullLog.push_back(outputLayer->erErrSeq);
    }
    if (dynamicStep) {
        erSlideWindow();
    }
}

void PvrnnNetwork::onlinePlanGeneration(float *_input, float *_mask, string savePath, bool dynamicStep, bool saveIter) {
    Map<VectorXf> input(_input, outputSize);
    vectorXf1DContainer mask;
    if (_mask != nullptr) {
        for (int i = 0; i < window; i++) {
            Map<VectorXf> dmask(_mask+(i*outputLayer->outputSize), outputLayer->outputSize);
            mask.push_back(dmask);
        }
    } else {
        mask = vector<VectorXf>();
    }
    onlinePlanGeneration(input, mask, savePath, dynamicStep, saveIter);
}

VectorXf PvrnnNetwork::onlineErrorRegression(Map<VectorXf> &input, vectorXf1DContainer mask, string savePath, bool verbose, bool saveIterPredictions) {
    /*
    This function receives the current joint angle and returns the one time-step look-ahead prediction
    This function should be called from Python iterface at every sensory time step
    */
    // if (seed != -1) {
    //   mt19937 mt;
    //   mt.seed(5);
    // }
    onlineErStep++;
    data->onlineErKeepTarget( input );

    int itrLimit = nItr;
    if (window > onlineErStep - 1) itrLimit = nInitItr;

    if ( !gWindow && window > onlineErStep - 1 ){
        return VectorXf{ VectorXf::Zero( outputSize )};
    }
    else{
        string erDirectory;
        if (!savePath.empty()) {
            erDirectory = savePath;
            savePath += ".npz";
            std::filesystem::create_directories(std::filesystem::path(erDirectory).parent_path());
            std::filesystem::remove(savePath); // remove any previous save
            data->onlineErSaveTarget(savePath); // save the current target (in analog dim)
        }
        erRegisterParameters();
        optimizer->erRegisterParameters(erAPtr, erAGradPtr);
        erTarget = *(data->onlineErTarget);
        double _kld;
        string itrSave = "";
        for ( int itr = 0; itr < nItr; itr++ ){
            if (!savePath.empty() && (saveIterPredictions || itr==nItr-1)) itrSave = "itr" + to_string( itr );
            onlineErIteration(erTarget, mask, savePath, itrSave);
            _kld = 0.0;
            for ( int i = 0; i < nLayers; i++ ){
                if (wKLD)
                    _kld += w[i] * accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / layers[i]->currentWindowSize;
                else
                    _kld += accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / layers[i]->currentWindowSize;
            }

            if (saveIterPredictions) { // generate predictions every iteration
                onlineErPrediction(onlineErPredStep, savePath, itrSave, (!layers[0]->initialWindow && itr==nItr-1));
            } else if (itr==nItr-1) { // otherwise only save on last iteration
                onlineErPrediction(onlineErPredStep, savePath, itrSave, (layers[0]->initialWindow ? false: true));
            }
            // Save KLD and rec. error
            for (int i = 0; i < nLayers; i++) {
                _kldLog[i].push_back(accumulate(layers[i]->erKldSeq.begin(), layers[i]->erKldSeq.end(), 0.0) / layers[i]->currentWindowSize);
                if (wKLD) _kldLog[i].back() *= w[i];
                _kldFullLog[i].push_back(layers[i]->erKldSeq);
            }
            _recErrLog.push_back(accumulate(outputLayer->erErrSeq.begin(), outputLayer->erErrSeq.end(), 0.0) / outputLayer->erErrSeq.size());
            _recErrFullLog.push_back(outputLayer->erErrSeq);

            if (verbose) {
                cout << "step " << right << setw(3) << onlineErStep << '.' << left << setw(3) << itr
                << " recerr: " << accumulate(outputLayer->erErrSeq.begin(), outputLayer->erErrSeq.end(), 0.0) / outputLayer->erErrSeq.size()
                << " kld: " << _kld << "\n";
            }
        }
        erSlideWindow();
        VectorXf output{};
        int max_t = gWindow ? min(onlineErStep, window) : window;
        output = (*(outputLayer->erOutSeq))[max_t];
        return output;
    }
}

void PvrnnNetwork::onlineErrorRegression(float *_input, float *output, float *_mask, string directory, bool verbose, bool saveIterPredictions) {
    Map<VectorXf> input(_input, outputSize);
    vectorXf1DContainer mask;
    if (_mask != nullptr) {
        int outwinlen = gWindow ? min(onlineErStep+1, window) : window;
        for (int i = 0; i < outwinlen; i++) {
            Map<VectorXf> dmask(_mask+(i*outputLayer->outputSize), outputLayer->outputSize);
            mask.push_back(dmask);
        }
    } else {
        mask = vector<VectorXf>();
    }
    Map<VectorXf>(output, outputSize) = onlineErrorRegression(input, mask, directory, verbose, saveIterPredictions);
}

void PvrnnNetwork::onlineErSaveSequences(string saveFilepath, string prefix, bool prediction){
    for ( int i = 0; i < nLayers; i++ ){
        layers[i]->erSaveSequences(saveFilepath, prefix + "_layer" + to_string( nLayers - i ), prediction);
    }
    outputLayer->erSaveSequence(saveFilepath, prefix + "_output", prediction, true);
}

/* This function returns prediction with prior without error regression (for testing) */
VectorXf PvrnnNetwork::priorGeneration() {
    erSeqReset();
    erPriorStep();
    return data->returnOriginal(outputLayer->erxSeq[0]);
}

// FIXME: remove input here, obviously useless.
void PvrnnNetwork::priorGeneration(float *_input, float *output) {
    Map<VectorXf>(output, outputSize) = priorGeneration();
}

void PvrnnNetwork::erSaveSequences(string saveFilepath, string prefix, bool prediction){
    for ( int i = 0; i < nLayers; i++ ){
        layers[i]->erSaveSequences(saveFilepath, prefix + "_layer" + to_string(nLayers - i), prediction);
    }
    outputLayer->erSaveSequence(saveFilepath, prefix + "_output", prediction, false);
}

void PvrnnNetwork::saveSequences(string directory, bool force) {
    if (force || (epoch > 0)) {  // no sequence generated before one epoch is done (and generate a segfault if no training)
        std::filesystem::create_directories(directory);
        for (int i = 0; i < nLayers; i++) {
            layers[i]->saveSequences(directory + "/layer" + to_string(nLayers - i));
        }
        if ( outputLayer->model == "sm" ){ // FIXME: do we  need that? Was it for debug purposes?
            outputLayer->saveSequences(directory + "/smLayer");
        }
        data->saveRawSeq(outputLayer->xSeq, directory + "/output");
    }
}

void PvrnnNetwork::saveParameters(string directory) {
    std::filesystem::create_directories(directory);
    for (int i = 0; i < nLayers; i++) {
        layers[i]->saveParameters(directory + "/layer" + to_string(nLayers - i));
    }
    outputLayer->saveParameters(directory + "/" + outputLayer->model + "Layer");
}

void PvrnnNetwork::loadParameters(string directory, bool rowMajorW) {
    for (int i = 0; i < nLayers; i++) {
        layers[i]->loadParameters(directory + "/layer" + to_string(nLayers - i), rowMajorW);
    }
    outputLayer->loadParameters(directory + "/" + outputLayer->model + "Layer", rowMajorW);
}

void PvrnnNetwork::loadMetadata(string directory) {
    cout << "loading metadata from " << directory << " ...";
    try {
        metadata = toml::parse_file(directory + "/metadata.toml");
        data->rawMin = metadata["data"]["rawMin"].value_or<double>(-1.0);
        data->rawMax = metadata["data"]["rawMax"].value_or<double>(-1.0);
        data->normMin = metadata["data"]["normMin"].value_or<double>(-1.0);
        data->normMax = metadata["data"]["normMax"].value_or<double>(-1.0);
        /* Data parameters are also setup from config */
        cout << "configuring data...";
        data->nSeq = nSeq;
        data->seqLen = seqLen;
        data->rawDim = metadata["data"]["rawDim"].value_or<double>(-1.0); // load this from metadata since it might not be configured
        data->outputSize = (data->outputLayer == "sm") ? outputSize * smUnit : outputSize;
        if (data->outputLayer == "sm") {
            data->_computeScalingMatrix();
            data->_computeSigmaArray();
            data->_computeRefVec();
        }
        saveRowMajorW = metadata["save"]["rowMajorW"].value_or<bool>(false);
        cout << "done\n";
    } catch (const toml::parse_error &e) {
        if (data->rawData.size() != 0) {
            saveRowMajorW = false; // assume old save structure
            cout << "failed, using training data instead\n";
        } else {
            throw runtime_error("Failed to get metadata, training data must be loaded first");
        }
    }
}


//void PvrnnNetwork::tBPTT(int minibatchInd){
//    zeroGradient();
//    for ( int seqInd = 0; seqInd < minibatchSize; seqInd++ ){
//        for ( int interval = 0; interval < bpttInd.size() - 1; interval++ ){
//            initBPTT();
//            for ( int _t = bpttInd[interval] - 1; _t > bpttInd[interval+1] - 1; _t-- ){
//                computeGradient(minibatchInd, seqInd, _t);
//            }
//        }
//    }
//}


//void PvrnnNetwork::clipGradient(int minibatchInd){
//    for ( int l = 0; l < nLayers; l++ ){
//        layers[l]->clipGradient( minibatchInd );
//    }
//}
