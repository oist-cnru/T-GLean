
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>

#include "network_config.h"


void networkConfig::displayTrainConfig() {
    // print out configuration
    cout << "\n NETWORK CONFIGURATION \n";
    cout << "         #d  #z tau      w   beta" << "\n";
    for ( size_t i = 0; i < dSize.size(); i++ ){
        cout << "layer#" << setw(1) << dSize.size()-i << " "
                         << setw(3) << dSize[i]       << " "
                         << setw(3) << zSize[i]       << " "
                         << setw(3) << tau[i]         << " "
                         << setw(6) << w[i]           << " "
                         << setw(6) << beta[i]        << "\n";
    }
    cout << "output layer: ";
    if ( outputLayer == "sm" ){
        cout << "softmax \n";
        cout << "output-dim: " << outputSize << "  Softmax-unit/dim: " << smUnit << "\n";
        cout << "sigma for softmax transformation: ";
        for (size_t i = 0; i < smSigma.size(); i++ ){
            cout << smSigma[i] << " ";
        }
        cout << "\n";
    }
    else if ( outputLayer == "fc" ){
        cout << "fully-connected layer with tanh activation function\n";
        cout << "output-dim: " << outputSize << "\n";
    }
    else {
        cout << "unknown output layer: " << outputLayer << endl;
    }
    if (sigmaMinVal != 0 && sigmaMaxVal != 0) {
        if (sigmaMinVal != 0) { cout << "sigma_min: " << sigmaMinVal << " "; }
        if (sigmaMaxVal != 0) { cout << "sigma_max: " << sigmaMaxVal; }
        cout << endl;
    } else {
        cout << "no sigma clipping \n";
    }


    cout << "\n DATASET STRUCTURE \n";
    cout << "sequence-length: " << seqLen << "  #seq: " << nSeq << "  #seq/minibatch: " << minibatchSize << "  #minibatch: " << nSeq/minibatchSize << "\n";
    cout << "Data normalization upper limit: " << normMax << "  lower limit: " << normMin << "\n";
    cout << "\n LEARNING SCHEDULE \n";
    cout << "#epoch: " << nEpoch << "  save-interval: " << saveInterval << "\n";
    cout << "\n ADAM OPTIMIZER PARAMETERS \n";
    cout << "alpha = " << alpha << "  beta1 = " << beta1 << "  beta2 = " << beta2 << "\n\n";
}

void networkConfig::displayBatchErConfig() {
    // print out configuration
    cout << "\n ERROR REGRESSION CONFIGURATION \n";
    cout << "window-size: " << window << " #epoch: " << nInitItr << " (t=1), " << nItr << "(else) \n";
    cout << "w values during error regression: ";
    for ( size_t i = 0; i < erW.size(); i++ ){
        cout << erW[i] << " ";
    }
    cout << "\n";
    cout << "\n ADAM OPTIMIZER PARAMETERS \n";
    cout << "alpha=" << erAlpha << "  beta1=" << erBeta1 << "  beta2=" << erBeta2 << "\n\n";
}

void networkConfig::displayOnlineErConfig() {
    // print out configuration
    cout << "\n ERROR REGRESSION CONFIGURATION \n";
    cout << "growing-window: " << gWindow << "\n";
    cout << "window-size: " << window << " #epoch: " << nItr << "\n";
    cout << "w values during error regression: ";
    for ( size_t i = 0; i < erW.size(); i++ ){
        cout << erW[i] << " ";
    }
    cout << "\n";
    cout << "\n ADAM OPTIMIZER PARAMETERS \n";
    cout << "alpha=" << erAlpha << "  beta1=" << erBeta1 << "  beta2=" << erBeta2 << "\n\n";
}

int networkConfig::epochToLoad() {
    if (_epochToLoad == -1) { // try to read last saved epoch
        ifstream lastEpoFile(saveDirectory + "/learning/lastEpo.txt");
        if (lastEpoFile.fail()) {
            throw runtime_error("Failed to load last saved epoch in " + saveDirectory + "/learning/lastEpo.txt");
        } else {
            string line;
            getline(lastEpoFile, line);
            try {
                _epochToLoad = stoi(line);
            } catch (const exception& e) {
                cerr << "error: failed to *parse* last saved epoch in " << saveDirectory + "/learning/lastEpo.txt" << endl;
                throw;
            }
            lastEpoFile.close();
        }
    }
    return _epochToLoad;
}

int networkConfig::epochToLoad(int fallback) {
    if (_epochToLoad == -1) { // try to read last saved epoch
        ifstream lastEpoFile(saveDirectory + "/learning/lastEpo.txt");
        if (lastEpoFile.fail()) {
            cerr << "warning: failed to load last saved epoch, falling back to epoch " << fallback << endl;
            _epochToLoad = fallback;
        } else {
            string line;
            getline(lastEpoFile, line);
            try {
                _epochToLoad = stoi(line);
            } catch(const exception& e) {
                cerr << "warning: failed to parse last saved epoch, falling back to epoch " << fallback << endl;
                _epochToLoad = fallback;
            }
            lastEpoFile.close();
        }
    }
    return _epochToLoad;
}

void networkConfig::importConfig(pvrnnConfig cfg, bool verbose) {
    /*
     * Takes network/ER configuration from an external source
     */
    if (cfg.nLayers > 0) {
        if ((cfg.dSize).size() > 0) dSize.clear();
        if ((cfg.zSize).size() > 0) zSize.clear();
        if ((cfg.tau).size() > 0) tau.clear();
        if ((cfg.w).size() > 0) w.clear();
        if ((cfg.beta).size() > 0) beta.clear();
        if ((cfg.erW).size() > 0) erW.clear();
        if ((cfg.erBeta).size() > 0) erBeta.clear();
        for (int l = 0; l < cfg.nLayers; l++) {
            if ((cfg.dSize).size() > 0) dSize.push_back((cfg.dSize)[l]);
            if ((cfg.zSize).size() > 0) zSize.push_back((cfg.zSize)[l]);
            if ((cfg.tau).size() > 0) tau.push_back((cfg.tau)[l]);
            if ((cfg.w).size() > 0) w.push_back((cfg.w)[l]);
            if ((cfg.beta).size() > 0) beta.push_back((cfg.beta)[l]);
            if ((cfg.erW).size() > 0) erW.push_back((cfg.erW)[l]);
            if ((cfg.erBeta).size() > 0) erBeta.push_back((cfg.erBeta)[l]);
        }
    }
    if (cfg.outputSize != -1) outputSize = cfg.outputSize;
    if (cfg.smUnit != -1) smUnit = cfg.smUnit;
    if (cfg.seqLen != -1) seqLen = cfg.seqLen;
    if (cfg.nSeq != -1) nSeq = cfg.nSeq;
    if (cfg.sigmaMaxVal != -1.0) sigmaMaxVal = cfg.sigmaMaxVal;
    if (cfg.sigmaMinVal != -1.0) sigmaMinVal = cfg.sigmaMinVal;
    if (!cfg.outputLayer.empty()) outputLayer = cfg.outputLayer;
    if (cfg.minibatchSize != -1) minibatchSize = cfg.minibatchSize;
    if (cfg.smSigma.size() > 0) {
        smSigma.clear();
        for (size_t i = 0; i < cfg.smSigma.size(); i++) smSigma.push_back(cfg.smSigma[i]);
    }
    if (cfg.normMin != cfg.normMax) {
        normMin = cfg.normMin;
        normMax = cfg.normMax;
    }
    if (cfg.dataMin != cfg.dataMax) {
        dataMin = cfg.dataMin;
        dataMax = cfg.dataMax;
    }
    if (cfg.datasetDirectory != "") datasetDirectory = cfg.datasetDirectory;
    if (cfg.nEpoch != -1) nEpoch = cfg.nEpoch;
    if (cfg.saveInterval != -1) saveInterval = cfg.saveInterval;
    if (!cfg.optimizerAlgorithm.empty()) optimizerAlgorithm = cfg.optimizerAlgorithm;
    if (cfg.alpha != -1.0) alpha = cfg.alpha;
    if (cfg.beta1 != -1.0) beta1 = cfg.beta1;
    if (cfg.beta2 != -1.0) beta2 = cfg.beta2;
    if (cfg.a_eps != -1.0) a_eps = cfg.a_eps;
    if (!cfg.saveDirectory.empty()) saveDirectory = cfg.saveDirectory;
    if (cfg.gWindow != -1) gWindow = (bool)cfg.gWindow;
    if (cfg.window != -1) window = cfg.window;
    if (cfg.erSeqLen != -1) erSeqLen = cfg.erSeqLen;
    if (cfg.nInitItr != -1) nInitItr = cfg.nInitItr;
    if (cfg.nItr != -1) nItr = cfg.nItr;
    if (cfg.erStep != -1) erStep = cfg.erStep;
    if (!cfg.erOptimizerAlgorithm.empty()) erOptimizerAlgorithm = cfg.erOptimizerAlgorithm;
    if (cfg.erAlpha != -1.0) erAlpha = cfg.erAlpha;
    if (cfg.erBeta1 != -1.0) erBeta1 = cfg.erBeta1;
    if (cfg.erBeta2 != -1.0) erBeta2 = cfg.erBeta2;
    if (!cfg.erDataDirectory.empty()) erDataDirectory = cfg.erDataDirectory;
    if (!cfg.erSaveDirectory.empty()) erSaveDirectory = cfg.erSaveDirectory;
    if (cfg.epochToLoad != 0) _epochToLoad = cfg.epochToLoad;
    if (cfg.predStep != -1) predStep = cfg.predStep;
    if (cfg.totalStep != -1) totalStep = cfg.totalStep;
    if (!cfg.baseDirectory.empty()) trainConfigPath = cfg.baseDirectory;
    if (!cfg.erBaseDirectory.empty()) {
        batchErConfigPath = filesystem::path(cfg.erBaseDirectory);
        onlineErConfigPath = filesystem::path(cfg.erBaseDirectory);
    }

    if (!cfg.pOutputLayer.empty()) pOutputLayer = cfg.pOutputLayer;
    if (!cfg.pDatasetDirectory.empty()) pDatasetDirectory = cfg.pDatasetDirectory;
    if (!cfg.vDatasetDirectory.empty()) vDatasetDirectory = cfg.vDatasetDirectory;
    for (size_t i = 0; i < cfg.adSize.size(); i++) adSize.push_back(cfg.adSize[i]);
    for (size_t i = 0; i < cfg.pdSize.size(); i++) pdSize.push_back(cfg.pdSize[i]);
    for (size_t i = 0; i < cfg.vdSize.size(); i++) vdSize.push_back(cfg.vdSize[i]);
    for (size_t i = 0; i < cfg.azSize.size(); i++) azSize.push_back(cfg.azSize[i]);
    for (size_t i = 0; i < cfg.pzSize.size(); i++) pzSize.push_back(cfg.pzSize[i]);
    for (size_t i = 0; i < cfg.vzSize.size(); i++) vzSize.push_back(cfg.vzSize[i]);
    for (size_t i = 0; i < cfg.aTau.size(); i++) aTau.push_back(cfg.aTau[i]);
    for (size_t i = 0; i < cfg.pTau.size(); i++) pTau.push_back(cfg.pTau[i]);
    for (size_t i = 0; i < cfg.vTau.size(); i++) vTau.push_back(cfg.vTau[i]);
    for (size_t i = 0; i < cfg.aW.size(); i++) aW.push_back(cfg.aW[i]);
    for (size_t i = 0; i < cfg.pW.size(); i++) pW.push_back(cfg.pW[i]);
    for (size_t i = 0; i < cfg.vW.size(); i++) vW.push_back(cfg.vW[i]);
    for (size_t i = 0; i < cfg.aBeta.size(); i++) aBeta.push_back(cfg.aBeta[i]);
    for (size_t i = 0; i < cfg.pBeta.size(); i++) pBeta.push_back(cfg.pBeta[i]);
    for (size_t i = 0; i < cfg.vBeta.size(); i++) vBeta.push_back(cfg.vBeta[i]);
    if (cfg.pOutputSize != -1) pOutputSize = cfg.pOutputSize;
    if (cfg.vOutputSize != -1) vOutputSize = cfg.vOutputSize;

    if (verbose) {
        displayTrainConfig();
        displayOnlineErConfig();
    }
}
