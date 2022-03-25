#include <iostream>
#include <fstream>
#include <filesystem>

#include "output_layer.h"
using namespace std;

OutputLayer::OutputLayer(int inputSize, int outputSize, int seqLen, int nSeq, int minibatchSize,
                         default_random_engine* engine):
    inputSize(inputSize), outputSize(outputSize), seqLen(seqLen), nSeq(nSeq), minibatchSize(minibatchSize),
    engine(engine)
{}

void OutputLayer::_initMatrices(bool vectorZeroInit) {
    // initializing weight matrices and biases
    W = glorotNormalInit(outputSize, inputSize, engine);
    if (vectorZeroInit) { b = static_cast<VectorXf>(VectorXf::Zero  (outputSize)); }
    else                { b = static_cast<VectorXf>(VectorXf::Random(outputSize)); }

    dedo = VectorXf(outputSize);
    wdLdo = VectorXf(W.cols());

    _xSeq = VectorXf(outputSize);   // I guess these are needed to keep a pointer before the gradient computation
    _target = VectorXf(outputSize);
    btarget = Tensor3f(minibatchSize, seqLen, outputSize);
    bxSeq   = Tensor3f(minibatchSize, seqLen, outputSize);
}

void OutputLayer::initializeSequence() {
    xSeq = Tensor3f(nSeq, seqLen, outputSize);
    curSeqPos = vector<int>(nSeq, 0);
}

void OutputLayer::computeOutput(const VectorXf &d, int seqInd){
    // cout << "D " << d << endl;
    x = _forwardCompute(d);
    // cout << "X " << x << endl;
    setInTensor3<VectorXf>(xSeq, x, seqInd, curSeqPos[seqInd]);
    curSeqPos[seqInd] = min(curSeqPos[seqInd] + 1, seqLen - 1);
}


    /* Backward Computation */

void OutputLayer::zeroGradient(){
    dLdW = MatrixXf::Zero(outputSize, inputSize);
    dLdb = VectorXf::Zero(outputSize);
}



    /* Loading and Saving */

void OutputLayer::saveParameters(string directory) {
    _saveParametersNpy(directory);
}

void OutputLayer::loadParameters(string directory, bool rowMajorW) {
    _loadParametersNpy(directory, rowMajorW);
}

void OutputLayer::saveSequences(string directory) {
    _saveSequencesNpz(directory);
}

// NPY

void OutputLayer::_saveParametersNpy(string directory) {
    std::filesystem::create_directories(directory);

    cnpy::npy_save(directory + "/W.npy", swapLayout(W).data(), {(size_t)W.rows(), (size_t)W.cols()}, "w");
    cnpy::npy_save(directory + "/b.npy", b.data(), {(size_t)b.size()}, "w");
}

void OutputLayer::_loadParametersNpy(string directory, bool rowMajorW) {
    cout << "loading parameters from " << directory << " ...";
    cnpy::NpyArray _W;
    cnpy::NpyArray _b;
    try {
        _W = cnpy::npy_load(directory + "/W.npy");
        _b = cnpy::npy_load(directory + "/b.npy");
    } catch (const std::runtime_error& e) {
        cerr << "failed to load parameter file" << endl;
        throw;
    }

    checkedLoadFromNp(_W, W, outputSize, inputSize, rowMajorW, "W");
    checkedLoadFromNp(_b, b, outputSize, "b");

    cout << "done\n";
}

void OutputLayer::_saveSequencesNpz(string directory) {
    std::filesystem::create_directories(directory);
    size_t xLen = xSeq.dimension(1);
    size_t xDim = xSeq.dimension(2);
    Tensor2f seq(xLen, xDim);
    string mode = "";
    for (int seqInd = 0; seqInd < nSeq; seqInd++) {
        mode = (seqInd > 0) ? "a" : "w"; // overwrite any existing files
        getTensor2FromTensor3(xSeq, seq, seqInd);
        cnpy::npz_save(directory + "/x.npz", "x_" + to_string(seqInd), seq.data(), {xLen, xDim}, mode);
    }
}


    // Error Regression

void OutputLayer::erInitialize(int window) {
    this->window = window;
}

void OutputLayer::erReset() {
    erxSeq.clear();
    erErrSeq.clear();
}

void OutputLayer::erComputeOutput(const VectorXf& d) {
    x = _forwardCompute(d);
    erxSeq.push_back(x);
}


    // Loading & Saving [ER]

void OutputLayer::erSaveErr(string directory) {
    std::filesystem::create_directories(directory);
    ofstream _err(directory + "/recErr.txt");
    for (size_t t = 0; t < erErrSeq.size(); t++) {
        _err << erErrSeq[t] << "\n";
    }
    _err.close();
}
