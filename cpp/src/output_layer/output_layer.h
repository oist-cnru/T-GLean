/*
This is OutputLayer class which integrates Softmaxlayer and FCLayer
*/

#ifndef output_layer_h
#define output_layer_h

#include <string>
#include <data.h>
#include "../includes.h"


// This is an abstract class
class OutputLayer {

public:
    virtual ~OutputLayer() {};
    Data* data; // hold a pointer to the current data object for softmax
    string model;   // model name: "fc" or "sm"
    int inputSize;
    int outputSize;
    int seqLen;
    int nSeq;
    int minibatchSize;
    bool vectorZeroInit;

    default_random_engine* engine; // random number generator

    // common part
    MatrixXf W;      // mapping d_{t} for computing v_t
    MatrixXf dLdW;
    VectorXf b;      // bias for computing v_t
    VectorXf dLdb;
    VectorXf x;      // output
    Tensor3f xSeq;   // hold multiple output sequences
    VectorXf dedo;
    VectorXf wdLdo;  // gradient propagated to the connected layer
    vector<double> erErrSeq;
    vectorXf1DContainer erxSeq;
    vectorXf1DContainer* erOutSeq;

    // Forward computation
    void initializeSequence();
    void computeOutput(const VectorXf &d, int seqInd);
    virtual double computeErr(const Tensor3f& target, const vector<int>& minibatch) = 0;

    // Backward computation
    virtual void computeGradient(const Tensor3f& target, const Tensor3f& dSeq, int seqInd, int t) = 0;
    void zeroGradient();

    // ER computation
    void erInitialize(int window);
    void erComputeOutput(const VectorXf& d);
    virtual void erComputeErr(const vectorXf1DContainer& target, const vectorXf1DContainer& mask) = 0;
    void erComputeErr(const vectorXf1DContainer& target) { erComputeErr(target, vectorXf1DContainer()); };
    virtual void erComputeGradient(const vectorXf1DContainer& target, const vectorXf1DContainer& mask, int _t) = 0;
    void erComputeGradient(const vectorXf1DContainer& target, int _t) { erComputeGradient(target, vectorXf1DContainer(), _t); };
    void erReset();

    // Saving & loading parameters and sequences.
    void saveSequences(string directory);
    void saveParameters(string directory);
    void loadParameters(string directory, bool rowMajorW);
    void loadParameters(string directory);

    // FIXME: no np.array version
    void erSaveErr(string directory);
    virtual void erSaveSequence(string saveFilepath, string prefix, bool prediction, bool onlineEr) = 0;

protected:
    OutputLayer(int inputSize, int outputSize, int seqLen, int nSeq, int minibatchSize, default_random_engine* engine);
    vector<int> curSeqPos;      // current length of the sequence in tensor xSeq
    // Temporary tensors for minibatch
    Tensor3f btarget;
    Tensor3f bxSeq;
    Tensor3f _stopSignals;
    // Temporary vectors for gradient calculation
    VectorXf _xSeq;
    VectorXf _target;
    VectorXf _dBatch;
    VectorXf _stopCost;

    // error regression
    int window;

    void _initMatrices(bool vectorZeroInit);
    virtual VectorXf _forwardCompute(const VectorXf& d) = 0;

    void _saveSequencesNpz(string directory);
    void _saveParametersNpy(string directory);
    void _loadParametersNpy(string directory, bool rowMajorW);

};

#endif
