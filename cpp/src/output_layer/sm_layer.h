/*
This is SoftmaxtLayer class which is instantiated in PvrnnNetwork class.
*/

#ifndef sm_layer_h
#define sm_layer_h

#include "output_layer.h"

class SoftmaxLayer: public OutputLayer {


public:
    SoftmaxLayer(int inputSize, int anaDim, int smUnit, int seqLen, int nSeq,
                 int minibatchSize, vectorXf1DContainer* smErxSeq, default_random_engine* engine, bool vectorZeroInit=false);
    ~SoftmaxLayer() {}

    int anaDim;         // analog output dimension
    int smUnit;         // number of softmax dimension per analog dimension

    double computeErr(const Tensor3f& target, const vector<int>& minibatch) override;
    void computeGradient(const Tensor3f& target, const Tensor3f& dSeq, int seqInd, int t) override;

    void erComputeErr(const vectorXf1DContainer& target,
                      const vectorXf1DContainer& mask) override;               // compute rec. err. with optional mask of the same size as the output
    void erComputeGradient(const vectorXf1DContainer& target,
                           const vectorXf1DContainer& mask, int t) override;   // compute one step gradient with optional rec. err. masked

    void erSaveSequence(string saveFilepath, string prefix, bool prediction, bool onlineEr) override;

protected:
    MatrixXf sMat;      // scaling matrix (see details in implementation code)
    ArrayXf eps;        // small value to stabilize log operation

    VectorXf _forwardCompute(const VectorXf& d) override;

};

#endif /* sm_layer_h */
