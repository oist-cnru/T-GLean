/*
This is OutputLayer class which is instantiated in PvrnnNetwork class.
This class implements a simple fully connected layer with tanh activation function.
This can be replaced by other output layer such as softmax layer later.
*/

#ifndef fc_layer_h
#define fc_layer_h

#include "output_layer.h"


class FCLayer: public OutputLayer {

public:
    FCLayer(int inputSize, int outputSize, int seqLen, int nSeq, int minibatchSize, default_random_engine* engine, bool vectorZeroInit=false);
    ~FCLayer() {}

    double computeErr(const Tensor3f& target, const vector<int>& minibatch) override;
    void computeGradient(const Tensor3f& target, const Tensor3f& dSeq, int seqInd, int t) override;

    void erComputeErr(const vectorXf1DContainer& target,
                      const vectorXf1DContainer& mask) override;               // compute rec. err. with optional mask of the same size as the output
    void erComputeGradient(const vectorXf1DContainer& target,
                           const vectorXf1DContainer& mask, int t) override;   // compute one step gradient with optional rec. err. masked

    void erSaveSequence(string saveFilepath, string prefix, bool prediction, bool onlineEr) override;

protected:
    VectorXf _forwardCompute(const VectorXf& d) override;
};


#endif /* fc_layer.h */
