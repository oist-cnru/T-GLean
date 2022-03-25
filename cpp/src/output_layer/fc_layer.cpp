#include <iostream>
#include <fstream>
#include <filesystem>

#include "fc_layer.h"


FCLayer::FCLayer(int inputSize, int outputSize, int seqLen, int nSeq, int minibatchSize,
                 default_random_engine* engine, bool vectorZeroInit):
                 OutputLayer(inputSize, outputSize, seqLen, nSeq, minibatchSize, engine)
{
    model = "fc";
    erOutSeq = &erxSeq;
    _initMatrices(vectorZeroInit);
}

VectorXf FCLayer::_forwardCompute(const VectorXf& d){
    /* eq f-3, f-14 */
    return ( W * d + b ).array().tanh();
}

double FCLayer::computeErr(const Tensor3f &target, const vector<int>& minibatch) {
    gatherFromTensor3(target, btarget, minibatch);
    gatherFromTensor3(xSeq, bxSeq, minibatch);
    /* eq e-3 */
    Tensor0f err = 0.5 * (btarget - bxSeq).square().sum();
    return (double)*(err.data()) / (outputSize * seqLen * minibatchSize);
}

void FCLayer::computeGradient(const Tensor3f &target, const Tensor3f &dBatch, int seqInd, int t){
    _dBatch = VectorXf(dBatch.dimension(2));
    getFromTensor3<VectorXf>(target, _target, seqInd, t);
    getFromTensor3<VectorXf>(xSeq, _xSeq, seqInd, t);
    getFromTensor3<VectorXf>(dBatch, _dBatch, seqInd, t);
    dedo  = ( _xSeq - _target ).array() * ( 1.0 - _xSeq.array().square() ) / ( outputSize * minibatchSize ); // normalize the gradient
    wdLdo.noalias() = W.transpose() * dedo;
    dLdW += dedo * _dBatch.transpose();
    dLdb += dedo;
}

void FCLayer::erComputeErr(const vectorXf1DContainer& target, const vectorXf1DContainer& mask) {
    // this error is not normalized by sequence length
    if (!mask.empty()) {
        for (size_t t = 0; t < erxSeq.size(); t++) {
            erErrSeq.push_back(0.5 * ((erxSeq[t] - target[t]).array()*mask[t].array()).array().square().sum() / outputSize);
        }
    } else {
        for (size_t t = 0; t < target.size(); t++){
            erErrSeq.push_back(0.5 * ( erxSeq[t] - target[t] ).array().square().sum() / outputSize);
        }
    }
}

void FCLayer::erComputeGradient(const vectorXf1DContainer& target, const vectorXf1DContainer& mask, int t) {
    if (!mask.empty()) {
        dedo = (((erxSeq[t] - target[t]).array() * mask[t].array()).array()
                * (1.0 - erxSeq[t].array().square())).matrix() / outputSize;
    } else {
        dedo = ((erxSeq[t] - target[t]).array() * (1.0 - erxSeq[t].array().square())).matrix() / outputSize;
    }
    wdLdo.noalias() = W.transpose() * dedo;
}

void FCLayer::erSaveSequence(string saveFilepath, string prefix, bool prediction, bool onlineEr){
    string mode = (std::filesystem::exists(saveFilepath) ? "a" : "w");

    size_t pWindowSize = (prediction ? erxSeq.size() : (onlineEr ? erErrSeq.size() : window));
    Tensor2f _x = Tensor2f(pWindowSize, erxSeq[0].size());
    for ( size_t _t = 0; _t < pWindowSize; _t++ ){
        setInTensor2<VectorXf>(_x, erxSeq[_t], _t);
    }
    
    cnpy::npz_save(saveFilepath, prefix + "_x", _x.data(), {pWindowSize, (size_t)erxSeq[0].size()}, mode);
    cnpy::npz_save(saveFilepath, prefix + "_recErr", erErrSeq.data(), {(size_t)erErrSeq.size()}, "a");
}
