#include <iostream>
#include <fstream>
#include <filesystem>

#include "sm_layer.h"


SoftmaxLayer::SoftmaxLayer(int inputSize, int anaDim, int smUnit, int seqLen, int nSeq,
                           int minibatchSize, vectorXf1DContainer* smErxSeq, default_random_engine* engine, bool vectorZeroInit):
                           OutputLayer(inputSize, anaDim*smUnit, seqLen, nSeq, minibatchSize, engine), anaDim(anaDim), smUnit(smUnit)
{
    model = "sm";
    erOutSeq = smErxSeq;
    _initMatrices(vectorZeroInit);

    /* sMat is a matrix, I call scalling matrix, which may simplify the implementation of softmax transformation.
     * sMat simply tiles each element of a pre-transformed vector in softmax dimension.
     * e.g. for v = (1, 2)^T and softmax-unit/dim (`smUnit`) = 3, sMat * v = (1, 1, 1, 2, 2, 2)^T
     */
    sMat = MatrixXf::Zero( anaDim * smUnit, anaDim );
    for ( int dim = 0; dim < anaDim; dim++ ){
        sMat.block(dim * smUnit, dim, smUnit, 1).setConstant(1.0);
    }
    zeroGradient();
}

// Does a softmax transform
VectorXf SoftmaxLayer::_forwardCompute(const VectorXf& d) {
    /* eq f-3, f-15 */
    ArrayXf o = ( W * d + b ).array().exp();
    Map<MatrixXf> _norm(o.data(), smUnit, anaDim);
    return o / ( sMat * _norm.colwise().sum().transpose() ).array();
}

/* EXPERIMENTAL SETTINGS FOR TTM */
#define STOP_COST_FACTOR 0.1f
#define STOP_SM_TEMP 0.02f // this is NOT RELATED to softmax output layer
#define STOP_COST_TRAIN false // use TTM in training (not recommended)
#define STOP_COST_PLAN false // use TTM in planning
#define STOP_COST_BACKPROP true // backprop plan cost for error minimization (recommended)
/* EXPERIMENTAL SETTINGS FOR TTM */

double SoftmaxLayer::computeErr(const Tensor3f &target, const vector<int>& minibatch) {
    gatherFromTensor3(target, btarget, minibatch);
    gatherFromTensor3(xSeq, bxSeq, minibatch);
    /* eq e-4 */
    Tensor0f err = (btarget * clippedLog(btarget / bxSeq)).sum();

    #if STOP_COST_TRAIN
    // Extract stop signals
    Tensor3f outBxSeq(bxSeq.dimension(0), bxSeq.dimension(1), anaDim); // gather minibatch
    data->_invTransformSeq(bxSeq, outBxSeq); // inv softmax
    _stopSignals = outBxSeq.slice(Eigen::array<long, 3>{0, 0, outBxSeq.dimension(2)-1}, Eigen::array<long, 3>{outBxSeq.dimension(0), outBxSeq.dimension(1), 1}); // get stop signals

    // Low-temp softmax
    auto tsmt = [](auto ta, auto tb) { ta = ta / ta.constant(STOP_SM_TEMP); };
    mapToTensor3(_stopSignals, _stopSignals, tsmt);
    // Linear penalty function
    VectorXf _lref = VectorXf::LinSpaced(_stopSignals.dimension(1), 0.0, 1.0);
    MatrixXf _mref(_lref.size(), 1); // size dims * 1
    _mref.colwise() = Eigen::Map<Eigen::VectorXf>(_lref.data(), _stopSignals.dimension(1));
    Tensor3f _tref(_stopSignals.dimension(0), _stopSignals.dimension(1), _stopSignals.dimension(2));
    for (int i = 0; i < minibatch.size(); i++) setInTensor3<MatrixXf>(_tref, _mref, i);

    auto tsmc = [](auto ta, auto tb) { Tensor0f tx = ta.exp().sum(); ta = (ta.exp() / ta.constant(tx(0))) * tb; }; // convert raw signal to softmax
    mapToTensor3(_stopSignals, _tref, tsmc);
    Tensor0f plcost = _stopSignals.sum();
    // cout << "plcost: " << plcost(0) << endl;

    return (double)(*(err.data()) + (plcost(0)*STOP_COST_FACTOR)) / (outputSize * seqLen * minibatchSize);
    #else
    return (double)*(err.data()) / (outputSize * seqLen * minibatchSize);
    #endif
}

void SoftmaxLayer::computeGradient(const Tensor3f &target, const Tensor3f &dBatch, int seqInd, int t){
    _dBatch = VectorXf(dBatch.dimension(2));
    getFromTensor3<VectorXf>(target, _target, seqInd, t);
    getFromTensor3<VectorXf>(xSeq, _xSeq, seqInd, t);
    getFromTensor3<VectorXf>(dBatch, _dBatch, seqInd, t);

    dedo  = ( _xSeq - _target ) / ( outputSize * minibatchSize ); // normalize the gradient
    #if STOP_COST_TRAIN && STOP_COST_BACKPROP
    // Apply plan cost
    MatrixXf __stopCost(seqLen, 1);
    getFromTensor3<MatrixXf>(_stopSignals, __stopCost, seqInd);
    VectorXf ___stopCost = Map<VectorXf>(__stopCost.data(), seqLen);
    if (dedo.tail(smUnit).sum() < 0) dedo.tail(smUnit) *= STOP_COST_FACTOR*(1+___stopCost.sum())/outputSize; // apply cost multiplier to goal reached gradient
    #endif

    wdLdo.noalias() = W.transpose() * dedo;
    dLdW += dedo * _dBatch.transpose();
    dLdb += dedo;
}

void SoftmaxLayer::erComputeErr(const vectorXf1DContainer& smTarget, const vectorXf1DContainer& mask) {
    #if STOP_COST_PLAN
    ArrayXf stopSignal(smTarget.size());
    #endif
    // this error is not normalized by sequence length
    if (!mask.empty()) {
        for (size_t t = 0; t < smTarget.size(); t++) {
            erErrSeq.push_back(((smTarget[t].array() * (clippedLog(smTarget[t].array() / erxSeq[t].array())).array()).array() * mask[t].array()).sum()
                            / outputSize);
            #if STOP_COST_PLAN
            stopSignal[t] = (*erOutSeq)[t][anaDim-1]; // collect stop signals
            #endif
        }
    } else {
        for (size_t t = 0; t < smTarget.size(); t++) {
            erErrSeq.push_back((smTarget[t].array() * (clippedLog(smTarget[t].array() / erxSeq[t].array())).array()).sum()
                            / outputSize);
            #if STOP_COST_PLAN
            stopSignal[t] = (*erOutSeq)[t][anaDim-1]; // collect stop signals
            #endif
        }
    }
    #if STOP_COST_PLAN
    stopSignal /= STOP_SM_TEMP; // low-temp softmax
    ArrayXf penalty = ArrayXf::LinSpaced(stopSignal.size(), 0.0, 1.0);
    _stopCost = (stopSignal.exp() / stopSignal.exp().sum()) * penalty;
    for (size_t t = 0; t < erErrSeq.size(); t++) erErrSeq[t] += STOP_COST_FACTOR*_stopCost[t];
    #endif
}


void SoftmaxLayer::erComputeGradient(const vectorXf1DContainer& smTarget, const vectorXf1DContainer& mask, int t) {
    if (!mask.empty()) {
        dedo = ((erxSeq[t] - smTarget[t]).array() * mask[t].array()) / outputSize;
    } else {
        dedo = (erxSeq[t] - smTarget[t]) / outputSize;
    }
    #if STOP_COST_PLAN && STOP_COST_BACKPROP
    dedo.tail(smUnit).array() = -STOP_COST_FACTOR*(_stopCost[t]/outputSize); // negative gradient = shorter plans
    #endif
    wdLdo.noalias() = W.transpose() * dedo;
}

void SoftmaxLayer::erSaveSequence(string saveFilepath, string prefix, bool prediction, bool onlineEr){
    string mode = (std::filesystem::exists(saveFilepath) ? "a" : "w");

    size_t pWindowSize = (prediction ? erxSeq.size() : (onlineEr ? erErrSeq.size() : window));
    // Tensor2f _sm = Tensor2f(pWindowSize, erxSeq[0].size());
    Tensor2f _x = Tensor2f(pWindowSize, (*erOutSeq)[0].size());

    for ( size_t _t = 0; _t < pWindowSize; _t++ ){
        // setInTensor2<VectorXf>(_sm, erxSeq[_t], _t);
        setInTensor2<VectorXf>(_x, (*erOutSeq)[_t], _t);
    }

    // cnpy::npz_save(saveFilepath, prefix + "_sm", _sm.data(), {pWindowSize, (size_t)erxSeq[0].size()}, mode);
    cnpy::npz_save(saveFilepath, prefix + "_x", _x.data(), {pWindowSize, (size_t)(*erOutSeq)[0].size()}, mode);
    cnpy::npz_save(saveFilepath, prefix + "_recErr", erErrSeq.data(), {(size_t)erErrSeq.size()}, "a");
}
