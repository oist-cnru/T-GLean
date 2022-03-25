
#include <iostream>
#include <math.h>
#include "includes.h"
#include "optimizer.h"


Optimizer::Optimizer(string algorithm, float alpha, float beta1, float beta2){

    if (algorithm != "adam") throw invalid_argument("Optimizer: only adam is supported as algorithm\n");

    this->algorithm  = algorithm;
    this->alpha      = alpha;
    this->beta1      = beta1;
    this->beta2      = beta2;
    eps              = 1.0e-4;
    epoch            = 1.0;  // step in parameter update
}

void Optimizer::registerParameters(vector<MatrixXf*>& weight, vector<MatrixXf*>& weightGrad,
                                   vector<VectorXf*>& bias,   vector<VectorXf*>& biasGrad,
                                   vector<Tensor3f*>& APtr,   vector<Tensor3f*>& AGradPtr, int nMinibatch){
    /*
    receive vector of pointers of parameters and corresponding gradients and initialize optimizer
    */
    this->weight     = weight;
    this->weightGrad = weightGrad;
    this->bias       = bias;
    this->biasGrad   = biasGrad;
    this->APtr       = APtr;
    this->AGradPtr   = AGradPtr;
    this->nMinibatch = nMinibatch; // necesarry for update of A
    initializeOptimizer();
}

void Optimizer::initializeOptimizer(){
  /*
  prepare each momentum and delta quantity used in Adam computation
  */
  vector<MatrixXf> _matrix;
  for ( MatrixXf* _weight: weight ){
    _matrix.push_back(MatrixXf::Zero(_weight->rows(), _weight->cols()));
  }
  weight1stMomentum = _matrix;
  weight2ndMomentum = _matrix;
  weightDelta       = _matrix;

  vectorXf1DContainer _vector;
  for ( VectorXf* _bias: bias ){
    _vector.push_back(VectorXf::Zero(_bias->rows()));
  }
  bias1stMomentum = _vector;
  bias2ndMomentum = _vector;
  biasDelta       = _vector;

  vector<Tensor3f> _AContainer;
  for ( Tensor3f* _A: APtr ){
    _AContainer.push_back(Tensor3f(_A->dimension(0), _A->dimension(1), _A->dimension(2)));
    _AContainer.back().setZero();
  }
  A1stMomentum = _AContainer;
  A2ndMomentum = _AContainer;
  ADelta       = _AContainer;
}

void Optimizer::computeDelta(vector<int> minibatch){
  /*
  compute all deltas

  NOTE:
  Consider training a PV-RNN network with a dataset which has n minibatches.
  While weights and biases are updated for n times in one epoch, A terms are updated only once.
  Since the number of updates plays an important role in Adam, this asymmetry among parameters may affect the learning.
  To deal with this potential problem, another quantity "a_epoch" to count the number of updates for A terms in the following.
  */

  // denominator for beta1, beta2
  // note: not multiplying by the inverse for now because:
  // 1. keep the same results as the previous code.
  // 2. -ffast_math might be taking care of it anyway (benchmarking needed).

  float a_epoch{floor( ( epoch - 1.0f ) / nMinibatch ) + 1.0f}; // epoch for updating A temrms

  _computeDeltaAdam(weight1stMomentum, weight2ndMomentum, weightDelta, weightGrad, epoch);
  _computeDeltaAdam(bias1stMomentum,   bias2ndMomentum,   biasDelta,   biasGrad,   epoch);
  _computeDeltaAdam(A1stMomentum, A2ndMomentum, ADelta, AGradPtr, minibatch, a_epoch);

  epoch += 1.0;
}

void Optimizer::_computeDeltaAdam(vector<MatrixXf> &fMomentum, vector<MatrixXf> &sMomentum, vector<MatrixXf> &delta, vector<MatrixXf*> &grad, float _epoch){
    /*
    compute delta for matrix parameters following Adam algorithm
    */
    double b1_d{ 1.0 - pow(beta1, _epoch) };
    double b2_d{ 1.0 - pow(beta2, _epoch) };

    for ( size_t i = 0; i < fMomentum.size(); i++ ){
        fMomentum[i] = beta1 * fMomentum[i] + ( 1.0 - beta1 ) * (*grad[i]);
        sMomentum[i] = beta2 * sMomentum[i] + ( 1.0 - beta2 ) * (*grad[i]).array().square().matrix();
        delta[i]     = alpha * ( fMomentum[i] / b1_d ).array() / ( ( sMomentum[i] / b2_d ).array().sqrt() + eps );
        *grad[i] *= 0;
    }
}

void Optimizer::_computeDeltaAdam(vectorXf1DContainer &fMomentum, vectorXf1DContainer &sMomentum, vectorXf1DContainer &delta, vector<VectorXf*> &grad, float _epoch){
    /*
    compute delta for vector parameters following adam algorithm
    */
    double b1_d{ 1.0 - pow(beta1, _epoch) };
    double b2_d{ 1.0 - pow(beta2, _epoch) };

    for ( size_t i = 0; i < fMomentum.size(); i++ ){
        fMomentum[i] = beta1 * fMomentum[i] + ( 1.0 - beta1 ) * (*grad[i]);
        sMomentum[i] = beta2 * sMomentum[i] + ( 1.0 - beta2 ) * (*grad[i]).array().square().matrix();
        delta[i]     = alpha * ( fMomentum[i] / b1_d ).array() / ( ( sMomentum[i] / b2_d ).array().sqrt() + eps );
        *grad[i] *= 0;
    }
}

void Optimizer::_computeDeltaAdam(vector<Tensor3f> &fMomentum, vector<Tensor3f> &sMomentum, vector<Tensor3f> &delta, vector<Tensor3f*> &grad, const vector<int> &minibatch, float _epoch){
    /*
    Given minibatch, compute delta for tensor parameters following adam algorithm
    */
    float b1_d = 1.0f - pow(beta1, _epoch);
    float b2_d = 1.0f - pow(beta2, _epoch);

    for ( size_t i = 0; i < fMomentum.size(); i++ ){
      // cout << "tensor grad:\n" << *grad[i] << "\n";
      for ( int seqInd: minibatch ){
        fMomentum[i].chip(seqInd, 0) = beta1 * fMomentum[i].chip(seqInd, 0) + ( 1.0 - beta1 ) * (*grad[i]).chip(seqInd, 0);
        sMomentum[i].chip(seqInd, 0) = beta2 * sMomentum[i].chip(seqInd, 0) + ( 1.0 - beta2 ) * (*grad[i]).chip(seqInd, 0).square();
        delta[i].chip(seqInd, 0) = alpha * (fMomentum[i].chip(seqInd, 0)/b1_d) / (((sMomentum[i].chip(seqInd, 0)/b2_d).sqrt())+eps);
        (*grad[i]).chip(seqInd, 0).setZero();
      }
    }
}

void Optimizer::updateParameters(vector<int> minibatch) {
  /*
  update parameters following Adam
  */
  for (size_t i = 0; i < weight.size(); i++) {
      *weight[i] -= weightDelta[i];
      *weightGrad[i] *= 0;
  }
  for (size_t i = 0; i < bias.size(); i++) {
      *bias[i] -= biasDelta[i];
      *biasGrad[i] *= 0;
  }

  auto tsub = [](auto ta, auto tb) { ta = ta - tb; };
  auto tzero = [](auto ta, auto tb) { ta.setZero(); };
  for ( size_t i = 0; i < APtr.size(); i++ ){
      mapToTensor3(*(APtr[i]), ADelta[i], minibatch, tsub);
      mapToTensor3(*(AGradPtr[i]), ADelta[i], minibatch, tzero);
  }
}

void Optimizer::erInitialize(string erAlgorighm, float erAlpha, float erBeta1, float erBeta2){
  /*
  receive hyper parameters for ER
  */
    // if (erAlgorithm != "adam") throw invalid_argument("Optimizer: only adam is supported as erAlgorithm (received: " + erAlgorithm + ")");
    // not sure this needs to be checked here
    this->erAlgorithm = erAlgorighm;
    this->alpha = erAlpha;
    this->beta1 = erBeta1;
    this->beta2 = erBeta2;
}

void Optimizer::erRegisterParameters(vector<VectorXf*>& _erAPtr, vector<VectorXf*>& _erAGradPtr){
  /*
  receive vector of pointers of AMu and ASigma inside ER, and corresponding gradietns
  and initialize optimzier
  */
    this->erAPtr     = _erAPtr;
    this->erAGradPtr = _erAGradPtr;
    if (_erAPtr.size() != _erAGradPtr.size()) throw invalid_argument("Optimizer: erAPtr.size() != erAGradPtr.size()\n");
    erResetOptimizer();
}

void Optimizer::erComputeDelta(){
  _computeDeltaAdam(erA1stMomentum, erA2ndMomentum, erADelta, erAGradPtr, epoch);
  epoch += 1.0;
}

void Optimizer::erUpdateParameters(){
  for (size_t i = 0; i < erAPtr.size(); i++) {
      *erAPtr[i] -= erADelta[i];
      *erAGradPtr[i] *= 0;
  }
}

void Optimizer::erResetOptimizer(){
  /*
  prepare each momentum and delta quantity for A terms
  */
  erA1stMomentum.clear();
  erA2ndMomentum.clear();
  erADelta.clear();

  vectorXf1DContainer _erAZero;
  for ( size_t i = 0; i < erAPtr.size(); i++ ){
    _erAZero.push_back(VectorXf::Zero(erAPtr[i]->rows()));
  }
  erA1stMomentum = _erAZero;
  erA2ndMomentum = _erAZero;
  erADelta       = _erAZero;
  epoch = 1.0;
}
