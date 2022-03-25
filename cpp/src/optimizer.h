/* optimizer class. Currently, only Adam is implmented as an alorithm of optimizer.
   When implementing other algorithms, this class needs to be refactored.           */

#ifndef optimizer_h
#define optimizer_h

#include "includes.h"


class Optimizer{

public:
    Optimizer(string algorithm, float alpha, float beta1, float beta2);
    /*
    In this implementation, "delta" refers to the the quantity used to update the parameters
    */

    // general
    string              algorithm;      // name of algorithm used during training
    string              erAlgorithm;    // name of algorithm used during error regression
    int                 nMinibatch;     // # minibatches in the data set: used in the update of A

    // Adam specific
    float alpha, beta1, beta2, eps; // Adam parameters

    vector<MatrixXf*> weight;     // pointers of weight matrices
    vector<MatrixXf*> weightGrad; // pointers of gradient of weight matrices
    vector<VectorXf*> bias;       // pointers of weight biases
    vector<VectorXf*> biasGrad;   // pointers of gradient of weight biases
    vector<Tensor3f*> APtr;       // pointers of AMu and ASigma
    vector<Tensor3f*> AGradPtr;   // pointers of gradient of AMu and Asigma

    vector<VectorXf*> erAPtr;     // pointers of AMu and ASigma inside the ER window
    vector<VectorXf*> erAGradPtr; // pointers of corresponding gradients


    void registerParameters(vector<MatrixXf*>& weight, vector<MatrixXf*>& weightGrad,
                            vector<VectorXf*>& bias,   vector<VectorXf*>& biasGrad,
                            vector<Tensor3f*>& APtr,   vector<Tensor3f*>& AGrad, int nMinibatch);
    void initializeOptimizer();
    void computeDelta(vector<int> minibatch);
    void updateParameters(vector<int> minibatch);
    void erInitialize(string erAlgorighm, float erAlpha, float erBeta1, float erBeta2);
    void erRegisterParameters(vector<VectorXf*>& erAPtr, vector<VectorXf*>& erAGradPtr);
    void erComputeDelta();
    void erUpdateParameters();
    void erResetOptimizer();

  private:
    float epoch; // count of update step
    // Adam specific in training
    vector<MatrixXf>    weight1stMomentum; // 1st momentum for gradient of weight matrices
    vector<MatrixXf>    weight2ndMomentum; // 2nd momentum
    vector<MatrixXf>    weightDelta;       // delta quantity computed by Adam to update weight matrices
    vectorXf1DContainer bias1stMomentum;   // 1s momentum for gradient of weight biases
    vectorXf1DContainer bias2ndMomentum;   // 2nd momentum
    vectorXf1DContainer biasDelta;         // delta quantity computed by Adam to update weight biases
    vector<Tensor3f>    A1stMomentum;      // 1st momentum for gradient of AMu and ASigma
    vector<Tensor3f>    A2ndMomentum;      // 2nd momentum
    vector<Tensor3f>    ADelta;            // delta quantity computed by Adam to update AMu and ASigma
    // Adam specific in ER
    vectorXf1DContainer erA1stMomentum;    // 1st momentum for gradient of AMu and ASigma inside ER window
    vectorXf1DContainer erA2ndMomentum;    // 2nd momentum
    vectorXf1DContainer erADelta;          // delta quantity computed by Adam to update AMu and Asigma inside ER window

    void _initializeAdam();
    void _computeDeltaAdam(vector<MatrixXf> &fMomentum, vector<MatrixXf> &sMomentum, vector<MatrixXf> &delta, vector<MatrixXf*> &grad, float _epoch);
    void _computeDeltaAdam(vectorXf1DContainer &fMomentum, vectorXf1DContainer &sMomentum, vectorXf1DContainer &delta, vector<VectorXf*> &grad, float _epoch);
    void _computeDeltaAdam(vector<Tensor3f> &fMomentum, vector<Tensor3f> &sMomentum, vector<Tensor3f> &delta, vector<Tensor3f*> &grad, const vector<int> &minibatch, float _epoch);
};

#endif // OPTIMIZER_H_INCLUDED
