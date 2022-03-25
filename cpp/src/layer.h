/*
This is PvrnnLayer class which is instanciated in PvrnnNetwork class.
*/


#ifndef layer_h
#define layer_h

#include <random>
#include <string>

#include "includes.h"


class PvrnnLayer {

public:
    bool   topLayer;
    int    dSize;         // # d neurons
    int    zSize;         // # z neurons
    int    inputSize;     // input dim from higher layer
    int    tau;           // time constant
    double _tau;          // 1/tau for simplyfing the computation
    double w;             // meta-prior during training
    double beta;          // this is for weighting the KLD at t=1. Do not confuse with beta1 and beta2 used in Adam optimizer
    int    seqLen;        // sequence length
    int    nSeq;          // # sequence in training dataset
    int    minibatchSize; // # sequence in one minibatch

    int epoch;            // current epoch. Mainly used for targetted debugging.
    int t;                // time step during forward computation
    vector<double> erKldSeq; // vector container to keep KLD value at each time step inside window during error regression

    default_random_engine* engine; // random number generator

    const bool vectorZeroInit;    // initialize below vectors with zeros. Set to false to initialize with random (default)

    MatrixXf Wdh;         // mapping d_{t-1} for computing h_t
    MatrixXf Wzh;         // mapping z_t for computing h_t
    MatrixXf Whdh;        // mapping d_t in higher layer for computing h_t
    VectorXf Bh;          // bias for computing h_t

    MatrixXf Wdmp;        // mapping d_{t-1} for computing \mu^p_t
    VectorXf Bmp;         // bias for computing \mu^p_t

    MatrixXf Wdsp;        // mapping d_{t-1} for computing \sigma^p_t
    VectorXf Bsp;         // bias for computing \sigma^p_t

    // for A terms to compute the posterior
    Tensor3f AMuSeq, ASigmaSeq;

    // neuron activities
    VectorXf d, z, h, u, s;
    VectorXf mup, sigmap, muq, sigmaq;
    #ifdef USE_EIGENRAND
    Rand::NormalGen<float> eps_gen{0.0f, 1.0f};
    #else
    normal_distribution<float> eps_dist{0.0f, 1.0f};
    #endif
    VectorXf eps;     // this is for the reparameterization. Do not confuse with the epsilon used in Adam optimizer
    VectorXf d0, h0;  // initial state

    // storing a sequence
    Tensor3f dSeq, zSeq, mupSeq, sigmapSeq, muqSeq, sigmaqSeq, epsSeq;

    /////////////////////////// gradients ////////////////////////////////////////////////

    double clipGradThreshold = 1.0e20; // gradient clipping. Set to 0 to disable
    double sigmaMinVal = 0.0;//1e-3;   // clip sigma within range. Set both to 0 to disable
    double sigmaMaxVal = 0.0;//1e3;

    VectorXf dLdd;
    VectorXf dLdh;
    VectorXf drdu;
    VectorXf drds;
    VectorXf drdmuq;
    VectorXf drdsigmaq;
    VectorXf dLdmuq;
    VectorXf dLdsigmaq;
    VectorXf wdLdh;     // gradient propagated to higher layer

    MatrixXf dLdWdh;
    MatrixXf dLdWhdh;
    VectorXf dLdBh;

    MatrixXf dLdWzh;

    MatrixXf dLdWdmp;
    VectorXf dLdBmp;
    MatrixXf dLdWdsp;
    VectorXf dLdBsp;

    VectorXf dLdAMu;
    VectorXf dLdASigma;
    Tensor3f dLdAMuSeq, dLdASigmaSeq; // keep gradient for each A
    //////////////////////////////////////////////////////////////////////////////////////

    size_t window, currentWindowSize;
    bool erRandomInitA;
    double erW, erBeta;
    vectorXf1DContainer erAMuSeq, erASigmaSeq;
    VectorXf dLastState, hLastState;
    bool initialWindow;
    vectorXf1DContainer erdSeq, erzSeq, erMupSeq, erSigmapSeq, erMuqSeq, erSigmaqSeq, erEpsSeq;
    vectorXf1DContainer erdLdAMuSeq, erdLdASigmaSeq;
    vectorXf1DContainer era_dLdAMuSeq, erv_dLdAMuSeq, ers_dLdAMuSeq;
    vectorXf1DContainer era_dLdASigmaSeq, erv_dLdASigmaSeq, ers_dLdASigmaSeq;
    bool gWindow;
    ///////////////////////////////////////////////////////////////////////////////////////////////

    // use this to initialize the a layer.
    // set inputSize to 0 if this is the top layer, rather than a layer which receives input
    // from higher layer in forward computation.
    PvrnnLayer(int dSize, int zSize, int inputSize, int tau, double w, double beta,
               int seqLen, int nSeq, int minibatchSize, double sigmaMinVal, double sigmaMaxVal,
               default_random_engine* engine, bool vectorZeroInit=false);

    // Forward computation
    void computeMuSigma(int seqInd);                                                       // compute \mu^p_t, \sigma^p_t, \mu^q_t, and \sigma^q_t
    void computePriorMuSigma();                                                            // only compute \mu^p and \sigma^p
    void sampleZ(const VectorXf &mu, const VectorXf &sigma);                               // sample z using the reparameterization trick
    void layerComputeMtrnn(const VectorXf &hd);                                            // MTRNN computation for layer with input from higher layer (hd)
    void topLayerForwardStep(int seqInd);                                                  // one time step forward computation in top layer
    void layerForwardStep(const VectorXf &hd, int seqInd);                                 // one time step forward computation in layer
    void topLayerPriorStep(int seqInd);                                                    // one time step prior computetion in top layer
    void layerPriorStep(const VectorXf &hd, int seqInd);                                   // one time spte prior computation in layer
    double computeKld(const VectorXf &muq, const VectorXf &sigmaq, const VectorXf &mup, const VectorXf &sigmap);       // compute KLD (not weighted)
    double computeKldSeq(const vector<int>& minibatch);                                    // sum up KLD for one minibatch
    void initializeSequence();                                                             // initialize sequence containers
    void initializeState();                                                                // initialize network state to start forward computation
    void randomizeA();                                                                     // set the A matrices to a random value.

    // Backward computation
    void zeroGradient();                                                                   // initialize all gradients to zero
    void initAGradient();                                                                  // initialize gradients for A terms
    void initBPTT();                                                                       // initialize deltas to start BPTT
    void topLayerComputeGradient(const VectorXf &wdLdh, int seqInd, int _t);               // one time step backward computation in top layer. wdLdh is error from lower layer
    void layerComputeGradient(const Tensor3f &hdSeq, const VectorXf &wdLdh, int seqInd, int _t); // one time step backward computation in layer. wdLdh is error from lower layer

    // Error regression
    void erInitialize(int window, double erW, double erBeta, bool gWindow=false,
                      bool erRandomInitA=false);                                           // setting hyper parameters for error regression
    void erSlideWindow();                                                                  // slide regression window
    void erComputeMuSigma(const VectorXf &d);                                              // compute \mu^p_t, \sigma^p_t, \mu^q_t, and \sigma^q_t
    void erTopLayerForwardStep();                                                          // one time step forward computation in top layer
    void erLayerForwardStep(const VectorXf &hd);                                           // one time step forward computation in layer
    void erKeepLastState();                                                                // keep d_0 and h_0 for forward computation in next time step (after sliding window)
    void erReset();                                                                        // initialize network state to start forward computation
    void erSeqReset();
    void erComputeGradient(const VectorXf &_wdLdh, int _t);                                // one time step backward computation
    void erInitializeGradient();                                                           // initialize gradient to start BPTT
    void erInitializeAdam();                                                               // initialize all quantities for Adam optimizer
    void erComputeAdam(int epoch);                                                         // compute Adam
    void erUpdateParameters();                                                             // update A
    void erTopLayerPriorStep();                                                            // generating one time step prediction with prior in top layer
    void erLayerPriorStep(const VectorXf &hd);                                             // generating one time step prediction with prior in layer
    double erSumKldSeq();
    void erSaveSequences(string saveFilepath, string prefix, bool prediction);             // save generated sequences during error regression
    void erSavePriorSequences(string saveFilepath, string prefix);                         // save generated sequences with prior during error regression

    // Utility
    void saveSequences(string directory);                                                  // save generated sequences
    void saveParameters(string directory);                                                 // save trained parameters
    void loadParameters(string directory, bool rowMajorW);                                 // load saved parameters
    VectorXf clipVecGradient(VectorXf grad);                                               // clipping vector form gradient
    VectorXf vecInit(Index size);                                                          // return either Random or Zero VectorXf based on vectorZeroInit
//    VectorXf replaceNanVec(VectorXf vec);
//    MatrixXf clipMatGradient(MatrixXf grad);
//    MatrixXf replaceNanMat(MatrixXf mat);
//    void clipGradient(int minibatchInd);

    // Trial: for internal analysis from here
    VectorXf _Whd;                  // weighted input from a higher layer
    vectorXf1DContainer erWhdSeq;
    vectorXf1DContainer erWTdLdhSeq; // keep weighted error propagated to a higher layer
    void erSaveWhdSeq(string saveFilepath, string prefix);
    void erSaveWTdLdhSeq(string saveFilepath, string prefix);
    // To here
protected:
    void _saveSequencesNpz(string directory);                                              // save generated sequences in NPZ format
    void _saveParametersNpz(string directory);                                             // save trained parameters in NPZ format
    void _loadParametersNpz(string directory, bool rowMajorW);                            // load saved parameters in NPZ format
private:
    // Temporary tensors for minibatch
    Tensor3f bmuqSeq;
    Tensor3f bsigmaqSeq;
    Tensor3f bmupSeq;
    Tensor3f bsigmapSeq;
    // Temporary vectors for gradient calculation
    VectorXf d_t;
    ArrayXf mu_p_t, sigma_p_t;
    ArrayXf mu_q_t, sigma_q_t;
    ArrayXf eps_t;
    VectorXf z_t;
    VectorXf hd_t;
};

#endif /* layer_h */
