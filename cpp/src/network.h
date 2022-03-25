/*
This is PvrnnNetwork class which instanciates multi-layered PV-RNN with an output layer.
*/

#ifndef network_h
#define network_h

#include <random>
#include <string>

#include "includes.h"
#include "layer.h"
#include "output_layer/output_layer.h"
#include "data.h"
#include "network_config.h"
#include "optimizer.h"


class PvrnnNetwork {

public:
    PvrnnNetwork(bool wKLD=false, int seed=-1);
    ~PvrnnNetwork();

    // Constructors for new Model interface
    PvrnnNetwork(Data* data, int seed, bool wKLD=true);
    void add_layer(int d_size, int z_size, int input_size, double tau, double w, double beta,
                   double sigma_min, double sigma_max, bool zero_init);
    void add_output_layer(bool zero_init);
    void add_optimizer(string algorithm, double alpha, double beta1, double beta2, int n_minibatch);
    void online_er_init(int epoch_to_load, int er_window, bool growing_window, int n_itr,
                        int pred_step, int total_step, string er_save_directory,
                        vector<double> er_w, vector<double> er_beta, bool er_random_init_A);
    void online_er_init_optimizer(string name, double alpha, double beta1, double beta2);
    void set_layer_w(vector<double> ws);

    /*
    Layer configuration is represented as a vector for each parameter
    The first item insider a vector is for the highest (top) layer and the last item is for the lowest (bottom) layer
    */
    double sigmaMinVal = 0.0;  // clip sigma within range. Set both to 0 to disable
    double sigmaMaxVal = 0.0;

    int   nSeq;                 // # sequences in training dataset
    int   seqLen;               // sequence length
    int   outputSize;           // target dimension before softmax operation  // FIXME: data->outputSize is *after* softmax operation.
    int   minibatchSize;        // # sequences in one minibatch

    int   nLayers = 0;          // # PV-RNN layers excluding softmax layer
    vector<int>   dSize;        // # d neurons
    vector<int>   zSize;        // # z neurons
    vector<int>   tau;          // time constant
    vector<double> w;           // meta-prior
    vector<double> beta;        // beta. Do not confuse with beta1 and beta2 in Adam optimizer

    int   smUnit;               // softmax-unit/target dimension

    int   nMinibatch;           // # minibatches in dataset

    vector<double> smSigma;     // sigma value for softmax transformation. sigma=0.05 is recommended in this implementation

    int nEpoch;                 // number of epoch to train

    default_random_engine engine; // random number generator

    int epoch = -1;             // current epoch
    string saveDirectory = "";
    int saveInterval = -1;
    bool saveRowMajorW = true; // new saves have matrices transposed, legacy saves do not
    int last_saved_epoch = -123;


    vector<PvrnnLayer*> layers;          // vector to keep pointers of each pvrnn layer instance  //QUESTION:Fabien: any reason for pointers rather than plain instances here?
    OutputLayer* outputLayer = nullptr;
    Data*             data    = nullptr; // instance dealing with dataset
    networkConfig* config;               // instance loading configuration file
    Optimizer* optimizer = nullptr;      // optimizer instance


    /*
    Pointer to each layer's parameters
    */
    vector<MatrixXf*> weight;
    vector<MatrixXf*> weightGrad;
    vector<VectorXf*> bias;
    vector<VectorXf*> biasGrad;
    vector<Tensor3f*> APtr;        // pointers of all AMu and ASigma
    vector<Tensor3f*> AGradPtr;    // pointers of corresponding gradients
    vector<VectorXf*> erAPtr;      // pointers of all AMu and ASigma inside the error-regression window
    vector<VectorXf*> erAGradPtr;  // pointers of corresponding gradients

    // error regression
    int window, nInitItr, nItr, erSeqLen; // same as layer
    vectorXf1DContainer erTarget;         // target sequence inside error regression window

    // online error regression
    int onlineErStep = 0;       // sensory time step during online error regression
    int onlineErPredStep;       // # time steps to make prediction after the window
    int onlineErTotalStep;
    bool gWindow;               // whether growing window is applied

    void initializeNetwork(bool zeroInit=false);                                 // sets parameters and initializes all layers
                                                                                 // zeroInit=true initializes vectors with zeros instead of random
    void randomizeA();                                                           // randomize the values of the A of all layers

    // forward computaion
    void forwardStep(int seqInd, int steps=1);                                   // forward computation
    void priorStep(int seqInd, int steps=1);                                     // forward computation (prior only)
    void initializeSequence();                                                   // initialize vectors to store sequences
    void initializeState();                                                      // initialize network state to start forward computation
    void forwardSequence(const vector<int>& minibatch);                          // generate one-minibatch sequences
    void testInitialize(int epoch=-1);                                           // initializes the network for test generation (load parameters without ER setup)
    void postGenAndSave(string directory);                                       // generate posterior reconstruction and save it
    void priGenAndSave(int postStep, int totalStep, string directory);           // generate prior and save it
    // FIXME: rename into priorGenAndSave

    // backward computation
    void zeroGradient();                                                         // initialize all gradients to zero
    void initAGradient();                                                        // initialize gradients of A terms
    void computeGradient(int seqInd, int _t);                                    // compute gradients at each time step
    void initBPTT();                                                             // prepare to operate BPTT
    void BPTT(const vector<int>& minibatch);                                     // operate BPTT
    void registerParameters();                                                   // register parameters to the optimizer instance

    void initTraining(int startEpoch=0);                                         // to be called before calling train on epoch
    void trainOneEpoch(double &epochErr, double &epochKld, bool backward=true);  // train one epoch
    void trainOneEpoch();                                                        // call trainOneEpoch, print out state of training, and save
    void train();                                                                // train the network following the loaded configuration,
                                                                                 // calls initTraining and trainOneEpoch repeatedly.

    // error regression
    void erInitialize(int epoch=-1);                                             // initializes the network for (batch) error regression
    void erSlideWindow();                                                        // slide regression window by one time step
    void erForwardStep();                                                        // one time step forward computation
    void erPriorStep();                                                          // one time step prior prediction
    void erForwardSequence(bool updateLastState=false);                          // forward computation inside the window
    void erPrediction(int predSteps, string saveFilepath, string prefix, bool updateLastState=true);                         // generate new prediction after error regression
    void erKeepLastState();                                                      // keep d_0 and h_0 for forward computation in next time step (after sliding window)
    void erReset();                                                              // initialize network state to start a new forward computation
    void erSeqReset();
    void erInitializeGradient();                                                 // initialize gradient
    void erInitializeAdam();                                                     // initialize Adam at each time step
    void erComputeAdam(int epoch);                                               // compute Adam
    void erUpdateParameters();                                                   // update A terms
    void erIteration(vectorXf1DContainer erTarget, vectorXf1DContainer mask, string saveFilepath, string prefix);            // one epoch of ER with optional masking, including BPTT
    void erSaveSequences(string saveFilepath, string prefix, bool prediction);   // save sequences
    void errorRegression();                                                      // start error regression
    void errorRegressionStep(int _t, vectorXf1DContainer mask, string savePath, bool verbose, bool saveIterPredictions);     // one step of error regression with optional masked rec error
    void errorRegressionStep(int _t, float *_mask, string savePath, bool verbose, bool saveIterPredictions);                 // wrapper for GPvrnn
    void errorRegressionStep(int _t, string savePath, bool verbose=true, bool saveIterPredictions=false)
       { errorRegressionStep(_t, vectorXf1DContainer(), savePath, verbose, saveIterPredictions); };                          // no mask
    void erRegisterParameters();                                                 // register parameters to the optimizer instance

    // online error regression
    void onlineErInitialize(int epoch=-1);                                       // initializes the network for online error regression
    void onlineErIteration(vectorXf1DContainer onlineErTarget, vectorXf1DContainer mask, string saveFilepath, string prefix, bool updateLastState=false);            // one epoch computation with optional dimension masking
    void onlineErSaveSequences(string saveFilepath, string prefix, bool prediction);                                          // save online ER sequences
    void onlineErPrediction(int predSteps, string saveFilepath, string prefix, bool updateLastState=true);                    // prediction generation after error regression
    VectorXf onlineErrorRegression(Map<VectorXf> &input, vectorXf1DContainer mask, string directory, bool verbose=true, bool saveIterPredictions=false);             // call this function from python interface for online error regression
    VectorXf onlineErrorRegression(Map<VectorXf> &input, string directory, bool verbose=true, bool saveIterPredictions=false)
    { return onlineErrorRegression(input, vectorXf1DContainer(), directory, verbose, saveIterPredictions); };
    void onlineErrorRegression(float *_input, float *output, float *_mask, string directory, bool verbose=false, bool saveIterPredictions=true);                     // wrapper for GPvrnn
    void onlineErrorRegression(float *_input, float *output, string directory, bool verbose=false, bool saveIterPredictions=true)
       { onlineErrorRegression(_input, output, nullptr, directory, verbose, saveIterPredictions); };

    void onlinePlanInitialize(int epoch=-1);
    void onlinePlanGeneration(Map<VectorXf> &input, vectorXf1DContainer mask, string savePath, bool dynamicStep, bool saveIter);
    void onlinePlanGeneration(float *_input, float *_mask, string savePath, bool dynamicStep, bool saveIter);
    VectorXf priorGeneration();                                                   // free generation
    VectorXf priorGeneration(Map<VectorXf> &input) { return priorGeneration(); }  // the input, it does nothing
    void priorGeneration(float *_input, float *output);                           // wrapper for library

    // utility
    double computeKldSeq(const vector<int>& minibatch);                           // compute and sum up KLD in one minibatch
    void saveTraining();                                                          // save current training epochs
    void saveSequences(string directory, bool force=false);                       // save generated sequences
    void saveParameters(string directory);                                        // save trained parameters
    void loadParameters(string directory, bool rowMajorW);                        // load saved parameters
    void loadParameters(string directory);                                        // load saved parameters, first calling loadMetadata
    void loadMetadata(string directory);                                          // load metadata about dataset
    double getKldFromLog(int layer, int step);                                    // grab raw KLD from log (timestep=-1 returns last entry in log)
    double getRecErrFromLog(int step);                                            // grab reconstruction error from log (timestep=-1 returns last entry in log)
    void getErKldFromLog(int layer, int step, double *kld);                       // get KLD per timestep for whole window at given iteration & layer
    void getErRecErrFromLog(int step, double *recerr);                            // get rec. error per timestep for whole window at given iteration
    void getErSequence(float *seq);                                               // get ER output from window
    void getErASequence(float *AMu, float *ASigma);                               // get ER A variables from window
    void getErPriorMuSigmaSequence(float *mu, float *sigma);                      // get ER prior mu and sigma values
    void getErPosteriorMuSigmaSequence(float *mu, float *sigma);                  // get ER posterior mu and sigma values
//    void clipGradient(int minibatchInd);
//    void tBPTT(int minibatchInd);                                                 // compute truncated BPTT

private:
    const bool wKLD;
    toml::table metadata;
    vector<double> _recErrLog;
    vector<vector<double>> _kldLog;
    vector<vector<double>> _recErrFullLog;                                        // save rec. err. for all timesteps in window
    vector<vector<vector<double>>> _kldFullLog;                                   // save KLD for all timesteps in window
    void _erInitialize(int epoch=-1);                                             // shared ER initialization
};

#endif /* network_h */
