/*
This is networkConfig class which is instanciated in PvrnnNetwork class.
It reads configuration file and pass to the network class.
*/

#ifndef NETWORK_CONFIG_H_INCLUDED
#define NETWORK_CONFIG_H_INCLUDED

#include <string>
#include <vector>
#include <filesystem>
using namespace std;

struct pvrnnConfig {
    int nLayers = 0;
    vector<int> dSize;
    vector<int> zSize;
    vector<int> tau;
    vector<double> w;
    vector<double> beta;
    double sigmaMaxVal = -1.0;
    double sigmaMinVal = -1.0;

    int outputSize = -1;
    string outputLayer = "sm";
    int smUnit = -1;
    vector<double> smSigma;
    double normMin = -1.0;
    double normMax = -1.0;
    double dataMin = -1.0;
    double dataMax = -1.0;

    int seqLen = -1;
    int nSeq = -1;
    int minibatchSize = -1;
    string datasetDirectory = "";

    int nEpoch = -1;
    int saveInterval = -1;

    string optimizerAlgorithm = "";
    double alpha = -1.0;
    double beta1 = -1.0;
    double beta2 = -1.0;
    double a_eps = -1.0;
    string saveDirectory = "";

    int gWindow = -1;
    int window = -1;
    int erSeqLen = -1;
    int nInitItr = -1;
    int nItr = -1;
    int erStep = -1;
    vector<double> erW;
    vector<double> erBeta;

    bool erRandomInitA = false;
    string erOptimizerAlgorithm = "";
    double erAlpha = -1.0;
    double erBeta1 = -1.0;
    double erBeta2 = -1.0;
    string erDataDirectory = "";
    string erSaveDirectory = "";
    int epochToLoad = 0;

    int predStep = -1;
    int totalStep = -1;

    string baseDirectory = "";
    string erBaseDirectory = "";

    // branch network
    string model = ""; // only branch is supported at the moment
    vector<int> adSize, pdSize, vdSize;
    vector<int> azSize, pzSize, vzSize;
    vector<int>   aTau,   pTau,   vTau;
    vector<double>    aW,    pW,    vW;
    vector<double> aBeta, pBeta, vBeta;
    int pOutputSize = -1;
    int vOutputSize = -1;
    string pOutputLayer = "";
    string vOutputLayer = "";
    string pDatasetDirectory = "";
    string vDatasetDirectory = "";
};

class networkConfig {

public:
    networkConfig() {};
    ~networkConfig() {};

    filesystem::path trainConfigPath;
    filesystem::path batchErConfigPath;
    filesystem::path onlineErConfigPath;

    string task;

    // network configuration
    vector<int>    dSize;
    vector<int>    zSize;
    vector<int>    tau;
    vector<double> w;
    vector<double> beta;
    int            outputSize;
    int            smUnit;
    double sigmaMinVal;
    double sigmaMaxVal;

    /*branch network*/
    string model;
    string pDatasetDirectory;
    string vDatasetDirectory;
    vector<int> adSize, pdSize, vdSize;        // dSize for assoc., prop., and vision module
    vector<int> azSize, pzSize, vzSize;        // zSize for assoc., prop., and vision module
    vector<int>   aTau,   pTau,   vTau;        // tau for assoc., prop., and vision module
    vector<double>      aW,      pW,      vW;  // w for assoc., prop., and vision module
    vector<double>   aBeta,   pBeta,   vBeta;  // beta for assoc., prop., and vision module
    vector<double>    aErW,    pErW,    vErW;
    vector<double> aErBeta, pErBeta, vErBeta;
    int pOutputSize, vOutputSize;
    string pOutputLayer;
    string vOutputLayer;
    string onlineErMode;

    // dataset property
    int seqLen;
    int nSeq;
    int minibatchSize;
    int nMinibatch;
    string outputLayer;
    vector<double> smSigma;
    double normMin = 0.0;
    double normMax = 1.0;
    double dataMin = 0.0;
    double dataMax = 0.0;
    string datasetDirectory;

    // training setting
    int nEpoch;
    int saveInterval;

    string optimizerAlgorithm;
    // Adam optimizer setting
    double alpha;
    double beta1;
    double beta2;
    double a_eps;

    // directory for saving training result
    string saveDirectory;

    // error-regression setting
    int window;
    int erSeqLen;
    int nInitItr;
    int nItr;
    int erStep;
    bool erRandomInitA = false;
    vector<double> erW;
    vector<double> erBeta;
    double erAlpha;
    double erBeta1;
    double erBeta2;
    string erDataDirectory;
    string erSaveDirectory;
    string parameterDirectory;
    string erOptimizerAlgorithm;
    int _epochToLoad = -1;

    // online error regression
    int predStep;  // # time steps to make prediction after the window
    int totalStep; // # total time step to conduct online error regression
    bool gWindow;  // whether growing window is applied

    int epochToLoad(); // load and return last saved epoch value from lastEpo.txt. Fatal error if lastEpo can't be loaded
    int epochToLoad(int fallback); // try to load saved epoch value from lastEpo.txt, return fallback epoch otherwise

    void displayTrainConfig();
    void displayBatchErConfig();
    void displayOnlineErConfig();

    void importConfig(struct pvrnnConfig, bool verbose=true); // bypass the loader
};


#endif // NETWORK_CONFIG_H_INCLUDED
