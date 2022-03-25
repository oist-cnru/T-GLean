/*
This is Data class which is instantiated in PvrnnNetwork class.
In this implementation, dataset is preprocessed before softmax transformation.
That is, firstly the dataset is normalized in each dimension such that max=1.0 and min=0.0.
Then, reference vector is computed between -0.1 and 1.1, and the normalized data is transformed into softmax dimension with the reference vector.
Introducing normalization as preprocesing facilitates the choice of sigma value used in softmax transformation.
In most cases, sigma being 0.001 works well in this implementation.
*/

#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED

#include <string>

#include "includes.h"


// TODO: const attributes
class Data {
public:
    // Constructor without providing data. For norm min/max (resp. raw min/max) to be set
    // automatically, set the min and max at the same value.
    Data(bool enableNorm, double rawMin, double rawMax, double normMin, double normMax,
         string outputLayer, int smUnit, vector<double> smSigma);
    // Constructor that will also call the data loader if extension matches .npy
    Data(bool enableNorm, double rawMin, double rawMax, double normMin, double normMax,
         string outputLayer, int smUnit, vector<double> smSigma,
         int nSeq, int seqLen, int minibatchSize, string rawDataPath="");
    ~Data();

    bool enableNorm = true; // turn normalization on or off (if false, rawData == normData)
    double rawMin    = 0.0; // minimum value in the analog data
    double rawMax    = 0.0; // maximum value in the analog data
    double rawMargin = 0.0; // margin when automatically computing the raw bounds (see example.toml for details)

    double normMin   = 0.0; // minimum value of the normalized data
    double normMax   = 0.0; // maximum value of the normalized data

    string outputLayer;     // output layer: softmax layer or fully-connected layer with tanh activating function

    // softmax transform parameters
    int            smUnit;  // # softmax unit per data dimension
    vector<double> smSigma; // sigma in softmax transformation

    int nSeq   = -1;        // # training sequences
    int seqLen = -1;        // sequence length
    int rawDim = -1;        // analog data dimension
    int outputSize = -1;    // size of output, either rawDim (FC) or rawDim*smUnit (Softmax)

    int minibatchSize;      // # sequences in one minibatch

    string rawDataPath;     // path prefix of the training sequences

    // training data
    Tensor3f rawData;       // raw dataset
    Tensor3f normData;      // normalized dataset
    Tensor3f trainData;     // train data: if output layer is softmax  : softmax-transformed
                            //             if output layer is full.con.: == normData
    Tensor3f* targetData;   // this is going to be either the pointer of normData or smData; easy access to the target data regardless of the choice of output layer


    // training
    void setRawDataC(float *data, int n_seq, int seq_len, int dim);       // load dataset from a buffer (C-order input)
    void setRawDataF(float *data, int n_seq, int seq_len, int dim);       // load dataset from a buffer (F-order input)
    void loadNpyData(string npyPath);                                     // load dataset from npy file with cnpy

    vector<vector<int>> minibatches();                                    // provides the sequence indexes of random minibatches.

    void saveRawSeq(Tensor3f smSeq, string directory, string prefix="x"); // inverse transform sequences and save them to disk
    void saveInvDataset(string directory);                                // inverse-transform dataset to check the quality of softmax transformation (sm-only?)

    VectorXf returnOriginal(const VectorXf &output);

    // to extract data toward python
    float* getRawData()   { return rawData.data(); }
    float* getNormData()  { return normData.data(); }
    float* getTrainData() { return trainData.data(); }

protected:
    friend class PvrnnNetwork;                                      // let the parent network directly manipulate the data
    friend class SoftmaxLayer;                                      // allow access to softmax.....
    float* rawDataBuf = nullptr;                                    // holds data read by loadNpyData, and is mapped to rawData
    // post-processing training dataset
    void _postProcessRawData();                                     // do all transformation of data after loading the dataset
    void _postProcessErData();                                      // same thing but for ER data
    // normalization
    void _computeMinMax();                                          // compute min and max value of the raw data
    VectorXf _normalizeVec(const VectorXf &rawVec);                 // normalize vector. Note that this normalization is not one making the norm of a vector 1.
    void _normalizeData();                                          // normalize the entire dataset
    // softmax transformation
    VectorXf _refVec;                                               // reference vector
    MatrixXf _sMat;                                                 // scaling matrix. See also SoftmaxLayer
    void _computeRefVec();                                          // the reference vector is right now just a linear spaced vector repeated
    void _computeScalingMatrix();                                   // the scaling matrix is a map of raw dim to softmax dims
    void _computeSigmaArray();                                      // the sigma array is either set by dimension or all the same
    VectorXf _smTransformVec(const VectorXf &normVec);              // operate softmax transformation
    ArrayXf _sigmaArray;                                            // array filled with sigma values
    void _smTransformData();                                        // softmax-transform the normalized dataset

    // saving training outputs
    void _saveInvTransformSeqNpz(Tensor3f smSeq, string directory, string prefix);   // inverse softmax sequences into raw ones and save them as .npz
    void _invTransformSeq(const Tensor3f &smSeq, Tensor3f &outSeq);                  // inverse all the softmax sequences in a tensor FIXME: unprotect
    VectorXf _invSmTransform(const VectorXf &smVec);                                 // inverse a softmax vector, return a normalized vector
    void _saveRecNormSeqNpz(Tensor3f normSeq, string directory, string prefix);      // inverse normalized sequences into raw ones and save them as .npz
    VectorXf _reconstructNormalizedVec(const VectorXf &normalizedVec);               // inverse a normalized vector



// unsorted (public/private) functions and attributes. TODO: sort.
public:

    // error regression
    int erSeqLen;
    size_t window;

    vectorXf1DContainer erRawData;     // raw sequence for error regression
    vectorXf1DContainer erNormData;    // normalized sequence
    vectorXf1DContainer erSmData;      // softmax transformed sequence
    vectorXf1DContainer* erData;       // either erRawData (fc output) or erSmData (sm output)
    vectorXf1DContainer erxSeq;        // inverse-softmax-transformed sequence

    // online error regression
    vectorXf1DContainer onlineErAnalogTarget;
    vectorXf1DContainer onlineErNormTarget;
    vectorXf1DContainer onlineErSmTarget;
    vectorXf1DContainer onlineErAnalogOutput;
    vectorXf1DContainer* onlineErTarget;

    // error regression
    void erSetRawData(float *data, int length, int dims);               // load data for a single ER sequence from a buffer
    void erNormalizeData();                                             // normalize the loaded sequence
    void erTransformData();                                             // softmax-transform the normalized sequence
    void erReset();                                                     // clear the vector container to store the inverse-transformed sequence
    void erInvTransform(const VectorXf &smVec);                         // inverse-transform the network output

    // online error regression
    void onlineErInitialize(int window);                                // initialization for online error regression
    void onlineErSaveTarget(string saveFilepath);                       // save the target sequence for the current window
    void onlineErKeepTarget(const VectorXf &analogInput);               // update the target sequence inside the window (both analog and softmax)
    void onlineErSaveSequence(const vectorXf1DContainer& seq, string saveFilepath, string prefix);
    VectorXf checkMinMax(VectorXf analogInput);                         // check if the analog input lies within the training dataset range, replace if not

};

#endif // DATA_H_INCLUDED
