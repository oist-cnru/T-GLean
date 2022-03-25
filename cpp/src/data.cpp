#include <iostream>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <random>
#include <cmath>
#include "../cnpy/cnpy.h"

#include "data.h"


// Initialize a Data instance without loading the dataset
Data::Data(bool enableNorm, double rawMin, double rawMax, double normMin, double normMax,
           string outputLayer, int smUnit, vector<double> smSigma):
    enableNorm(enableNorm), rawMin(rawMin), rawMax(rawMax), normMin(normMin), normMax(normMax),
    outputLayer(outputLayer), smUnit(smUnit), smSigma(smSigma)
{
    if (outputLayer == "sm") {
        outputSize = rawDim * smUnit;
        erData = &erSmData;
    } else {
        outputSize = rawDim;
        erData = &erRawData;
    }
    rawData = Tensor3f();
}


Data::Data(bool enableNorm, double rawMin, double rawMax, double normMin, double normMax,
           string outputLayer, int smUnit, vector<double> smSigma,
           int nSeq, int seqLen, int minibatchSize, string rawDataPath):
    enableNorm(enableNorm), rawMin(rawMin), rawMax(rawMax), normMin(normMin), normMax(normMax),
    outputLayer(outputLayer), smUnit(smUnit), smSigma(smSigma),
    nSeq(nSeq), seqLen(seqLen), minibatchSize(minibatchSize), rawDataPath(rawDataPath)
{
    if (outputLayer == "sm") {
        outputSize = rawDim * smUnit;
        erData = &erSmData;
    } else {
        outputSize = rawDim;
        erData = &erRawData;
    }
    rawData = Tensor3f();
    if (!rawDataPath.empty()) {
        if (std::filesystem::path(rawDataPath).extension() == ".npy") {
            loadNpyData(rawDataPath);
        } else {
            cerr << "Data: File type of '" << std::filesystem::path(rawDataPath).filename() << "' not recognized, skipping load" << endl;
        }
    }
}


Data::~Data() {
    if (rawDataBuf != nullptr) free(rawDataBuf);
}


// load dataset from npy file with cnpy
void Data::loadNpyData(string npyPath) {
    cnpy::NpyArray arr;
    try {
        arr = cnpy::npy_load(npyPath);
    } catch (const exception& e){
        cerr << "Data.loadNpyData: failed to load data" << endl;
        throw;
    }
    // validate the dtype and check wordsize
    if (arr.dtype != 'f') throw invalid_argument("Expected npy data of type float, double or long double");
    rawDim = arr.shape[2]; // for some reason dimensions from config is not passed into Data, so infer here
    rawDataBuf = (float *)checkedMalloc(sizeof(float) * arr.shape[0] * arr.shape[1] * arr.shape[2]);
    // copy into persistent buffer, convert down to float if necessary
    if (arr.word_size == 4)
        copy(arr.data<float>(), arr.data<float>() + arr.shape[0] * arr.shape[1] * arr.shape[2], rawDataBuf);
    else if (arr.word_size == 8)
        copy(arr.data<double>(), arr.data<double>() + arr.shape[0] * arr.shape[1] * arr.shape[2], rawDataBuf); // convert double to float
    else if (arr.word_size == 16)
        copy(arr.data<long double>(), arr.data<long double>() + arr.shape[0] * arr.shape[1] * arr.shape[2], rawDataBuf); // convert long double to float
    else
        throw runtime_error("Unexpected npy word size " + to_string(arr.word_size));

    setRawDataC(rawDataBuf, arr.shape[0], arr.shape[1], arr.shape[2]);
}

// set raw data and post-process to compute normalized data
// `data` is a row major-ordered buffer for a 3D matrix of dimensions:
// (`n_seq`,`seq_len`,`dim`)
void Data::setRawDataC(float *data, int n_seq, int seq_len, int dim) {
    rawData = TensorMap<Tensor3f>(data, n_seq, seq_len, dim); // *data must remain valid for the duration of execution
    _postProcessRawData();
}

// set raw data and post-process to compute normalized data
// `data` is a column major-ordered buffer for a 3D matrix of dimensions:
// (`n_seq`,`seq_len`,`dim`)
void Data::setRawDataF(float *data, int n_seq, int seq_len, int dim) {
    rawData = transposeTensor3(TensorMap<Tensor3f>(data, dim, seq_len, n_seq)); // mangle dimension order for transpose
    _postProcessRawData();
}

// assumes `this->rawData` is set
void Data::_postProcessRawData() {
    // checking dimensions of rawData match what we know.
    int n_seq = rawData.dimension(0);
    if (nSeq <= 0) { nSeq = n_seq; }
    if (nSeq != n_seq) {
        throw invalid_argument("Declared number of training sequences (" + to_string(nSeq) + ") does "
                               "not match loaded data (" + to_string(n_seq) + " seq.)."); }
    int seq_len = rawData.dimension(1);
    if (seqLen <= 0) { seqLen = seq_len; }
    if (seqLen != seq_len) {
        throw invalid_argument("Declared length of training sequences (" + to_string(seqLen) + ") does "
                               "not match loaded data (length " + to_string(seq_len) + ")."); }
    int raw_dim = rawData.dimension(2);
    if (rawDim <= 0) { rawDim = raw_dim; }
    if (rawDim != raw_dim) {
        throw invalid_argument("Declared dimension of training sequences (" + to_string(rawDim) + ") does "
                               "not match loaded data (dim " + to_string(raw_dim) + ")."); }

    if (minibatchSize <= 0) { minibatchSize = nSeq; }
    if (nSeq % minibatchSize != 0) {
        throw invalid_argument("n_seq (=" + to_string(nSeq) + ") not divisible by minibatch_size (=" + to_string(minibatchSize) + ")");
    }

    outputSize = (outputLayer == "sm") ? rawDim * smUnit : rawDim;

    _normalizeData();
    if ( outputLayer == "sm" ){ _smTransformData(); }
    else                      { smUnit=1; trainData = normData; } // nothing more to do for FC layer
}

// obtain the min and max value of the raw data
void Data::_computeMinMax(){
    Tensor0f tmin = rawData.minimum();
    Tensor0f tmax = rawData.maximum();
    double raw_data_min = (double)(*tmin.data());
    double raw_data_max = (double)(*tmax.data());

    if (rawMin >= rawMax) {  // auto set and apply margin
        rawMin = raw_data_min;
        rawMax = raw_data_max;
        cout << "The raw data range is [" << raw_data_min << ", " << raw_data_max << "].\n";
    } else if (enableNorm) {  // manually set; verify bounds are consistent with data
        if (raw_data_min < rawMin) { throw invalid_argument(string("raw data actual minimum (" + to_string(raw_data_min) + ") is stricly inferior to `raw_min` value (" + to_string(rawMin) + ")")); }
        if (rawMax < raw_data_max) { throw invalid_argument(string("raw data actual maximum (" + to_string(raw_data_max) + ") is stricly superior to `raw_max` value (" + to_string(rawMax) + ")")); }
    }
}


VectorXf Data::_normalizeVec(const VectorXf &rawVec){
    /*
    this function normalize the vector to [normMin, normMax]
    Since it is sometimes hard for the network to give, with a typical activating function, extreme outputs, such as 0 and 1,
    this normalizing function normalizes the raw data with some margin. For example, when normMin=0.1 and normMax=0.9,
    the value of the normalized data takes between 0.1 and 0.9
    */
    VectorXf _normalizedVec = VectorXf( rawDim );
    /* eq d-1 */
    for ( int i = 0; i < rawDim; i++ ){
        _normalizedVec[i] = ((normMax - normMin) * (rawVec[i] - rawMin)) / (rawMax - rawMin) + normMin;
    }
    return _normalizedVec;
}

void Data::_normalizeData(){
    _computeMinMax();
    if (!enableNorm) {
        cout << "data normalization disabled. \n";
    } else {
        if (normMax <= normMin) {
            normMin = (outputLayer == "sm")? DEFAULT_NORM_MIN_SM : DEFAULT_NORM_MIN_FC;
            normMax = (outputLayer == "sm")? DEFAULT_NORM_MAX_SM : DEFAULT_NORM_MAX_FC;
        }

        cout << "normalizing data from [" << rawMin << ", " << rawMax << "] to [" << normMin << ", " << normMax << "]...";
        normData = ((float)(normMax - normMin) * (rawData - (float)rawMin)) / (float)(rawMax - rawMin) + (float)normMin;
        cout << "done\n";
    }
}

void Data::_computeScalingMatrix() {
    /* sMat is a matrix, I call scalling matrix, which may simplify the implementation of softmax transformation.
     * sMat simply tiles each element of a pre-transformed vector in softmax dimension.
     * e.g. for v = (1, 2)^T and softmax-unit/dim (`smUnit`) = 3, sMat * v = (1, 1, 1, 2, 2, 2)^T
     */
    if (rawDim < 1) {
        throw runtime_error("rawDim not configured");
    }
    _sMat = MatrixXf::Zero( rawDim * smUnit, rawDim );
    for (int dim = 0; dim < rawDim; dim++ ){
        _sMat.block(dim * smUnit, dim, smUnit, 1).setConstant(1.0);
    }
}

void Data::_computeSigmaArray() {
    // if the data dim is more than 1 and only one sigma value is provided,
    // use the same sigma value for all data dimensions
    VectorXf _sigmaVec = VectorXf( rawDim );
    if ( rawDim != 1 && smSigma.size() == 1 ){
        for ( int i = 0; i < rawDim; i++ ){
            _sigmaVec[i] = smSigma[0];
        }
        _sigmaArray = ( _sMat * _sigmaVec ).array();
    }
    else if ( rawDim == (int)smSigma.size() ){
        for ( int i = 0; i < rawDim; i++ ){
            _sigmaVec[i] = smSigma[i];
        }
        _sigmaArray = _sMat * _sigmaVec;
    }
    else {
        throw invalid_argument("The size of smSigma (" + to_string(smSigma.size()) + ") should be 1 or the same as data dimension");
    }

}

void Data::_computeRefVec() {
    // cout << "computing reference vector...";
    VectorXf _v = VectorXf::LinSpaced( smUnit, -0.1, 1.1);
    _refVec = _v.replicate(rawDim, 1);
}

VectorXf Data::_smTransformVec(const VectorXf &normVec){
    /* eq d-2, d-3 */
    ArrayXf sm = ( - ( _refVec - _sMat * normVec ).array().square() / _sigmaArray ).exp();
    Map<MatrixXf> _n(sm.data(), smUnit, rawDim);
    return sm / ( _sMat * _n.colwise().sum().transpose() ).array();
}

void Data::_smTransformData(){
    cout << "softmax-transforming data...";
    _computeScalingMatrix();
    _computeSigmaArray();
    _computeRefVec();

    trainData = Tensor3f(normData.dimension(0), normData.dimension(1), normData.dimension(2)*smUnit);
    VectorXf _seq(rawDim);
    VectorXf _smSeq;
    for ( int seqInd = 0; seqInd < nSeq; seqInd++ ){
        for ( int t = 0; t < seqLen; t++ ){
            getFromTensor3<VectorXf>(normData, _seq, seqInd, t);
            _smSeq = _smTransformVec(_seq);
            setInTensor3<VectorXf>(trainData, _smSeq, seqInd, t);
        }
    }
    cout << "done\n";
}


vector<vector<int>> Data::minibatches() {
    vector<int> seqIndexes(nSeq);
    iota(seqIndexes.begin(), seqIndexes.end(), 0);  // fill with range of increasing values
    shuffle( seqIndexes.begin(), seqIndexes.end(), default_random_engine());
    vector<vector<int>> _minibatches;
    int nMinibatch = nSeq / minibatchSize;
    for ( int i = 0; i < nMinibatch; i++ ){
        vector<int> minibatch;
        for ( int j = 0; j < minibatchSize; j++ ){
            minibatch.push_back(seqIndexes[i * minibatchSize + j]);
        }
        _minibatches.push_back(minibatch);
    }
    return _minibatches;
}



void Data::saveRawSeq(Tensor3f trainSeq, string directory, string prefix){
    if (outputLayer == "sm") {
         _saveInvTransformSeqNpz(trainSeq, directory, prefix);
    } else {
        _saveRecNormSeqNpz(trainSeq, directory, prefix);
    }
}

VectorXf Data::returnOriginal(const VectorXf &output){
  if (outputLayer == "sm") { return _reconstructNormalizedVec(_invSmTransform(output)); }
  else                     { return _reconstructNormalizedVec(output); }
}

void Data::_saveInvTransformSeqNpz(Tensor3f smSeq, string directory, string prefix) {
    std::filesystem::create_directories(directory);
    size_t smLen = smSeq.dimension(1);
    Tensor3f outSeq(nSeq, smLen, rawDim);
    _invTransformSeq(smSeq, outSeq);
    string mode = "";
    Tensor2f seq(smLen, rawDim);
    for (int seqInd = 0; seqInd < nSeq; seqInd++) {
        mode = (seqInd > 0) ? "a" : "w";  // overwrite any existing files
        getTensor2FromTensor3(outSeq, seq, seqInd);
        cnpy::npz_save(directory + "/" + prefix + ".npz", prefix + "_" + to_string(seqInd), seq.data(), {smLen, (size_t)rawDim}, mode);
    }
}

void Data::_invTransformSeq(const Tensor3f &smSeq, Tensor3f &outSeq) {
    static VectorXf smVec(rawDim * smUnit);
    static VectorXf inv(rawDim);
    for (int seqInd = 0; seqInd < nSeq; seqInd++) {
        for (int _t = 0; _t < smSeq.dimension(1); _t++) {
            getFromTensor3<VectorXf>(smSeq, smVec, seqInd, _t);
            inv.noalias() = _sMat.transpose() * ( smVec.array() * _refVec.array() ).matrix();
            setInTensor3<VectorXf>(outSeq, inv, seqInd, _t);
        }
    }
    outSeq = (outSeq - (float)normMin) * ((float)rawMax - (float)rawMin) / ((float)normMax - (float)normMin) + (float)rawMin;
}

VectorXf Data::_invSmTransform(const VectorXf &smVec) {
    return _sMat.transpose() * ( smVec.array() * _refVec.array() ).matrix();

}

// inverse of _normalizeVec
VectorXf Data::_reconstructNormalizedVec(const VectorXf &normalizedVec){
    return ( normalizedVec.array() - normMin )*( rawMax - rawMin )/( normMax - normMin ) + rawMin;
}

void Data::saveInvDataset(string directory){
    return _saveInvTransformSeqNpz(trainData, directory, "/inv_");
}

void Data::_saveRecNormSeqNpz(Tensor3f normalizedOutput, string directory, string prefix) {
    std::filesystem::create_directories(directory);
    size_t xLen = normalizedOutput.dimension(1);
    size_t xDim = rawDim;
    Tensor2f _seq(xLen, xDim);
    string mode = "";
    for (int seqInd = 0; seqInd < nSeq; seqInd++) {
        mode = (seqInd > 0) ? "a" : "w"; // overwrite any existing files
        getTensor2FromTensor3(normalizedOutput, _seq, seqInd);
        cnpy::npz_save(directory + "/" + prefix + ".npz", prefix + "_" + to_string(seqInd), _seq.data(), {xLen, xDim}, mode);
    }
}


    // Error Regression

void Data::erSetRawData(float *data, int length, int dims) {
    erRawData.clear();
    for (int t = 0; t < length; t++) {
        Map<VectorXf> row(data+(t*dims), dims);
        erRawData.push_back(row);
    }
    erSeqLen = length;

    _postProcessErData();
}

void Data::erNormalizeData(){
    for ( int t = 0; t < erSeqLen; t++ ){
        erNormData.push_back( _normalizeVec( erRawData[t] ) );
    }
}

void Data::erTransformData(){
    for ( int t = 0; t < erSeqLen; t++ ){
        erSmData.push_back( _smTransformVec( erNormData[t] ));
    }
}

void Data::_postProcessErData() {
#ifndef NDEBUG
    if (erRawData.size() < 1) throw runtime_error("erRawData not set");
#endif
    if (outputLayer == "sm") {
        erNormData.clear();
        erSmData.clear();
        erNormalizeData();
        erTransformData();
    }
}

void Data::erReset(){
    erxSeq.clear();
}

void Data::erInvTransform(const VectorXf &smVec){
    erxSeq.push_back(returnOriginal(smVec));
}

void Data::onlineErInitialize(int window){
    this->window = window;
    onlineErAnalogTarget.clear();
    onlineErSmTarget.clear();
    onlineErNormTarget.clear();
    if      (outputLayer == "fc") onlineErTarget = &onlineErNormTarget;
    else if (outputLayer == "sm") onlineErTarget = &onlineErSmTarget;
    else throw invalid_argument("outputLayer " + outputLayer + " is not valid");
}

void Data::onlineErSaveTarget(string saveFilepath){
    /*
    This function saves the current target sequence (analog value) inside the window
    */
    string mode = (std::filesystem::exists(saveFilepath) ? "a" : "w");

    Tensor2f _target = Tensor2f(onlineErAnalogTarget.size(), onlineErAnalogTarget[0].size());
    for ( size_t _t = 0; _t < onlineErAnalogTarget.size(); _t++ ){
        setInTensor2<VectorXf>(_target, onlineErAnalogTarget[_t], _t);
    }

    cnpy::npz_save(saveFilepath, "target", _target.data(), {onlineErAnalogTarget.size(), (size_t)onlineErAnalogTarget[0].size()}, mode);
}

void Data::onlineErKeepTarget(const VectorXf &analogInput) {
    /*
    This function updates the target sequence inside the window both in analog and softmax values if needed
    */

    onlineErAnalogTarget.push_back( analogInput );
    if ( outputLayer == "sm" ){
        onlineErSmTarget.push_back( _smTransformVec( _normalizeVec( analogInput ) ) );
    }
    else if ( outputLayer == "fc" ){
        onlineErNormTarget.push_back( _normalizeVec( analogInput) );
    }
    // remove the target at the first time step in the previous window
    if ( onlineErAnalogTarget.size() > window ){
        onlineErAnalogTarget.erase( onlineErAnalogTarget.begin() );
        (*onlineErTarget).erase( (*onlineErTarget).begin() );
    }
}

VectorXf Data::checkMinMax(VectorXf analogInput){
    /*
    This function checks if the new input values are within the range of raining dataset.
    If not, the outlier is replaced with the maximum or minimum value in the training dataset
    to properly transform the analog input data into a softmax value.
    */
    for ( int dim = 0; dim < rawDim; dim++ ){
        if ( analogInput[dim] > rawMax ){
            analogInput[dim] = rawMax;
        }
        if ( analogInput[dim] < rawMin ){
            analogInput[dim] = rawMin;
        }
    }
    return analogInput;
}

void Data::onlineErSaveSequence(const vectorXf1DContainer& sequence, string saveFilepath, string prefix) {
    /*
    This function receives a sequence from outputLayer, transform it to the original form:
    if outputLayer is smLayer, transforming sofmax dimension to analog dimension
    if outputLayer is fcLayer, transforming normalized data to original scale
    and save it.
    */
    string mode = (std::filesystem::exists(saveFilepath) ? "a" : "w");

    Tensor2f _output = Tensor2f(sequence.size(), sequence[0].size());
    for ( size_t _t = 0; _t < sequence.size(); _t++ ){
        setInTensor2<VectorXf>(_output, sequence[_t], _t);
    }

    cnpy::npz_save(saveFilepath, prefix + "_output", _output.data(), {sequence.size(), (size_t)sequence[0].size()}, mode);
}
