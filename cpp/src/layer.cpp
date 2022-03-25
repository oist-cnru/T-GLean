
#include <iostream>
#include <fstream>
#include <filesystem>

#include "layer.h"


PvrnnLayer::PvrnnLayer(int dSize, int zSize, int inputSize, int tau, double w, double beta,
                       int seqLen, int nSeq, int minibatchSize, double sigmaMinVal, double sigmaMaxVal,
                       default_random_engine* engine, bool vectorZeroInit)
    : dSize{dSize}, zSize{zSize}, tau{tau}, _tau{1.0/tau}, w{w}, beta{beta}, seqLen{seqLen},
      nSeq{nSeq}, minibatchSize{minibatchSize}, engine(engine), vectorZeroInit(vectorZeroInit),
      sigmaMinVal{sigmaMinVal}, sigmaMaxVal{sigmaMaxVal}
{
    if (minibatchSize < 1) {
        throw runtime_error("minibatchSize is not set");
    }
    this->topLayer = inputSize == 0;
    epoch = 0;
    eps = VectorXf(zSize);

    // initializing weight matrices and biases
    Wdh  = glorotNormalInit(dSize, dSize, engine);
    Wzh  = glorotNormalInit(dSize, zSize, engine);
    Bh   = vecInit(dSize);
    Wdmp = glorotNormalInit(zSize, dSize, engine);
    Bmp  = vecInit(zSize);
    Wdsp = glorotNormalInit(zSize, dSize, engine);
    Bsp  = vecInit(zSize);

    if ( !topLayer ) {
        this->inputSize = inputSize;
        Whdh = glorotNormalInit(dSize, inputSize, engine);
    }

    // initialize A terms
    muq = VectorXf(zSize);
    sigmaq = VectorXf(zSize);
    AMuSeq = Tensor3f(nSeq, seqLen, zSize);
    ASigmaSeq = Tensor3f(nSeq, seqLen, zSize);
    if (vectorZeroInit) {
        AMuSeq.setZero();
        ASigmaSeq.setZero();
    } else {
        randomizeA();
    }

    // initial state
    d0 = VectorXf::Zero(dSize);
    h0 = VectorXf::Zero(dSize);
    d = d0;
    h = h0;

    d_t = VectorXf(dSize);
    mu_p_t = ArrayXf(zSize);
    sigma_p_t = ArrayXf(zSize);
    mu_q_t = ArrayXf(zSize);
    sigma_q_t = ArrayXf(zSize);
    eps_t = ArrayXf(zSize);
    z_t = VectorXf(zSize);
    bmuqSeq = Tensor3f(minibatchSize, seqLen, zSize);
    bsigmaqSeq = Tensor3f(minibatchSize, seqLen, zSize);
    bmupSeq = Tensor3f(minibatchSize, seqLen, zSize);
    bsigmapSeq = Tensor3f(minibatchSize, seqLen, zSize);

    zeroGradient();
    initAGradient();
}

void PvrnnLayer::randomizeA() {
    #ifdef EIGEN_REPRODUCIBLE  // NOTE: once Eigen 3.4 is out, setting srand is another way.
    static normal_distribution<double> n_dist(0,1);
    for (int i = 0; i < AMuSeq.dimension(0); i++) {
        MatrixXf m_mu = MatrixXf::Zero(AMuSeq.dimension(1),AMuSeq.dimension(2)).unaryExpr([&](float dummy){return float(n_dist(*engine));});
        setInTensor3<MatrixXf>(AMuSeq, m_mu, i);
        MatrixXf m_sigma = MatrixXd::Zero(ASigmaSeq.dimension(1),ASigmaSeq.dimension(2)).unaryExpr([&](float dummy){return float(n_dist(*engine));});
        setInTensor3<MatrixXf>(ASigmaSeq, m_sigma, i);
    }
    #else
    AMuSeq.setRandom<Eigen::internal::NormalRandomGenerator<float>>();
    ASigmaSeq.setRandom<Eigen::internal::NormalRandomGenerator<float>>();
    #endif
}

void PvrnnLayer::computeMuSigma(int seqInd) {
    /*
    compute mu and sigma
    */
    // at t=1, z^p is standard normal dist
    if ( t == 0 ){
        /* eq f-5, f-6 */
        mup = VectorXf::Zero(zSize);
        sigmap = VectorXf(zSize);
        sigmap.fill(1.0f);
    }
    // otherwise, z^p is mapped from d
    else {
        /* eq f-6, f-8, f-11, f-12 */
        mup    = ( Wdmp * d + Bmp ).array().tanh();
        sigmap = ( Wdsp * d + Bsp ).array().exp();
    }
    if (sigmaMinVal != 0.0) { sigmap = sigmap.cwiseMax(sigmaMinVal); }  // FIXME: if (min < max)
    if (sigmaMaxVal != 0.0) { sigmap = sigmap.cwiseMin(sigmaMaxVal); }

    // compute posterior mu and sigma
    if (seqInd >= 0) {
        getFromTensor3<VectorXf>(AMuSeq, muq, seqInd, t);
        getFromTensor3<VectorXf>(ASigmaSeq, sigmaq, seqInd, t);
        /* eq f-11, f-12 */
        muq    = muq.array().tanh();
        sigmaq = sigmaq.array().exp();
        if (sigmaMinVal != 0.0) { sigmaq = sigmaq.cwiseMax(sigmaMinVal); }  // FIXME: if (min < max)
        if (sigmaMaxVal != 0.0) { sigmaq = sigmaq.cwiseMin(sigmaMaxVal); }
    }
}

void PvrnnLayer::computePriorMuSigma(){
    return computeMuSigma(-1);
}

void PvrnnLayer::sampleZ(const VectorXf &mu, const VectorXf &sigma){
    /*
    reparametrization: sample noise and compute z given mu and sigma
    */
    #ifdef USE_EIGENRAND
    eps = eps_gen.generate<VectorXf>(zSize, 1, *engine);
    #else
    for (int i = 0; i < zSize; i++) eps(i) = eps_dist(*engine);
    #endif
    /* eq f-13 */
    z = mu + (sigma.array() * eps.array()).matrix();
}

void PvrnnLayer::layerComputeMtrnn(const VectorXf &hd){
    /*
    compute hidden state based on MTRNN
    */
    /* eq f-1, f-14 */
    if (!topLayer) {
        _Whd.noalias() = Whdh * hd;
        h = ( 1.0f - 1.0f / tau ) * h + (Wdh * d + Wzh * z + _Whd + Bh) / tau;
    } else {
        h = ( 1.0f - 1.0f / tau ) * h + (Wdh * d + Wzh * z + Bh) / tau;
    }
    /* eq f-2 */
    d = h.array().tanh();
}

double PvrnnLayer::computeKld(const VectorXf &muq, const VectorXf &sigmaq, const VectorXf &mup, const VectorXf &sigmap){
    /*
    compute KLD given parameters of two gaussian distributions
    */
    // e-2 and e-3, but w or beta is not applied here
    return (double)(( clippedLog(sigmap.array() / sigmaq.array()).array() +
           0.5 * ( ( mup - muq ).array().square() + sigmaq.array().square() ) / sigmap.array().square() - 0.5 ).sum()) / zSize;
}

double PvrnnLayer::computeKldSeq(const vector<int>& minibatch){
    /*
    sum up KLD for a minibatch
    */
    gatherFromTensor3(muqSeq, bmuqSeq, minibatch);
    gatherFromTensor3(sigmaqSeq, bsigmaqSeq, minibatch);
    gatherFromTensor3(mupSeq, bmupSeq, minibatch);
    gatherFromTensor3(sigmapSeq, bsigmapSeq, minibatch);
    Tensor0f _kld = (clippedLog(bsigmapSeq / bsigmaqSeq)
                             + 0.5f * ((bmupSeq - bmuqSeq).square()
                                       + bsigmaqSeq.square()) / bsigmapSeq.square() - 0.5f).sum();

    // return the normalized KLD
    return (double)(*_kld.data()) / (zSize * minibatchSize * seqLen);
}

void PvrnnLayer::initializeSequence(){
    /*
    initialize tensors to hold network activities
    called every epoch during training
    */
    dSeq      = Tensor3f(nSeq, seqLen+1, dSize);
    zSeq      = Tensor3f(nSeq, seqLen, zSize);
    mupSeq    = Tensor3f(nSeq, seqLen, zSize);
    sigmapSeq = Tensor3f(nSeq, seqLen, zSize);
    muqSeq    = Tensor3f(nSeq, seqLen, zSize);
    sigmaqSeq = Tensor3f(nSeq, seqLen, zSize);
    epsSeq    = Tensor3f(nSeq, seqLen, zSize);

    if (!d0.isZero())
        for (int i = 0; i < nSeq; i++) setInTensor3<VectorXf>(dSeq, d0, i, 0); // currently unused
    else dSeq.setZero();
    zSeq.setZero();
    mupSeq.setZero();
    sigmapSeq.setZero();
    muqSeq.setZero();
    sigmaqSeq.setZero();
    epsSeq.setZero();
}

void PvrnnLayer::initializeState(){
    /*
    initialize network state
    called every time generating a sequence during training
    */
    t = 0;
    d = VectorXf::Zero( dSize );
    h = VectorXf::Zero( dSize );
}

void PvrnnLayer::layerForwardStep(const VectorXf &hd, int seqInd){
    /*
    one time step forward computation with posterior
    */
    // forward computation
    computeMuSigma(seqInd );
    sampleZ( muq, sigmaq );
    layerComputeMtrnn( hd );
    // store generated data
    setInTensor3<VectorXf>(dSeq, d, seqInd, t+1);
    setInTensor3<VectorXf>(zSeq, z, seqInd, t);
    setInTensor3<VectorXf>(mupSeq, mup, seqInd, t);
    setInTensor3<VectorXf>(sigmapSeq, sigmap, seqInd, t);
    setInTensor3<VectorXf>(muqSeq, muq, seqInd, t);
    setInTensor3<VectorXf>(sigmaqSeq, sigmaq, seqInd, t);
    setInTensor3<VectorXf>(epsSeq, eps, seqInd, t);
    t++;
}

void PvrnnLayer::topLayerForwardStep(int seqInd){
    static VectorXf dummy_hd;
    return layerForwardStep(dummy_hd, seqInd);
}

void PvrnnLayer::layerPriorStep(const VectorXf &hd, int seqInd){
    /*
    one time step forward computation with prior
    */
    computePriorMuSigma();
    sampleZ( mup, sigmap );
    layerComputeMtrnn( hd );
    setInTensor3<VectorXf>(dSeq, d, seqInd, t+1);
    setInTensor3<VectorXf>(zSeq, z, seqInd, t);
    setInTensor3<VectorXf>(mupSeq, mup, seqInd, t);
    setInTensor3<VectorXf>(sigmapSeq, sigmap, seqInd, t);
    setInTensor3<VectorXf>(epsSeq, eps, seqInd, t);
    t++;
}

void PvrnnLayer::topLayerPriorStep(int seqInd){
    static VectorXf dummy_hd;
    return layerPriorStep(dummy_hd, seqInd);
}


void PvrnnLayer::initAGradient(){
    /*
    initialize gradients for AMu and ASigma
    */
    dLdAMuSeq = Tensor3f(nSeq, seqLen, zSize);
    dLdAMuSeq.setZero();
    dLdASigmaSeq = Tensor3f(nSeq, seqLen, zSize);
    dLdASigmaSeq.setZero();
}

void PvrnnLayer::zeroGradient(){
    /*
    initialize gradients for weight and bias
    */
    dLdWdh  = MatrixXf::Zero(dSize, dSize);
    dLdWzh  = MatrixXf::Zero(dSize, zSize);
    dLdBh   = VectorXf::Zero(dSize);

    dLdWdmp = MatrixXf::Zero(zSize, dSize);
    dLdBmp  = VectorXf::Zero(zSize);
    dLdWdsp = MatrixXf::Zero(zSize, dSize);
    dLdBsp  = VectorXf::Zero(zSize);

    dLdAMu = VectorXf::Zero(zSize);
    dLdASigma = VectorXf::Zero(zSize);

    if (!topLayer){
        dLdWhdh = MatrixXf::Zero(dSize, inputSize);
    }
}

void PvrnnLayer::initBPTT(){
    /*
    initialize gradient for hidden states
    called before computing BPTT
    */
    dLdh = VectorXf::Zero(dSize);
    drdu = VectorXf::Zero(zSize);
    drds = VectorXf::Zero(zSize);
    dLdd = VectorXf::Zero(dSize);
}

void PvrnnLayer::topLayerComputeGradient(const VectorXf &_wdLdh, int seqInd, int _t){
    Tensor3f dummy_hdSeq;
    return layerComputeGradient(dummy_hdSeq, _wdLdh, seqInd, _t);
}

void PvrnnLayer::layerComputeGradient(const Tensor3f &hdSeq, const VectorXf &_wdLdh, int seqInd, int _t){
    /*
    one time step gradient computation
    */
    // d is one time step ahead because of d_0
    getFromTensor3<VectorXf>(dSeq, d_t, seqInd, _t+1);
    // compute each gradient
    /* eq b-1, b-23 */
    dLdd = _wdLdh + _tau * Wdh.transpose() * dLdh + Wdmp.transpose() * drdu + Wdsp.transpose() * drds;
    /* eq  b-2 */
    dLdh = ( 1.0f - _tau ) * dLdh
            + ( dLdd.array() * ( 1.0f - d_t.array().square() /* this is d_{t+1} */ ) ).matrix();
    dLdh = (clipGradThreshold != 0.0) ? clipVecGradient( dLdh ) : dLdh;

    getFromTensor3<VectorXf>(dSeq, d_t, seqInd, _t);
    getFromTensor3<ArrayXf>(mupSeq, mu_p_t, seqInd, _t);
    getFromTensor3<ArrayXf>(sigmapSeq, sigma_p_t, seqInd, _t);
    getFromTensor3<ArrayXf>(muqSeq, mu_q_t, seqInd, _t);
    getFromTensor3<ArrayXf>(sigmaqSeq, sigma_q_t, seqInd, _t);
    getFromTensor3<ArrayXf>(epsSeq, eps_t, seqInd, _t);
    getFromTensor3<VectorXf>(zSeq, z_t, seqInd, _t);

    // normalize KLD gradient w.r.t posterior
    /* common part in eq b-7 and b-9 */
    drdmuq    = ((mu_q_t - mu_p_t) / sigma_p_t.square()) / minibatchSize;
    /* common part in eq b-8 and b-10 */
    drdsigmaq = ((- 1.0f / sigma_q_t) + (sigma_q_t / sigma_p_t.square())) / minibatchSize;

    // apply beta at t=0
    if ( _t == 0 ){
        /* eq b-7 */
        drdmuq    *= (beta / zSize);
        /* eq b-8 */
        drdsigmaq *= (beta / zSize);
    }
    // apply w otherwise
    else {
        /* eq b-9 */
        drdmuq    *= (w / zSize);
        /* eq b-10 */
        drdsigmaq *= (w / zSize);
    }

    /* eq b-5 */
    dLdmuq    = drdmuq + _tau * Wzh.transpose() * dLdh;
    /* eq b-6 */
    dLdsigmaq = drdsigmaq + _tau * (( Wzh.transpose() * dLdh ).array() * eps_t).matrix();

    // add each gradient for parameter update
    /* eq b-11 */
    dLdWdh   += _tau * dLdh * d_t.transpose();
    /* eq b-12 */
    dLdWzh   += _tau * dLdh * z_t.transpose();
    if (!topLayer) {
        hd_t = VectorXf(hdSeq.dimension(2));
        getFromTensor3<VectorXf>(hdSeq, hd_t, seqInd, _t);
        /* eq b-21 */
        dLdWhdh += _tau * dLdh * hd_t.transpose();
    }
    /* eq b-13 */
    dLdBh   += _tau * dLdh;

    if ( _t != 0 ){
        /* eq b-14 */
        dLdWdmp += drdu * d_t.transpose();
        /* eq b-15 */
        dLdBmp  += drdu;
        /* eq b-16 */
        dLdWdsp += drds * d_t.transpose();
        /* eq b-17 */
        dLdBsp  += drds;
    }

    getFromTensor3<VectorXf>(dLdAMuSeq, dLdAMu, seqInd, _t);
    getFromTensor3<VectorXf>(dLdASigmaSeq, dLdASigma, seqInd, _t);
    /* eq b-18 */
    dLdAMu    += (dLdmuq.array() * (1.0f - mu_q_t.square())).matrix();
    /* eq b-19 */
    dLdASigma += (dLdsigmaq.array() * sigma_q_t).matrix();
    setInTensor3<VectorXf>(dLdAMuSeq, dLdAMu, seqInd, _t);
    setInTensor3<VectorXf>(dLdASigmaSeq, dLdASigma, seqInd, _t);

    // normalize KLD gradient w.r.t prior
    /* eq b-3 */
    drdu = ( w / zSize ) * (((mu_p_t - mu_q_t) * (1.0f - mu_p_t.square())) / sigma_p_t.square()) / minibatchSize;
    /* eq b-4 */
    drds = ( w / zSize ) * ( 1.0f - ( (mu_p_t - mu_q_t).square() + sigma_q_t.square() ) / sigma_p_t.square() ) / minibatchSize;
    if (!topLayer) { wdLdh.noalias() = _tau * Whdh.transpose() * dLdh; }
}


void PvrnnLayer::erInitialize(int window, double erW, double erBeta, bool gWindow, bool erRandomInitA) {
    this->window  = window;
    this->erW     = erW;
    this->erBeta  = erBeta;
    this->gWindow = gWindow;
    this->erRandomInitA = erRandomInitA;

    erAMuSeq.clear();
    erASigmaSeq.clear();
    erdLdAMuSeq.clear();
    erdLdASigmaSeq.clear();

    currentWindowSize = gWindow ? 1 : window;
    for (size_t _t = 0; _t < currentWindowSize; _t++ ){
        erAMuSeq.push_back( vecInit( zSize ) );
        erASigmaSeq.push_back( vecInit( zSize ) );
        erdLdAMuSeq.push_back( VectorXf::Zero( zSize ) );
        erdLdASigmaSeq.push_back( VectorXf::Zero( zSize ) );
    }

    dLastState = VectorXf::Zero(dSize);
    hLastState = VectorXf::Zero(dSize);
    t = 0;
    initialWindow = true;
}

void PvrnnLayer::erSlideWindow(){
    if ( erAMuSeq.size() == window ){
        erAMuSeq.erase( erAMuSeq.begin() );
        erASigmaSeq.erase( erASigmaSeq.begin() );
        erdLdAMuSeq.erase( erdLdAMuSeq.begin() );
        erdLdASigmaSeq.erase( erdLdASigmaSeq.begin() );
        initialWindow = false;
    }
    else{
        currentWindowSize++;
        initialWindow = true;
    }
    if (erRandomInitA) {  // initializing the A values that just got into the window randomly.
      erAMuSeq.push_back( vecInit( zSize ) );
      erASigmaSeq.push_back( vecInit( zSize ) );
    } else {  // intializing them after the A values of the previous timestep.
      erAMuSeq.push_back( erAMuSeq.back() );
      erASigmaSeq.push_back( erASigmaSeq.back() );
    }
    erdLdAMuSeq.push_back( VectorXf::Zero( zSize ) );
    erdLdASigmaSeq.push_back( VectorXf::Zero( zSize) );
}

void PvrnnLayer::erComputeMuSigma(const VectorXf &d){
    // at t=0 (in sensory time step), z^p is unit gaussian
    if ( initialWindow && t == 0 ){
        mup    = VectorXf::Zero(zSize);
        sigmap = VectorXf::Ones(zSize);
    }
    // otherwise, z^p is mapped from d
    else{
        mup        = ( Wdmp * d + Bmp ).array().tanh();
        if (sigmaMinVal != 0.0 && sigmaMaxVal != 0.0) {  // FIXME: should be if (sigmaMinVal < sigmaMaxVal)
            sigmap = ( Wdsp * d + Bsp ).array().cwiseMax(sigmaMinVal).cwiseMin(sigmaMaxVal).exp();
        } else {
            sigmap = ( Wdsp * d + Bsp ).array().exp();
        }
    }
    if ( t < (int)window ){
        muq        = erAMuSeq[t].array().tanh();
        if (sigmaMinVal != 0.0 && sigmaMaxVal != 0.0) {
            sigmaq = erASigmaSeq[t].array().cwiseMax(sigmaMinVal).cwiseMin(sigmaMaxVal).exp();
        } else {
            sigmaq = erASigmaSeq[t].array().exp();
        }
    }
}

void PvrnnLayer::erTopLayerForwardStep(){
    static VectorXf dummy_hd;
    return erLayerForwardStep(dummy_hd);
}

void PvrnnLayer::erLayerForwardStep(const VectorXf &hd){
    erComputeMuSigma( d );
    sampleZ(muq, sigmaq);
    layerComputeMtrnn( hd );
    t++;

    erdSeq.push_back( d );
    erzSeq.push_back( z );
    erMupSeq.push_back( mup );
    erMuqSeq.push_back( muq );
    erSigmapSeq.push_back( sigmap );
    erSigmaqSeq.push_back( sigmaq );
    erEpsSeq.push_back( eps );
    erKldSeq.push_back( computeKld(muq, sigmaq, mup, sigmap) );
    if (!topLayer) erWhdSeq.push_back( _Whd );
}

void PvrnnLayer::erTopLayerPriorStep(){
    static VectorXf dummy_hd;
    return erLayerPriorStep(dummy_hd);
}

void PvrnnLayer::erLayerPriorStep(const VectorXf &hd){
    computePriorMuSigma();
    sampleZ( mup, sigmap );
    layerComputeMtrnn( hd );
    t++;

    erdSeq.push_back( d );
    erzSeq.push_back( z );
    erMupSeq.push_back( mup );
    erSigmapSeq.push_back( sigmap );
    erEpsSeq.push_back( eps );
    if (!topLayer) erWhdSeq.push_back( _Whd );
}

void PvrnnLayer::erKeepLastState(){
    dLastState = d;
    hLastState = h;
}

void PvrnnLayer::erReset(){
    d = dLastState;
    h = hLastState;
    t = 0;

    erSeqReset();
}

void PvrnnLayer::erSeqReset(){
    erdSeq.clear();
    erdSeq.push_back( dLastState );
    erzSeq.clear();
    erMupSeq.clear();
    erSigmapSeq.clear();
    erMuqSeq.clear();
    erSigmaqSeq.clear();
    erEpsSeq.clear();
    erKldSeq.clear();
    erWhdSeq.clear();
    erWTdLdhSeq.clear();
}

void PvrnnLayer::erComputeGradient(const VectorXf &_wdLdh, int _t){
    // d is one time step slided because of d_0
    dLdd = _wdLdh + _tau * Wdh.transpose() * dLdh + Wdmp.transpose() * drdu + Wdsp.transpose() * drds;
    dLdh = ( 1.0f - _tau ) * dLdh + ( dLdd.array() * ( 1.0f - erdSeq[_t+1].array().square() )).matrix();

    // shortened notation
    mu_p_t = erMupSeq[_t].array();
    mu_q_t = erMuqSeq[_t].array();
    sigma_p_t = erSigmapSeq[_t].array();
    sigma_q_t = erSigmaqSeq[_t].array();

    // normalize KLD gradient
    drdmuq    = (( mu_q_t - mu_p_t ) / sigma_p_t.square()) / currentWindowSize;
    drdsigmaq = (( - 1.0f / sigma_q_t ) + ( sigma_q_t )/ sigma_p_t.square()) / currentWindowSize;

    // apply beta
    if ( initialWindow && _t == 0 ){
        drdmuq    *= (erBeta/zSize);
        drdsigmaq *= (erBeta/zSize);
    }
    // apply w
    else {
        drdmuq    *= (erW/zSize);
        drdsigmaq *= (erW/zSize);
    }

    dLdmuq    = drdmuq + _tau * Wzh.transpose() * dLdh;
    dLdsigmaq = drdsigmaq + (( _tau * Wzh.transpose() * dLdh ).array() * erEpsSeq[_t].array()).matrix();

    erdLdAMuSeq[_t]    = ( dLdmuq.array() * ( 1.0f - mu_q_t.square() )).matrix();
    erdLdASigmaSeq[_t] = ( dLdsigmaq.array() * sigma_q_t ).matrix();

    // normalize KLD
    drdu = ( erW / zSize ) * (( (mu_p_t - mu_q_t) * ( 1.0f - mu_p_t.square())) / sigma_p_t.square() ) / currentWindowSize;
    drds = ( erW / zSize ) * ( 1.0f - ( (mu_p_t - mu_q_t).square() + sigma_q_t.square() ) / sigma_p_t.square() ) / currentWindowSize;
    if ( !topLayer ){
        wdLdh.noalias() = Whdh.transpose() * dLdh;
        erWTdLdhSeq.push_back( wdLdh );
    }
}

void PvrnnLayer::erInitializeGradient(){
    dLdh = VectorXf::Zero(dSize);
    drdu = VectorXf::Zero(zSize);
    drds = VectorXf::Zero(zSize);
}

double PvrnnLayer::erSumKldSeq(){
  return accumulate(erKldSeq.begin(), erKldSeq.end(), 0.0) / erKldSeq.size();
}

void PvrnnLayer::saveSequences(string directory){
    _saveSequencesNpz(directory);
}

void PvrnnLayer::erSaveSequences(string saveFilepath, string prefix, bool prediction) {
    string mode = (std::filesystem::exists(saveFilepath) ? "a" : "w");

    /* Create temporary tensors for storage (can this whole pipeline be tensorized??) */
    size_t pWindowSize = (prediction ? erMupSeq.size() : currentWindowSize);
    Tensor2f _d = Tensor2f(pWindowSize, erdSeq[0].size());
    Tensor2f _mup = Tensor2f(pWindowSize, erMupSeq[0].size());
    Tensor2f _sigmap = Tensor2f(pWindowSize, erSigmapSeq[0].size());
    Tensor2f _muq = Tensor2f(currentWindowSize, erMuqSeq[0].size());
    Tensor2f _sigmaq = Tensor2f(currentWindowSize, erSigmaqSeq[0].size());
    Tensor2f _AMu = Tensor2f(currentWindowSize, erAMuSeq[0].size());
    Tensor2f _ASigma = Tensor2f(currentWindowSize, erASigmaSeq[0].size());

    for (size_t _t = 0; _t < currentWindowSize; _t++) {
        setInTensor2<VectorXf>(_d, erdSeq[_t+1], _t);
        setInTensor2<VectorXf>(_mup, erMupSeq[_t], _t);
        setInTensor2<VectorXf>(_sigmap, erSigmapSeq[_t], _t);
        setInTensor2<VectorXf>(_muq, erMuqSeq[_t], _t);
        setInTensor2<VectorXf>(_sigmaq, erSigmaqSeq[_t], _t);
        setInTensor2<VectorXf>(_AMu, erAMuSeq[_t], _t);
        setInTensor2<VectorXf>(_ASigma, erASigmaSeq[_t], _t);
    }
    if (prediction) {
        for (size_t _t = currentWindowSize; _t < erMupSeq.size(); _t++) {
            setInTensor2<VectorXf>(_d, erdSeq[_t+1], _t);
            setInTensor2<VectorXf>(_mup, erMupSeq[_t], _t);
            setInTensor2<VectorXf>(_sigmap, erSigmapSeq[_t], _t);
        }
    }

    cnpy::npz_save(saveFilepath, prefix + "_d", _d.data(), {pWindowSize, (size_t)erdSeq[0].size()}, mode);
    cnpy::npz_save(saveFilepath, prefix + "_mup", _mup.data(), {pWindowSize, (size_t)erMupSeq[0].size()}, "a");
    cnpy::npz_save(saveFilepath, prefix + "_sigmap", _sigmap.data(), {pWindowSize, (size_t)erSigmapSeq[0].size()}, "a");
    cnpy::npz_save(saveFilepath, prefix + "_muq", _muq.data(), {currentWindowSize, (size_t)erMuqSeq[0].size()}, "a");
    cnpy::npz_save(saveFilepath, prefix + "_sigmaq", _sigmaq.data(), {currentWindowSize, (size_t)erSigmaqSeq[0].size()}, "a");
    cnpy::npz_save(saveFilepath, prefix + "_AMu", _AMu.data(), {currentWindowSize, (size_t)erAMuSeq[0].size()}, "a");
    cnpy::npz_save(saveFilepath, prefix + "_ASigma", _ASigma.data(), {currentWindowSize, (size_t)erASigmaSeq[0].size()}, "a");
    cnpy::npz_save(saveFilepath, prefix + "_kld", erKldSeq.data(), {erKldSeq.size()}, "a");
}

void PvrnnLayer::erSavePriorSequences(string saveFilepath, string prefix) {
    string mode = (std::filesystem::exists(saveFilepath) ? "a" : "w");

    Tensor2f _d = Tensor2f(erMupSeq.size(), erdSeq[0].size());
    Tensor2f _mup = Tensor2f(erMupSeq.size(), erMupSeq[0].size());
    Tensor2f _sigmap = Tensor2f(erMupSeq.size(), erSigmapSeq[0].size());

    for ( size_t _t = 0; _t < erMupSeq.size(); _t++ ){
        setInTensor2<VectorXf>(_d, erdSeq[_t+1], _t);
        setInTensor2<VectorXf>(_mup, erMupSeq[_t], _t);
        setInTensor2<VectorXf>(_sigmap, erSigmapSeq[_t], _t);
    }

    cnpy::npz_save(saveFilepath, prefix + "_d", _d.data(), {erMupSeq.size(), (size_t)erdSeq[0].size()}, mode);
    cnpy::npz_save(saveFilepath, prefix + "_mup", _mup.data(), {erMupSeq.size(), (size_t)erMupSeq[0].size()}, "a");
    cnpy::npz_save(saveFilepath, prefix + "_sigmap", _sigmap.data(), {erMupSeq.size(), (size_t)erSigmapSeq[0].size()}, "a");
}

void PvrnnLayer::erSaveWhdSeq(string saveFilepath, string prefix) {
    string mode = (std::filesystem::exists(saveFilepath) ? "a" : "w");

    Tensor2f _Whd = Tensor2f(erWhdSeq.size(), erWhdSeq[0].size());

    for (size_t _t = 0; _t < erWhdSeq.size(); _t++){
        setInTensor2<VectorXf>(_Whd, erWhdSeq[_t], _t);
    }

    cnpy::npz_save(saveFilepath, prefix + "_Whd", _Whd.data(), {erWhdSeq.size(), (size_t)erWhdSeq[0].size()}, mode);
}

void PvrnnLayer::erSaveWTdLdhSeq(string saveFilepath, string prefix) {
    string mode = (std::filesystem::exists(saveFilepath) ? "a" : "w");

    Tensor2f _wdLdh = Tensor2f(erWTdLdhSeq.size(), erWTdLdhSeq[0].size());

    /* save the gradient in the forward order */
    std::reverse(erWTdLdhSeq.begin(), erWTdLdhSeq.end());
    for (size_t _t = 0; _t < erWTdLdhSeq.size(); _t++){
        setInTensor2<VectorXf>(_wdLdh, erWTdLdhSeq[_t], _t);
    }

    cnpy::npz_save(saveFilepath, prefix + "_WTdLdh", _wdLdh.data(), {erWTdLdhSeq.size(), (size_t)erWTdLdhSeq[0].size()}, mode);
}

void PvrnnLayer::_saveSequencesNpz(string directory) {
    std::filesystem::create_directories(directory);
    size_t qLen = muqSeq.dimension(1);
    size_t pLen = mupSeq.dimension(1);
    size_t qDim = muqSeq.dimension(2);
    size_t dDim = dSeq.dimension(2);
    Tensor2f _d(pLen, dDim);
    Tensor2f _mup(pLen, qDim);
    Tensor2f _sigmap(pLen, qDim);
    Tensor2f _muq(qLen, qDim);
    Tensor2f _sigmaq(qLen, qDim);
    string mode;
    for ( int seqInd = 0; seqInd < nSeq; seqInd++ ){
        mode = (seqInd > 0) ? "a" : "w"; // overwrite any existing files
        bool transpose = false;
        getTensor2FromTensor3(dSeq, _d, seqInd, transpose, 1); // trim away t=0
        getTensor2FromTensor3(mupSeq, _mup, seqInd, transpose);
        getTensor2FromTensor3(sigmapSeq, _sigmap, seqInd, transpose);
        getTensor2FromTensor3(muqSeq, _muq, seqInd, transpose);
        getTensor2FromTensor3(sigmaqSeq, _sigmaq, seqInd, transpose);

        cnpy::npz_save(directory + "/d.npz", "d_" + to_string(seqInd), _d.data(), {pLen, dDim}, mode);
        cnpy::npz_save(directory + "/mup.npz", "mup_" + to_string(seqInd), _mup.data(), {pLen, qDim}, mode);
        cnpy::npz_save(directory + "/sigmap.npz", "sigmap_" + to_string(seqInd), _sigmap.data(), {pLen, qDim}, mode);
        cnpy::npz_save(directory + "/muq.npz", "muq_" + to_string(seqInd), _muq.data(), {qLen, qDim}, mode);
        cnpy::npz_save(directory + "/sigmaq.npz", "sigmaq_" + to_string(seqInd), _sigmaq.data(), {qLen, qDim}, mode);
    }
}

void PvrnnLayer::saveParameters(string directory) {
    _saveParametersNpz(directory);
}

void PvrnnLayer::loadParameters(string directory, bool rowMajorW){
    _loadParametersNpz(directory, rowMajorW);
}

void PvrnnLayer::_saveParametersNpz(string directory) {
    std::filesystem::create_directories(directory);

    size_t ALen = AMuSeq.dimension(1);
    size_t ADim = AMuSeq.dimension(2);
    Tensor2f _AMu(ALen, ADim);
    Tensor2f _ASigma(ALen, ADim);

    cnpy::npz_save(directory + "/W.npz", "Wdh", swapLayout(Wdh).data(), {(size_t)Wdh.rows(), (size_t)Wdh.cols()}, "w"); // First write
    cnpy::npz_save(directory + "/W.npz", "Wzh", swapLayout(Wzh).data(), {(size_t)Wzh.rows(), (size_t)Wzh.cols()}, "a");
    if (!topLayer) {
        cnpy::npz_save(directory + "/W.npz", "Whdh", swapLayout(Whdh).data(), {(size_t)Whdh.rows(), (size_t)Whdh.cols()}, "a");
    }
    cnpy::npz_save(directory + "/W.npz", "Wdmp", swapLayout(Wdmp).data(), {(size_t)Wdmp.rows(), (size_t)Wdmp.cols()}, "a");
    cnpy::npz_save(directory + "/W.npz", "Wdsp", swapLayout(Wdsp).data(), {(size_t)Wdsp.rows(), (size_t)Wdsp.cols()}, "a");

    cnpy::npz_save(directory + "/B.npz", "Bh" , Bh.data() , {(size_t)Bh.size()} , "w");
    cnpy::npz_save(directory + "/B.npz", "Bmp", Bmp.data(), {(size_t)Bmp.size()}, "a");
    cnpy::npz_save(directory + "/B.npz", "Bsp", Bsp.data(), {(size_t)Bsp.size()}, "a");

    string mode;
    for ( int seqInd = 0; seqInd < nSeq; seqInd++ ){
        mode = (seqInd > 0) ? "a" : "w"; // overwrite any existing files
        getTensor2FromTensor3(AMuSeq, _AMu, seqInd);
        getTensor2FromTensor3(ASigmaSeq, _ASigma, seqInd);

        cnpy::npz_save(directory + "/A.npz", "AMu_"    + to_string(seqInd), _AMu   .data(), {ALen, ADim}, mode);
        cnpy::npz_save(directory + "/A.npz", "ASigma_" + to_string(seqInd), _ASigma.data(), {ALen, ADim}, "a");
    }
}

void PvrnnLayer::_loadParametersNpz(string directory, bool rowMajorW) {
    cout << "loading parameters from " << directory << " ...";

    cnpy::npz_t _W, _B, _A;

    try {
        _W = cnpy::npz_load(directory + "/W.npz");
        _B = cnpy::npz_load(directory + "/B.npz");
        _A = cnpy::npz_load(directory + "/A.npz");
    } catch (const runtime_error& e) {
        cerr << "failed to load parameter file" << endl;
        throw;
    }
    try {
        // loading matrices (for W)
        checkedLoadFromNp(_W.at("Wdh") , Wdh , dSize, dSize, rowMajorW, "Wdh" );
        checkedLoadFromNp(_W.at("Wzh") , Wzh , dSize, zSize, rowMajorW, "Wzh" );
        checkedLoadFromNp(_W.at("Wdmp"), Wdmp, zSize, dSize, rowMajorW, "Wdmp");
        checkedLoadFromNp(_W.at("Wdsp"), Wdsp, zSize, dSize, rowMajorW, "Wdsp");
        if (!topLayer) checkedLoadFromNp(_W.at("Whdh"), Whdh, dSize, inputSize, rowMajorW, "Whdh");
        // loading vectors (for B)
        checkedLoadFromNp(_B.at("Bh") , Bh , dSize, "Bh" );
        checkedLoadFromNp(_B.at("Bmp"), Bmp, zSize, "Bmp");
        checkedLoadFromNp(_B.at("Bsp"), Bsp, zSize, "Bsp");
        // loading tensors (for A)
        checkedLoadFromNp(_A, "AMu_"   , AMuSeq   , nSeq, seqLen, zSize);
        checkedLoadFromNp(_A, "ASigma_", ASigmaSeq, nSeq, seqLen, zSize);
    } catch (const out_of_range& e) {
        cerr << "failed to load parameter" << endl;
        throw;
    }

    cout << "done\n";
}

inline VectorXf PvrnnLayer::clipVecGradient(VectorXf grad){
    if (grad.hasNaN()) {
        #if !defined(NDEBUG) || defined(NAN_INF_STRICT)
        throw runtime_error("NaN in gradient");
        #endif
        cerr << "warning: NaN replaced in gradient! \n";
        return VectorXf::Zero(grad.size());
    }
    if (!grad.allFinite()) {
        #if !defined(NDEBUG) || defined(NAN_INF_STRICT)
        throw runtime_error("Inf in gradient");
        #endif
        cerr << "warning: Inf replaced in gradient! \n";
        return VectorXf::Constant(grad.size(), clipGradThreshold);
    }

    double norm = sqrt( grad.array().square().sum() );
    if ( norm > clipGradThreshold ){
        return grad * ( clipGradThreshold / norm );
    }
    else {
        return grad;
    }
}

inline VectorXf PvrnnLayer::vecInit(Index size) {
    if (vectorZeroInit) return VectorXf::Zero(size);
    else return VectorXf::Random(size);
}

//MatrixXf PvrnnLayer::clipMatGradient(MatrixXf grad){
//    for ( int i = 0; i < grad.rows(); i++ ){
//        for ( int j = 0; j < grad.cols(); j++ ){
//            if ( isnan( grad(i, j) )){
//                cout << "NaN replaced in matrix! \n";
//                return MatrixXf::Zero(grad.rows(), grad.cols());
//            }
//        }
//    }
//    double norm = sqrt( grad.array().square().sum() );
//    if ( norm > clipGradThreshold ){
//        return grad * ( clipGradThreshold / norm );
//    }
//    else return grad;
//}

//VectorXf PvrnnLayer::replaceNanVec(VectorXf vec){
//    for ( int i = 0; i < vec.size(); i++ ){
//        if ( isnan( vec[i] )){
//            return VectorXf::Zero( vec.size() );}
//    }
//    return vec;
//}
//
//MatrixXf PvrnnLayer::replaceNanMat(MatrixXf mat){
//    for ( int i = 0; i < mat.rows(); i++ ){
//        for ( int j = 0; j < mat.cols(); j++ ){
//            if ( isnan( mat(i, j) )){
//                return MatrixXf::Zero(mat.rows(), mat.cols());
//            }
//        }
//    }
//    return mat;
//}
