#pragma once

#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#ifdef EIGEN_HAS_OPENMP
#include <omp.h>
#endif
#ifdef USE_EIGENRAND
#include "../EigenRand/EigenRand"
#else
#include <random>
#endif
#include "../cnpy/cnpy.h"
#include "../toml/toml.h"

using namespace std;
using namespace Eigen;

typedef Matrix<float, -1, -1, RowMajor> MatrixXfr;

typedef vector<VectorXf> vectorXf1DContainer;
// [[deprecated]] typedef vector<vectorXf1DContainer> vectorXf2DContainer;
// [[deprecated]] typedef vector<vectorXf2DContainer> vectorXf3DContainer;
typedef vector<MatrixXf> matrixXf1DContainer;

typedef Tensor<float, 0, RowMajor, long> Tensor0f; // scalar
typedef Tensor<float, 1, RowMajor, long> Tensor1f;
typedef Tensor<float, 2, RowMajor, long> Tensor2f;
typedef Tensor<float, 3, RowMajor, long> Tensor3f;


const double DEFAULT_NORM_MIN_FC = -1;
const double DEFAULT_NORM_MAX_FC =  1;

const double DEFAULT_NORM_MIN_SM = 0.0;
const double DEFAULT_NORM_MAX_SM = 1.0;

#ifdef NDEBUG
// Fall back to clean exit on unhandled exception
inline void catchall_exceptions() {
    exception_ptr eptr = current_exception();
    if (eptr) {
        try {
            rethrow_exception(eptr);
        } catch (const exception& e) {
            cerr << "error: " << e.what() << endl;
        }
    } else {
        cerr << "an unknown error occured" << endl;
    }
    exit(EXIT_FAILURE);
}
#endif

inline void* checkedMalloc(size_t size) {
    void *p = malloc(size);
    if (p == NULL) abort(); // catastrophic failure
    return p;
}

inline VectorXf clippedLog(const VectorXf &x, float minval=1e-4) {
    return x.cwiseMax(minval).array().log();
}

inline Tensor3f clippedLog(const Tensor3f &t3, float minval=1e-4) {
    return t3.cwiseMax(minval).log();
}

inline MatrixXf glorotNormalInit(int fan_in, int fan_out, default_random_engine* rng) {
    #ifdef USE_EIGENRAND
    return Rand::normal<MatrixXf>(fan_in, fan_out, *rng, 0.0f, sqrt(2.0f/(fan_in+fan_out)));
    #else
    normal_distribution<float> dist{0.0f, sqrt(2.0f/(fan_in+fan_out))};
    MatrixXf rmat(fan_in, fan_out);
    for (int i = 0; i < fan_in; i++) {
        for (int j = 0; j < fan_out; j++) {
            rmat(i, j) = dist(*rng);
        }
    }
    return rmat;
    #endif
}

// Swap Matrix layout from column-major to row-major
inline MatrixXfr swapLayout(const MatrixXf &colmaj) {
    MatrixXfr rowmaj = colmaj;
    return rowmaj;
}

// Swap Matrix layout from row-major to column-major
inline MatrixXf swapLayout(const MatrixXfr &rowmaj) {
    MatrixXf colmaj = rowmaj;
    return colmaj;
}

// Takes a npz object with a_len keys, with each array being shape (b_len, c_len), and maps it to a Tensor3f. Fails if dimensions don't match
inline void checkedLoadFromNp(cnpy::npz_t &arm, string prefix, Tensor3f &t3, size_t a_len, size_t b_len, size_t c_len) {
    cnpy::NpyArray a2;
    for (size_t a = 0; a < a_len; a++) {
        a2 = arm.at(prefix + to_string(a)); // can throw
        if (a2.shape.size() != 2 || a2.shape[0] != b_len || a2.shape[1] != c_len) {
            string arrshape = to_string(a2.shape[0]);
            for (size_t d = 1; d < a2.shape.size(); d++) arrshape += ", " + to_string(a2.shape[d]);
            throw invalid_argument(prefix + to_string(a) + " expects shape (" + to_string(b_len) + ", " + to_string(c_len) + ") but loaded (" + arrshape + ")");
        }
        t3.chip(a, 0) = TensorMap<Tensor2f>(a2.data<float>(), b_len, c_len);
    }
}

// Takes a npy object and maps it to a MatrixXf. Fails if dimensions don't match
inline void checkedLoadFromNp(cnpy::NpyArray &arr, Ref<MatrixXf> mat, size_t rows, size_t cols, bool rowMajor=false, string label="Matrix") {
    if (arr.shape.size() != 2 || arr.shape[0] != rows || arr.shape[1] != cols) {
        string arrshape = to_string(arr.shape[0]);
        for (size_t d = 1; d < arr.shape.size(); d++) arrshape += ", " + to_string(arr.shape[d]);
        throw invalid_argument(label + " expects shape (" + to_string(rows) + ", " + to_string(cols) + ") but loaded (" + arrshape + ")");
    }
    if (rowMajor) {
        // First we load it in a row-major matrix to easily do the swap.
        MatrixXfr _rmat = Map<MatrixXfr>(arr.data<float>(), rows, cols);
        mat = _rmat;
    } else {
        mat = Map<MatrixXf>(arr.data<float>(), rows, cols);
    }
}

// Takes a npy object and maps it to a VectorXf. Fails if dimensions don't match
inline void checkedLoadFromNp(cnpy::NpyArray &arr, Ref<VectorXf> vec, size_t length, string label="Vector") {
    if (arr.num_vals != length) {
        throw invalid_argument(label + " expects length (" + to_string(length) + ") but loaded (" + to_string(arr.num_vals) + ")");
    }
    vec = Map<VectorXf>(arr.data<float>(), length);
}

// Transpose a rank 3 tensor without changing the declared order
inline Tensor3f transposeTensor3(const Tensor3f &t3) {
    return t3.shuffle(Eigen::array<int, 3>({2, 1, 0}));
}

//* Map t3(a_idx, b_idx, :) to a vector
template <typename Xf>
inline void getFromTensor3(const Tensor3f &t3, Ref<Xf> vec, size_t a_idx, size_t b_idx=0) {
    vec = Map<Xf>(const_cast<float*>(t3.data())+(a_idx*t3.dimension(1)*t3.dimension(2))+(b_idx*t3.dimension(2)), t3.dimension(2));
}

// Map t3(a_idx, :, :) to a row-major matrix, then converts it to column major
template <>
inline void getFromTensor3<MatrixXf>(const Tensor3f &t3, Ref<MatrixXf> mat, size_t a_idx, size_t b_idx) {
    MatrixXfr _rmat = Map<MatrixXfr>(const_cast<float*>(t3.data())+(a_idx*t3.dimension(1)*t3.dimension(2)), t3.dimension(1), t3.dimension(2));
    mat = _rmat;
}

// Takes a VectorXf/ArrayXf, maps it to a rank 1 tensor, then assigns it to the input tensor at (a_idx, b_idx, :)
template <typename Xf>
inline void setInTensor3(Tensor3f &t3, const Ref<const Xf> &vec, size_t a_idx, size_t b_idx=0) {
    t3.chip(a_idx, 0).chip(b_idx, 0) = TensorMap<Tensor1f>(const_cast<float*>(vec.data()), vec.size());
}

// Takes a MatrixXf, converts it to row major, maps it to a rank 2 tensor, then assigns it to the input tensor at (a_idx, :, :)
template <>
inline void setInTensor3<MatrixXf>(Tensor3f &t3, const Ref<const MatrixXf> &mat, size_t a_idx, size_t b_idx) {
    MatrixXfr _rmat = mat;
    t3.chip(a_idx, 0) = TensorMap<Tensor2f>(_rmat.data(), _rmat.rows(), _rmat.cols());
}

// Takes a VectorXf/ArrayXf, maps it to a rank 1 tensor, then assigns it to the input tensor at (a_idx, :)
template <typename Xf>
inline void setInTensor2(Tensor2f &t2, const Ref<const Xf> &vec, size_t a_idx) {
    t2.chip(a_idx, 0) = TensorMap<Tensor1f>(const_cast<float*>(vec.data()), vec.size());
}

// Takes a VectorXf/ArrayXf, maps it to a rank 1 tensor, then assigns it to the input tensor at (a_idx, :)
template <typename Xf>
inline void setInTensor2(Tensor2f &t2, const Xf &vec, size_t a_idx) {
    t2.chip(a_idx, 0) = TensorMap<Tensor1f>(const_cast<float*>(vec.data()), vec.size());
}

// Extract a rank 2 tensor with optional transpose and trimming
inline void getTensor2FromTensor3(const Tensor3f &t3, Tensor2f &t2, const size_t a_idx, const bool transpose=false, const int ftrim_b=0) {
    Tensor2f _t2 = t3.chip(a_idx, 0);
    if (ftrim_b > 0) {
        Tensor2f _t2s = _t2.slice(Eigen::array<long, 2>{ftrim_b, 0}, Eigen::array<long, 2>{_t2.dimension(0)-ftrim_b, _t2.dimension(1)});
        _t2 = _t2s;
    }
    if (transpose) t2 = _t2.shuffle(Eigen::array<long, 2>({1, 0}));
    else t2 = _t2;
}

// Gathers tensor slices from a vector of indices and puts them in the output tensor
inline void gatherFromTensor3(const Tensor3f &in, Tensor3f &out, const vector<int> &indices) {
    for (size_t i = 0; i < indices.size(); i++) {
        out.chip(i, 0) = in.chip(indices[i], 0);
    }
}

// Takes input tensor and satters tensor slices following a vector of indices
inline void scatterToTensor3(const Tensor3f &in, Tensor3f &out, const vector<int> &indices) {
    for (size_t i = 0; i < indices.size(); i++) {
        out.chip(indices[i], 0) = in.chip(i, 0);
    }
}

// Apply a lambda operation to pairs of tensor slices
template <typename Tensor2Op>
inline void mapToTensor3(Tensor3f &ta, Tensor3f &tb, const vector<int> &indices, Tensor2Op op) {
    for (size_t i = 0; i < indices.size(); i++) {
        op(ta.chip(indices[i], 0), tb.chip(indices[i], 0));
    }
}

template <typename Tensor2Op>
inline void mapToTensor3(Tensor3f &ta, Tensor3f &tb, Tensor2Op op) {
    for (size_t i = 0; i < ta.dimension(0); i++) {
        op(ta.chip(i, 0), tb.chip(i, 0));
    }
}
