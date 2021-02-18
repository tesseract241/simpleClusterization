
#pragma once
///@file simpleClusterization_common.hpp
///@brief common definitions that users of the library might need independently from the rest of the files 

#include <Eigen/Dense>

using namespace Eigen;

///Shorthand type for a RowMajor Matrix of floats
typedef Matrix<float, Dynamic, Dynamic, RowMajor>   MatrixXfR;
///Shorthand type for a ColMajor Matrix of bools
typedef Matrix<bool, Dynamic, Dynamic>              MatrixXb;
///Shorthand type for a RowMajor Matrix of bools
typedef Matrix<bool, Dynamic, Dynamic, RowMajor>    MatrixXbR;

///The type signature for the norm parameters of the functions in this library
typedef float squaredNorm_t(const VectorXf &v1, const VectorXf &v2);

///A basic example of norm that satisfies the type signature
float euclideanNorm(const VectorXf &v1, const VectorXf &v2);
