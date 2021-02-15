#pragma once
///@file StatisticalEntities.hpp
///@brief This is a simple library that provides basic functionality to clusterize data according to either kmeans or fuzzy cmeans algorithms

#include <Eigen/Dense>
#ifndef FCM_MAX_ITERATIONS
///The maximum number of iterations to do if the algorithm doesn't otherwise converge
//@note It's wrapped in an #ifndef to allow the user to override it at compile time
#define FCM_MAX_ITERATIONS  20
#endif

#ifndef FCM_THRESHOLD
///The threshold below which real numbers are deemed identical, used to avoid infinite weights and determine when the algorithm has converged
//@note It's wrapped in an #ifndef to allow the user to override it at compile time
#define FCM_THRESHOLD   1.0E-19
#endif

using namespace Eigen;

///A constant used in offsetting the centroids of one-datapoint clusters in fuzzy-cmeans, to avoid infinite weights
const float offsetConstant                  = 0.05;
///The number of different seeds to try for each value of clusters number when attempting a clusterization
const int   attemptsPerClustersNumber       = 3;
///If empty clusters happen, the algorithm for that number of clusters fails, so we try up to this number of times if this happens.
const int   maxIterationPerClustersNumber   = 5;


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

/*!
 * @brief       This function calculates the weights of a fuzzy c-means clusterization, according to the next formula:
 *              w_ij = 1 / (norm(centroids(i), entities(j)))
 * @note        The weights are not normalized, as depending on the use case the normalization will be columns-wise or row-wise
 * @param[in]   entities    The datapoints
 * @param[in]   centroids   The centroids of the clusters
 * @param[out]  weights     The resulting weights
 * @param[in]   norm        A pointer to the norm function you want to use
*/
void calculateFuzzyWeights(
        const Ref<const MatrixXf>   &entities,
        const Ref<const MatrixXf>   &centroids,
        Ref<MatrixXfR>              weights,
        squaredNorm_t               *norm
        );

/*!
 * @brief Generates a fixed number of clusters and their fuzzy weights 
 * @param[in]       entities    The datapoints
 * @param[out]      centroids   The centroids of the clusters. The number of rows are the required clusters to find
 * @param[out]      weights     The weights associated with the clusterization (array of floats)
 * @param[in]       norm        A pointer to the norm function you want to use
*/
void FCMGenerator(
        const Ref<const MatrixXf>   &entities, 
        Ref<MatrixXf>               centroids, 
        Ref<MatrixXfR>              weights, 
        squaredNorm_t               *norm
    );

/*! 
 * @brief       Returns a measure of the fit of the clusterization, it's strictly positive and the smaller it is, the best the fit
 * @details     The Davies-Boulding Index defines a measure of the "goodness" of a clusterization of a data population based on the following quantities:
                The scatter vector S_i= (1/T_i * sum_j (norm(C_i, X_j)))^(1/2) where T_i is the population size of the i-th cluster and the sum runs over the datapoints belonging to the i-th cluster
                The Cluster Separation Matrix M_ij = (norm(C_i, C_j))^(1/2)
                The Davies-Bouldin Matrix R_ij = (S_i + S_j)/M_ij
                The Davies-Bouldin Vector R_i = max_(j!=i) R_ij
                The Davies-Bouldin index is, in terms of the previous quantities, R = 1/N * sum_i R_i where N is the number of clusters
 * @param[in]   entities    The datapoints
 * @param[in]   centroids   The centroids of the clusters
 * @param[out]  weights     The weights associating each centroid to its cluster (it's an array of bools)
 * @param[in]   norm        A pointer to the norm function you want to use
 * @return     The Davies-Boulding index of the provided clusterization
*/ 
float daviesBouldinIndex(
        const Ref<const MatrixXf>    &entities,
        const Ref<const MatrixXf>    &centroids,
        const Ref<const MatrixXb>    &weights,
        squaredNorm_t                *norm
    );

/*!
 * @brief Returns a measure of how well the clusters fit the data
 * @warning TODO <b>Not implemented</b>
 * @param[in]   entities     The datapoints
 * @param[in]   clusters     The centroids of the clusters
 * @param[out]  weights      The weights associating each centroid to its cluster (it's an array of floats)
 * @param[in]   norm         A pointer to the norm function you want to use
 * @return     the fitness of the clusterization
*/
float silhouetteTest(
        const Ref<const MatrixXf>   &entities, 
        const Ref<const MatrixXf>   &clusters,
        const Ref<const MatrixXfR>  &weights,
        squaredNorm_t               *norm
        );

/*!
 * @brief       Given a dataset and centroids, returns a weights matrix that is true if the j-th centroid is the closest to the i-th datapoint, and false otherwise
 * @param[in]   entities    The datapoints
 * @param[in]   centroids   The centroids of the clusters
 * @param[out]  weights     The weights associating each centroid to its cluster (it's an array of bools)
 * @param[in]   norm        A pointer to the norm function you want to use
*/
void calculateBooleanWeights(
        const Ref<const MatrixXf>   &entities,
        const Ref<const MatrixXf>   &centroids,
        Ref<MatrixXbR>              weights,
        squaredNorm_t               *norm
    );

/*!
 * @brief       Given a dataset and a centroids matrix of k rows, it tries to identify the most probable k centroids to represent the dataset
 * @param[in]   entities    The datapoints
 * @param[out]   centroids   The centroids of the clusters
 * @param[out]  weights     The weights associating each centroid to its cluster (it's an array of bools)
 * @param[in]   norm        A pointer to the norm function you want to use
*/
void kmeansGenerator(
        const Ref<const MatrixXf>   &entities,
        Ref<MatrixXf>               centroids,
        Ref<MatrixXbR>              weights,
        squaredNorm_t               *norm
    );

/*!
 * @brief Finds the best fitting number of clusters for the given datapoints, up to the number of rows of centroids through an approximated algorithm compared to full fuzzy c-means
 * @param[in]   entities    The datapoints
 * @param[out]  centroids   The centroids of the clusters
 * @param[out]  weights     The weights associating each centroid to its cluster (it's an array of floats)
 * @param[out]  boolWeights The weights associating each centroid to its cluster (it's an array of bools)
 * @param[in]   norm        A pointer to the norm function you want to use
 * @return     the number of clusters generated
*/ 
int clusterGeneratorApproximate(
        const Ref<const MatrixXf>   &entities,
        Ref<MatrixXf>               centroids,
        Ref<MatrixXfR>              weights,
        Ref<MatrixXbR>              boolWeights,
        squaredNorm_t               *norm
    );

/*!
 * @brief Finds the best fitting number of clusters for the given datapoints, up to the number of rows of centroids using the fuzzy c-means algorithm
 * @param[in]   entities    The datapoints
 * @param[out]  centroids   The centroids of the clusters
 * @param[out]  weights     The weights associating each centroid to its cluster (it's an array of floats)
 * @param[in]   norm        A pointer to the norm function you want to use
 * @return     the number of clusters generated
*/ 
int clusterGeneratorExact(
        const Ref<const MatrixXf>   &entities,
        Ref<MatrixXf>               centroids,
        MatrixXfR                   &weights, 
        squaredNorm_t               *norm
    );

