#include <Eigen/Dense>
#include <algorithm>
#include <cfloat>
#include <random>
#include <iostream>
#include <simpleClusterization.hpp>

using namespace Eigen;

float euclideanNorm(
        const VectorXf &v1, 
        const VectorXf &v2
    ){
    return (v1 - v2).squaredNorm();
}

void calculateFuzzyWeights(
        const Ref<const MatrixXf>   &entities,
        const Ref<const MatrixXf>   &centroids,
        Ref<MatrixXfR>              weights,
        squaredNorm_t               *norm
        ){
    const int entitiesNumber  = entities.rows();
    const int centroidsNumber = centroids.rows();
    for(int i=0;i<centroidsNumber;++i){
        for(int j=0;j<entitiesNumber;++j){
            weights(i, j) = (*norm)(centroids.row(i), entities.row(j));
            assert(weights(i, j)> FCM_THRESHOLD && "A centroid and an entity coincide, this leads to infinite weights, correct\n");
        }
    }
    weights.array() = 1.0 / weights.array();
}


void FCMGenerator(
        const Ref<const MatrixXf>   &entities, 
        Ref<MatrixXf>               centroids, 
        Ref<MatrixXfR>              weights, 
        squaredNorm_t               *norm
    ){
    //The loop is as following:
    //1 - We initialize weights to random values from 0 to 1 (to check if they need to sum up to 1)
    //2 - We calculate the Centroids of each cluster (what is gonna end up in entities as a Statistical Entity according to the following formula: c_j = (Sum_i w_ij^m * x_i)/(Sum_i w_ij^m)
    //3 - We update weights according to this formula:  w_ij = 1 / (Sum_k (distance(x_i, c_j)/distance(x_i, c_k)) ^ (2 / m-1)) where m is a fuzziness parameter that's usually put equal to 2
    // Loop until Norm(W_i+1 - W_i) < Epsilon where Epsilon is a threshold decided by the coder
    assert(weights.cols()==entities.rows() && centroids.rows()==weights.rows() && centroids.cols()==entities.cols() && "Matrix sizes for FCMGenerator not compatibles\n");
    const int centroidsNumber = centroids.rows();
    MatrixXfR weightsOld(weights.rows(), weights.cols());
    MatrixXfR weights2(weights.rows(), weights.cols());
    MatrixXfR weightsOld2(weights.rows(), weights.cols());
    //Initialization of the weights at random values, might be improved if we could get a reasonable initial guess
    weightsOld = MatrixXf::Zero(weights.rows(), weights.cols());
    weights = MatrixXf::Random(weights.rows(), weights.cols());
    for(int loopIndex=0;loopIndex<FCM_MAX_ITERATIONS && (weights - weightsOld).squaredNorm() > FCM_THRESHOLD;++loopIndex){
        weightsOld = weights;
        weights2 = weights.array().square();
        //Calculation of the Centroids
        centroids = weights2 * entities;
        for(int i=0;i<centroidsNumber;++i){
            float multiplier  = 1.0 / weights2.row(i).sum();
            centroids.row(i) *= multiplier; 
        }
        //Update of the weights with the new Centroids
        calculateFuzzyWeights(entities, centroids, weights, norm);
        weights.rowwise().normalize();
    }
}

float daviesBouldinIndex(
        const Ref<const MatrixXf>    &entities,
        const Ref<const MatrixXf>    &centroids,
        const Ref<const MatrixXb>    &weights,
        squaredNorm_t                *norm
    ){
    const int clustersNumber = centroids.rows();
    const int startingEntitiesNumber = entities.rows();
    VectorXf scatterVector = VectorXf::Zero(clustersNumber);
    for(int i=0;i<clustersNumber;++i){
        for(int j=0;j<startingEntitiesNumber;++j){
            if(weights(i, j)){
                scatterVector(i)+=(*norm)(centroids.row(i), entities.row(j));
            }
        }
        float multiplier  = 1.0 / (float) weights.row(i).count();
        scatterVector(i) *= multiplier;
    }
    scatterVector = scatterVector.cwiseSqrt();
    MatrixXf clusterSeparationMatrix(clustersNumber, clustersNumber);
    for(int i=0;i<clustersNumber;++i){
        for(int j=0;j<clustersNumber;++j){
            clusterSeparationMatrix(i, j) = (*norm)(centroids.row(i), centroids.row(j));
        }
    }
    clusterSeparationMatrix = clusterSeparationMatrix.cwiseSqrt();
    float dbIndex = 0;
    for(int i=0;i<clustersNumber;++i){
        float dbIndexCluster = 0;
        for(int j=0;j<clustersNumber;++j){
            if(i!=j){
                dbIndexCluster = std::max((scatterVector(i) + scatterVector(j))/clusterSeparationMatrix(i, j), dbIndexCluster);
            }
        }
        dbIndex+=dbIndexCluster;
    }
    return dbIndex/float(clustersNumber);
}

float silhouetteTest(
        const Ref<const MatrixXf>   &entities, 
        const Ref<const MatrixXf>   &clusters,
        const Ref<const MatrixXfR>  &weights
        ){
    //TODO
    return 1.;
}


/*!
 * @brief      Takes data points and a k-long cluster matrix, generates initial values for k centroids.
 * @note       entities is copied and not referenced, as it needs to be altered for simplicity's sake
 * @param[in]  entities    The datapoints
 * @param[in-out] centroids   The centroids of the clusters generated by this function
 * @param[in]  norm        A pointer to the norm function you want to use
*/
void kmeansInitializer(
        const Ref<const MatrixXf>    &entities,
        Ref<MatrixXf>                centroids,
        squaredNorm_t                *norm
    ){
    const int statsNumber = entities.cols();
    const int startingEntitiesNumber = entities.rows();
    const int clustersNumber = centroids.rows();
    int currentClustersNumber = 0;
    MatrixXfR mEntities = entities;
    VectorXf squaredDistances(startingEntitiesNumber);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> intDistribution(0, startingEntitiesNumber - 1);
//   1. Choose one center uniformly at random among the data points.
    int randomIndex = intDistribution(gen);
    while(true){
        centroids.row(currentClustersNumber) = mEntities.row(randomIndex);
        mEntities.row(randomIndex) = mEntities.row(startingEntitiesNumber - currentClustersNumber - 1);
        mEntities.row(startingEntitiesNumber - currentClustersNumber - 1) = VectorXf::Zero(statsNumber);
        ++currentClustersNumber;
        if(currentClustersNumber==clustersNumber){
            break;
        }
        //   2. For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
        for(int j=0;j<startingEntitiesNumber - currentClustersNumber;++j){
            squaredDistances(j) = (*norm)(centroids.row(0), mEntities.row(j));
            for(int i=1;i<currentClustersNumber;++i){
                squaredDistances(j) = std::min((*norm)(centroids.row(i), mEntities.row(j)), squaredDistances(j));
            }
        }
        //   3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
        std::partial_sum(squaredDistances.data(), squaredDistances.data() + startingEntitiesNumber - currentClustersNumber - 1, squaredDistances.data());
        std::uniform_real_distribution<> floatDistribution(squaredDistances(0), squaredDistances(startingEntitiesNumber - currentClustersNumber - 1));
        float randomFloat = floatDistribution(gen);
        randomIndex = std::upper_bound(squaredDistances.data(), squaredDistances.data() + startingEntitiesNumber - currentClustersNumber - 1, randomFloat) - squaredDistances.data();
        //   4. Repeat Steps 2 and 3 until k centers have been chosen.
    }
//   5. Now that the initial centers have been chosen, proceed using standard k-means clustering.
}


void calculateBooleanWeights(
        const Ref<const MatrixXf>   &entities,
        const Ref<const MatrixXf>   &centroids,
        Ref<MatrixXbR>              weights,
        squaredNorm_t               *norm
    ){
    const int entitiesNumber = entities.rows();
    const int clustersNumber = centroids.rows();
    weights.setConstant(false);
    for(int j=0;j<entitiesNumber;j++){
        float minDistance = (*norm)(centroids.row(0), entities.row(j));
        int minIndex = 0;
        for(int i=1;i<clustersNumber;i++){
            float currentDistance = (*norm)(centroids.row(i), entities.row(j));
            if(currentDistance < minDistance){
                minDistance = currentDistance;
                minIndex = i;
            }
        }
        weights(minIndex, j) = true;
    }
}


void kmeansGenerator(
        const Ref<const MatrixXf>   &entities,
        Ref<MatrixXf>               centroids,
        Ref<MatrixXbR>              weights,
        squaredNorm_t               *norm
    ){
    assert(entities.cols()==centroids.cols() && "Called cmeansGenerator with entities and centroids having different dimensions\n");
    const int entitiesNumber = entities.rows();
    const int clustersNumber = centroids.rows();
    //We initialize the centroids with some datapoints that are spread out across the dataset, according to the kmeans++ algorithm
    kmeansInitializer(entities, centroids, norm);
    calculateBooleanWeights(entities, centroids, weights, norm);
    MatrixXb oldWeights(clustersNumber, entitiesNumber);
    oldWeights.setConstant(false);
    while(oldWeights!=weights){
        oldWeights = weights;
        centroids = (weights.cast<float>()) * entities;
        for(int i=0;i<clustersNumber;++i){
            float multiplier  = 1.0 / (float) weights.row(i).count();
            centroids.row(i) *= multiplier; 
        }
        calculateBooleanWeights(entities, centroids, weights, norm);
    }
}
    
int clusterGeneratorApproximate(
        const Ref<const MatrixXf>   &entities,
        Ref<MatrixXf>               centroids,
        Ref<MatrixXfR>              weights,
        Ref<MatrixXbR>              boolWeights,
        squaredNorm_t               *norm
    ){
    assert(centroids.cols()==entities.cols() && "clusterGeneratorApproximate: called with entities and centroids having different sizes");
    const int statsNumber = entities.cols();
    const int entitiesNumber = entities.rows();
    const int maxClustersNumber = centroids.rows(); 
    float fitnessCandidate = 50.;
    float newFitness = 50.;
    MatrixXf currentClustersCandidate(maxClustersNumber, statsNumber); 
    MatrixXbR currentBoolWeightsCandidate(maxClustersNumber, entitiesNumber);
    int clustersNumber = 2;
    //The minimum amount of clusters is 2 because otherwise the Davies-Bouldin index fails
    for(int currentClustersNumber = 2; currentClustersNumber<=maxClustersNumber; ++currentClustersNumber){
        for(int i=0;i<attemptsPerClustersNumber;++i){
            int iterations = 0;
            do{
                currentClustersCandidate.topLeftCorner(currentClustersNumber, statsNumber).setZero();
                currentBoolWeightsCandidate.topLeftCorner(currentClustersNumber, entitiesNumber).setZero();
                kmeansGenerator(entities, currentClustersCandidate.topLeftCorner(currentClustersNumber, statsNumber), currentBoolWeightsCandidate.topLeftCorner(currentClustersNumber, entitiesNumber), norm);
                newFitness = daviesBouldinIndex(entities, currentClustersCandidate.topLeftCorner(currentClustersNumber, statsNumber), currentBoolWeightsCandidate.topLeftCorner(currentClustersNumber, entitiesNumber),  norm);
                ++iterations;
            }while(std::isnan(newFitness) && iterations < maxIterationPerClustersNumber);
            if (newFitness < fitnessCandidate){
                centroids.topLeftCorner(currentClustersNumber, statsNumber)            = currentClustersCandidate.topLeftCorner(currentClustersNumber, statsNumber);
                boolWeights.topLeftCorner(currentClustersNumber, entitiesNumber)       = currentBoolWeightsCandidate.topLeftCorner(currentClustersNumber, entitiesNumber);
                fitnessCandidate                                                = newFitness;
                clustersNumber                                                 = currentClustersNumber;
            }
        }
    }
    //Single-datapoint clusters lead to infinite fuzzy weights, so we offset them by a small vector.
    //The risk in doing this is that we might end up moving the centroid too much, so that its datapoint ends up in another cluster.
    //So to avoid this, we scale our offsetConstant by the dataset's dimensionality.
    //Additionally, we choose as a direction the one defined by the current vector and the average one the result is that we're pushing the centroid towards the center of the whole dataset, 
    //which makes it slightly harder for the worst case scenario to happen.
    const float shiftMultiplier = offsetConstant / float(statsNumber);
    const RowVectorXf averageEntity = entities.colwise().mean();
    for(int i=0;i<clustersNumber;++i){
        if(boolWeights.row(i).count()==1){
            centroids.row(i) += shiftMultiplier * (centroids.row(i) - averageEntity);
        }
    }
    calculateFuzzyWeights(entities, centroids.topRightCorner(clustersNumber, statsNumber), weights.topLeftCorner(clustersNumber, entitiesNumber), norm);
    return clustersNumber;
}

int clusterGeneratorExact(
        const Ref<const MatrixXf>   &entities,
        Ref<MatrixXf>               centroids,
        MatrixXfR                   &weights, 
        squaredNorm_t               *norm
    ){
    assert(weights.rows()==weights.cols() && "Called clusterGenerator with a non-square weights matrix\n");
    const int statsNumber = entities.cols();
    const int entitiesNumber = entities.rows();
    const int maxClustersNumber = centroids.rows();
    float fitnessCandidate = 0;
    float newFitness = 0;
    int centroidsNumber = 2;
    MatrixXf currentClustersCandidate(maxClustersNumber, statsNumber);
    //Initialized to catch the improbable case of silhouetteTest() always returning 0
    MatrixXfR currentWeightsCandidate, weightsCandidate;
    //Initialized to catch the improbable case of silhouetteTest() always returning 0
    for(int clustersNumber = 2; clustersNumber<=maxClustersNumber; ++clustersNumber){
        currentClustersCandidate.topLeftCorner(clustersNumber, statsNumber).setZero();
        currentWeightsCandidate.topLeftCorner(clustersNumber, entitiesNumber).setZero();
        FCMGenerator(entities, currentClustersCandidate, currentWeightsCandidate, norm);
        newFitness = silhouetteTest(entities, currentClustersCandidate, currentWeightsCandidate);
        if (newFitness > fitnessCandidate){
            centroids.topLeftCorner(clustersNumber, statsNumber) = currentClustersCandidate.topLeftCorner(clustersNumber, statsNumber);
            weights.topLeftCorner(clustersNumber, entitiesNumber) = currentWeightsCandidate.topLeftCorner(clustersNumber, entitiesNumber);
            fitnessCandidate = newFitness;
            centroidsNumber = clustersNumber;
        }
    }
    return centroidsNumber;
}
