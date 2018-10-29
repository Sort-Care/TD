#ifndef __RANDOM_SAMPLE
#define __RANDOM_SAMPLE

#include <random>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>
#include "conventions.hpp"

/*
 * Random Samle from a distribution array:
 * Input: 
 *       double distribution: an double array specifing a distribution, 
 *                            should add up to 1
 *       int size: the array size
 */
int random_sample_distribution(const double distribution[],const int size);

int random_sample_weights(const double weights[], const int size);

double random_zero_to_one();

double random_range(double range);

/*
 * Sample from a eigen vector
 */
int random_sample_eigen_vectors(const Eigen::VectorXd& vec);


#endif
