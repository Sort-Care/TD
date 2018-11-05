#ifndef __FOURIER__TD
#define __FOURIER__TD
#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>

#include "conventions.hpp"
/*
 * TD algorithm
 */
Eigen::VectorXd
temporal_difference(const int num_iterations, // number of iterations to update
                    const int K, // order
                    const int N, // size of state representation
                    const double step_size,
                    // problem specific td Update function, returns when episode ends
                    void (*tdFunc)(Eigen::VectorXd& weights,
                                   const int K,
                                   const double step_size)
                    );

void save_file(const std::string filename);

/*
 * The function used to generate Fourier multiplier.
 * Input:
 * - k: the order of the Fourier basis function
 * - n: the size of problem state size, for cart pole: 4, for gw: 2
 */
Eigen::MatrixXd generate_fourier_multiplier(const int K,
                                            const int N);

#endif
