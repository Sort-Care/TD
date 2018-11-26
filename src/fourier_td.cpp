#include "fourier_td.hpp"


/*
 * Needed functions
 * 1. A function to run TD update and return when finishes
 * Return:
 * - Eigen::VectorXd: recall that it isn't valid to return a reference to a local variable
 * TODO: Check the function to see if it is valid
 */
double
temporal_difference(const std::string filename,
                    const int num_iterations, // number of iterations
                    const int K, // order
                    const int N, // size of problem state presentation phi
                    const double step_size,
                    double (*tdFunc)(Eigen::VectorXd& weights,
                                   const int K,
                                   const double step_size) // TD update function provided by problem
                    ){
  int weightLen = pow((K+1), N) - 1;
  // initialize weight vector
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(weightLen);
  // Repeat for N episodes
  REP(i, 0, num_iterations){
    std::cout << "Function Approximation Iteration: " << i << std::endl;
    // Continue until this episode ends
    tdFunc(weights, K, step_size);// run TD on problem and get updates for weight vector
  }
  // run the evaluation of TD-error_MSE
  // typically callers are functions in problem environment implementation file.
  // E.g.: grid.cpp, cartpole.cpp, etc...
  double sum = 0.0;
  REP(i, 0, num_iterations){
    std::cout << "Function Approximation Evaluation: " << i << std::endl;
    Eigen::VectorXd resultW = weights;
    sum += tdFunc(resultW, K, step_size);
  }
  return sum / 100;
}

/*
 * For step size use: alpha_i = alpha_1 / ||c^i||2
 */

void save_file(const std::string filename){
  //save to file
}

/*
 * Generate the matrix used to multiply state representation
 * Input:
 * - k: Fourier function order
 * - n: state representation size
 * DONE: For loop version implementation
 * TODO: Is it possible to replace these loops with linear algebra?
 * TODO: Also, currently this function get called to many times, may be we
 *       can make it a global variable or embed it in class to save some
 *       computation work.
 * NOTE: Current implementation is inspired after a nap. The Fourier
 * basis function:
 * [0 0 0 0]
 * [0 0 0 1]
 * [0 0 0 2]
 *     .
 *     .
 *     .
 * [0 0 0 k]
 * [0 0 1 0]
 * [0 0 1 1]
 *     .
 *     .
 *     .
 * [k k k k]
 * is something like hex number representation. So,
 * the loop can loop through 0 to (k+1)^n-1, and compute their representation
 * in separate bits.
 * CAUTION: ! the number is from 0 to (k+1)^n - 1! Try imagine with binary numbers:
 * 1111 in binary is actually 2^3 + 2^2 + 2^1 + 2^1 = 2^4 - 1 = 31
 * where in binary case k = 1, n = 4
 */
Eigen::MatrixXd generate_fourier_multiplier(const int K,
                                            const int N){
  int rows = pow((K+1), N);
  Eigen::MatrixXd basis_multiplier = Eigen::MatrixXd::Zero(rows, N);
  REP(i, 0, rows-1){ // loop through all numbers
    int index = N-1; // variable for counting which bit we should update next
    int digit;   // the digit number that goes in next bit, range [0, k]
    int remainder = i; // variable to keep track of remainders, initially equals to i
    while(remainder != 0){ // i have more bits
      digit = remainder % (K+1); // mod by k+1, result in domain [0, k] inclusive
      basis_multiplier(i, index) = digit; // update the basis_multiplier
      index --; // increase count
      remainder = remainder / (K+1); // update remainder
    }
  }
  return basis_multiplier;
}

