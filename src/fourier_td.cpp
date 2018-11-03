
#include "fourier_td.hpp"


/*
 * Needed functions
 * 1. A function to run TD update and return when finishes
 * Return:
 * - Eigen::VectorXd: recall that it isn't valid to return a reference to a local variable
 * TODO: Check the function to see if it is valid
 */
Eigen::VectorXd
temporal_difference(const int num_iterations, // number of iterations
                    const int weightSize, // this params is problem specific
                    void (*tdFunc)(Eigen::VectorXd& weights) // TD update function provided by problem
                    ){
  // initialize weight vector
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(weightSize);
  // Repeat for N episodes
  REP(i, 0, 100){
    // Continue until this episode ends
    tdFunc(weights);// run TD on problem and get updates for weight vector
  }
  // returns the learned weight back to its caller,
  // typically callers are functions in problem environment implementation file.
  // E.g.: grid.cpp, cartpole.cpp, etc...
  return weights;
}

/*
 * For step size use: alpha_i = alpha_1 / ||c^i||2
 */

void save_file(const std::string filename){
  //save to file
}
