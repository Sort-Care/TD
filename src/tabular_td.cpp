#include "tabular_td.hpp"

double tabular_TD(const int num_iterations,
                  const int table_size,
                  const double step_size,
                  double (*TabularTD)(Eigen::VectorXd& vf,
                                      const double step_size)){
  // initialize value function to be zero
  Eigen::VectorXd value_function = Eigen::VectorXd::Zero(table_size);
  REP (i, 0, num_iterations){
    std::cout << "Tabular TD running iteration: " << i << std::endl;
    // perform td update during a whole episode
    TabularTD(value_function, step_size);
    //std::cout << value_function << std::endl;
  }
  double sum = 0.0;
  REP(i, 0, num_iterations){
    std::cout << "Tabular TD running Final Eval: " << i << std::endl;
    Eigen::VectorXd resultVf = value_function;
    sum += TabularTD(resultVf, step_size);
  }
  return sum / 100;
}
