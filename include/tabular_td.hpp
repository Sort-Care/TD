#ifndef __TABULAR_TD
#define __TABULAR_TD
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>

#include "conventions.hpp"

double
tabular_TD(const int num_iterations,
           const int table_size,
           const double step_size,
           double (*TabularTD)(Eigen::VectorXd&,
                               const double));

#endif
