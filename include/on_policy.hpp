#ifndef __ONPOLICY
#define __ONPOLICY

#include <Eigen/Dense>
#include <Eigen/Core>
#include <string>
#include <iostream>
#include <fstream>

#include "conventions.hpp"


/*
 * Sarsa Algorithm Declaration
 */
Eigen::RowVectorXd on_policy_function_approx(const int num_episodes,
                                             const double step_size,
                                             const int explore_mode, // 0 for epsilon greedy, 1 for softmax
                                             const int q_weight_size,
                                             double (*sarsa_fa)(const int episode,
                                                                Eigen::VectorXd& weights,
                                                                const double step_size,
                                                                const int explore_mode,
                                                                Eigen::VectorXd (*get_phi_s)(struct cart_state& cs)),
                                             Eigen::VectorXd (*get_phi_s)(struct cart_state& cs)
                             );

Eigen::RowVectorXd on_policy_tabular(const int num_episodes,
                                     const double step_size,
                                     const int explore_mode,
                                     const int q_table_size,
                                     double (*sarsa_tb)(const int episode,
                                                        Eigen::VectorXd& qf,
                                                        const double step_size,
                                                        const int explore_mode)
                                     );


#endif
