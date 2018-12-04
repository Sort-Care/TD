#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <fstream>

#include "conventions.hpp"

/*
 * Actor Critic Skeleton
 */
Eigen::RowVectorXd actor_critic(const int num_episodes,
                                const double step_size,
                                const double lambda,
                                const int explore_mode,
                                const int theta_param_size,
                                const int weight_size,
                                double (*ac_prob)(const int episode,
                                                  Eigen::VectorXd& weights,
                                                  Eigen::VectorXd& theta_param,
                                                  const double step_size,
                                                  const double lambda
                                                  )
                                );
