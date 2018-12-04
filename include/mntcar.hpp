#ifndef __MOUNTAIN_CAR
#define __MOUNTAIN_CAR

#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>
#include <fstream>

#include "random_sampling.hpp"
#include "norm.hpp"

struct car_state{
  double x;
  double v;
};

extern const double LEFT_END;
extern const double RIGHT_END;
extern const double CAR_GOAL;
extern const double CAR_REWARD;
extern const double CAR_INIT_X;
extern const double CAR_INIT_V;
extern const double V_MAX;
extern const double V_MIN;
extern const int CAR_ACTIONS[3];

bool is_goal_reached(const struct car_state& cs);
void update_car_state(struct car_state& cs,
                      const int action);
Eigen::VectorXd normalize_car_state(const struct car_state& cs);

double sarsa_lambda_mntcar(const int episode,
                           Eigen::VectorXd& qw,
                           const double step_size,
                           const int explore_mode,
                           Eigen::VectorXd (*get_phi_s)(struct car_state& cs));

double ql_lambda_mntcar(const int episode,
                        Eigen::VectorXd& qw,
                        const double step_size,
                        const int explore_mode,
                        Eigen::VectorXd (*get_phi_s)(struct car_state& cs)
                        );

double ac_mntcar(const int episode,
                 Eigen::VectorXd& weights,
                 Eigen::VectorXd& theta_param,
                 const double step_size,
                 const double lambda
                 );


#endif
