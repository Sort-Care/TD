#ifndef __CART_POLE
#define __CART_POLE
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>
#include <fstream>

#include "fourier_td.hpp"
#include "on_policy.hpp"
#include "random_sampling.hpp"
#include "norm.hpp"

struct cart_state {
  double x;        // position
  double x_dot;    // cart velocity
  double theta;    // pole angle
  double theta_dot;// pole angular velocity
};

extern const double FAIL_ANGLE;
extern const double FORCE;
extern const double GRAV;
extern const double CART_M;
extern const double POLE_M;
extern const double POLE_L;
extern const double INTERVAL;
extern const double MAXTIME;
extern const double X_MAX;
extern const double X_MIN;
extern const double X_DOT_MAX;
extern const double X_DOT_MIN;
extern const double THETA_MAX;
extern const double THETA_MIN;
extern const double THETA_DOT_MAX;
extern const double THETA_DOT_MIN;

extern const int NUM_BUCKETS;
extern const int NUM_LR;

enum CART_ACTIONS {CLEFT, CRIGHT};


/*
 * function that take in the current real state of cart pole
 * and map that into a bucket number.
 */
int get_bucket(struct cart_state& cs);

void update_state(struct cart_state& cs, const double& force);

double get_theta_ddot(struct cart_state& cs, const double& force);
double get_x_ddot(struct cart_state& cs, const double& force);

// takes in state representation, and normalize and generate state representation
Eigen::VectorXd normalize_state(const struct cart_state& cs);

// generate Fourier basis state representation for cartpole problem
Eigen::VectorXd get_Fourier_basis(struct cart_state& cs,
                                  const int K);

Eigen::VectorXd get_fourier3(struct cart_state& cs);
Eigen::VectorXd get_fourier5(struct cart_state& cs);
Eigen::VectorXd get_fourier7(struct cart_state& cs);
Eigen::VectorXd get_fourier9(struct cart_state& cs);

Eigen::VectorXd get_polynomial_basis(struct cart_state& cs,
                                     const int K);
Eigen::VectorXd get_poly2(struct cart_state& cs);
Eigen::VectorXd get_poly3(struct cart_state& cs);
Eigen::VectorXd get_poly4(struct cart_state& cs);
Eigen::VectorXd get_poly5(struct cart_state& cs);

double run_TD_cartpole(Eigen::VectorXd& weights,
                       const int K,
                       const double step_size);

void cartpole_start_TD(const std::string filename);

/*
 * Function to run sarsa update for one episode
 * Input:
 * - qw: the weight function vector
 * - step_size: learning step_size
 * - explore_mode: 0 for epsilon greedy, 1 for softmax
 * - get_phi_s: function pointer to generate different function approximation
 *              to represent a cart pole state.
 */
double run_sarsa_cartpole(const int i,// the number of iteration in sarsa
                          Eigen::VectorXd& qw,
                          const double step_size,
                          const int explore_mode,
                          Eigen::VectorXd (*get_phi_s)(struct cart_state& cs));

/*
 * Perform epsilon greedy action selection when using
 * Function approximation
 * Inputs:
 * - epsilon: the main parameter to control how greedy the agent is.
 * - qw: the weight function approximation for choosing the best action
 * - phi: the representation of current state
 * For this implementation, do following steps
 */
int fa_epsilon_greedy_cartpole(const double epsilon,
                               const Eigen::VectorXd& qw,
                               const Eigen::VectorXd& phi);

/*
 * Function to perform softmax action selection
 */
int fa_softmax_get_action(const double thu,
                          const Eigen::VectorXd& qw,
                          const Eigen::VectorXd& phi);

double run_qlearning_cartpole(const int i, // holder, not really used
                              Eigen::VectorXd& qw, // q function weights
                              const double step_size,
                              const int param_holder, // holder, not used
                              Eigen::VectorXd (*get_phi_s)(struct cart_state& cs));

int get_max_action(const Eigen::VectorXd& qw,
                   const Eigen::VectorXd& phi);

/*
 * The function that calls the sarsa function approximation on cartpole
 */
void start_sarsa_cartpole();
void start_qlearning_cartpole();
#endif
