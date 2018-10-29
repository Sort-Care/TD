#ifndef __CART_POLE
#define __CART_POLE
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>

#include "bbo.hpp"
#include "random_sampling.hpp"

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

double run_cartpole_on_policy(struct policy& po);

Eigen::MatrixXd cartpole_softmax(struct policy& po,
                                 const int rows,
                                 const int cols);

void run_cross_entropy_on_cartpole();

void run_FCHC_on_cartpole();


#endif
