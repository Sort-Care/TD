#ifndef __GRID_WORLD
#define __GRID_WORLD

#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <algorithm>
#include <Eigen/Dense>
#include <fstream>

#include "tabular_td.hpp"
#include "on_policy.hpp"

//Total number of states
extern const int STATE_NUM;
extern const int goal_state;
extern const int absorbing_state; //only accessible in transition table
extern const int NEG_INF;

//Grid World Size by rows x columns
extern const int row;
extern const int column;
extern const int NUM_ACTION; //AU, AD, AL, AR
extern const int NUM_OUTCOME;
extern const int X;
extern const int Y;



// Four actions
enum ACTIONS {AU, AD, AL, AR};//0, 1, 2, 3
enum OUTCOMES {SUC, STAY, VL, VR}; // with probabilities: 0.8, 0.1, 0.05, 0.05




/***** Data structures for simulating an agent in the gridworld ******/
extern const int NUM_EPISODES;
extern const double dis_gamma;


/*** Functions ****/
void generateInput();

int get_random_action(int total_num_actions);

void print_normal();

void print_for_py();

void value_iteration();

double simulate_random();// run simulation randomly choose actions

double simulate_optimal();// run simulation using optimal policy

void run_simulation_with_strategy(double (*f)());

double estimate_quantity();


void get_array_statistics(const double array[], const int size);

void get_standard_deviation(const double array[], const int size,
                            double *amean,
                            double *adevia);

double run_TD_gridworld(Eigen::VectorXd& vf,
                        const double step_size);

double gw_start_TD();

double run_sarsa_gridworld(const int episode,
                           Eigen::VectorXd& qf,
                           const double step_size,
                           const int explore_mode);

double run_qlearning_gridworld(const int episode,
                               Eigen::VectorXd& qf,
                               const double step_size,
                               const int explore_mode);

int tabular_epsilon_greedy(const double epsilon,
                           const Eigen::VectorXd& qf,
                           const int S_t);

int tabular_softmax(const double thu,
                    const Eigen::VectorXd& qf,
                    const int S_t);

int get_best_action(const Eigen::VectorXd& qf,
                    const int S_t);

void start_sarsa_gw();
void start_qlearning_gw();

double run_ql_lambda_gw()
#endif
