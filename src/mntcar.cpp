#include "mntcar.hpp"

const double LEFT_END = -1.2;// left boundary
const double RIGHT_END = 0.5;// right boundary
const double CAR_GOAL = 0.5; // Goal position
const double CAR_REWARD = -1;// Reward
const double CAR_INIT_X = - 0.5;// initial state position
const double CAR_INIT_V = 0.0;// initial state velocity
const double V_MAX = 0.07;
const double V_MIN = - 0.07;
const int CAR_ACTIONS[3] = {-1, 0, 1};// actions {reverse, neutral, forward}


bool is_goal_reached(const struct car_state& cs){
  if(cs.x >= 0.5) return true;
  else return false;
}

void update_car_state(struct car_state& cs,
                      const int action){
  cs.v += 0.001 * action - 0.0025 * cos(3 * cs.x);
  cs.x += cs.v;
  // the following cases simulate the collision with wall
  if(cs.x < LEFT_END){
    cs.x = LEFT_END;
    cs.v = 0;
  }
  if (cs.x > RIGHT_END){
    cs.x = RIGHT_END;
    cs.v = 0;
  }
}

Eigen::VectorXd normalize_car_state(const struct car_state& cs)
{
  Eigen::VectorXd normed(2); // state size 2 because it includes position and velocity
  normed(0) = min_max_norm(cs.x, RIGHT_END, LEFT_END);
  normed(1) = min_max_norm(cs.v, V_MAX, V_MIN);
  return normed;
}

double sarsa_lambda_mntcar(const int episode,
                           Eigen::VectorXd& qw,
                           const double step_size,
                           const int explore_mode,
                           Eigen::VectorXd (*get_phi_s)(struct car_state& cs))
{
  return 0.0;
}

double ql_lambda_mntcar(const int episode,
                        Eigen::VectorXd& qw,
                        const double step_size,
                        const int explore_mode,
                        Eigen::VectorXd (*get_phi_s)(struct car_state& cs)
                        )
{
  return 0.0;
}

double ac_mntcar(const int episode,
                 Eigen::VectorXd& weights,
                 Eigen::VectorXd& theta_param,
                 const double step_size,
                 const double lambda
                 )
{
  return 0.0;
}
