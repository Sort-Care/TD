#include "cartpole.hpp"

/*
 * Environment Set up
 */
const double FAIL_ANGLE = M_PI/2; // fail angle, if exceeds (-M_PI/2, M_PI/2)
const double FORCE = 10.0;        // Motor force, Newton
const double GRAV = 9.8;          // Gravitational constant
const double CART_M = 1.0;        // Cart mass
const double POLE_M = 0.1;        // Pole mass
const double POLE_L = 0.5;        // Pole half length
const double INTERVAL = 0.02;     // time step
const double MAXTIME = 20.2;      // Max time before end of episode

// The following are here for normalizing state representation
// in function approximation.
const double X_MAX = 3;
const double X_MIN = -3;
const double X_DOT_MAX = 10;
const double X_DOT_MIN = -10;
const double THETA_MAX = M_PI/2;
const double THETA_MIN = - M_PI/2;
const double THETA_DOT_MAX = M_PI;
const double THETA_DOT_MIN = - M_PI;

const int NUM_BUCKETS = 162;
const int NUM_LR = 2;


const double FORCES[2] = {-FORCE, FORCE};
const double random_pi[2] = {1, 1};

/*
 * Discretize the continuous state into buckets
 * The original state has:
 * 1. position :  [-3.0. 3.0]
 * 2. velocity :  (-inf, inf)
 * 3. pole angle: (-M_PI/2, M_PI/2)
 * 4. angular velocity: (-inf, inf)
 */

/*
 * function that take in the current real state of cart pole
 * and map that into a bucket number.
 */
int get_bucket(struct cart_state& cs)
{
  /*
   * THIS FUNCTION ACTUALLY DEFINES HOW WE SEPERATE THE BUCKETS
   */
  int bucket = 0;
  if (cs.x < -3 ||
      cs.x > 3 ||
      cs.theta < - FAIL_ANGLE ||
      cs.theta > FAIL_ANGLE) return (-1);// end of episode


  /*
   *|--------------- (-1) ------- 0 -------- 1 -----------------|
   *    BUCKET 0              BUCKET 1             BUCKET 2
   */
  if (cs.x < -1)				bucket = 0;
  else if (cs.x < 1)			bucket = 1;
  else						bucket = 2;

  if (cs.x_dot < -0.6)		;
  else if (cs.x_dot < 0.6)	bucket += 3;
  else						bucket += 6;

  if (cs.theta < - FAIL_ANGLE / 15)		;
  else if(cs.theta < -FAIL_ANGLE / 90)	bucket += 9;
  else if(cs.theta < 0)					bucket += 18;
  else if(cs.theta < FAIL_ANGLE / 90)		bucket += 27;
  else if(cs.theta < FAIL_ANGLE / 15)		bucket += 36;
  else									bucket += 45;

  if (cs.theta_dot < - FAIL_ANGLE / 2)	;
  else if (cs.theta_dot < FAIL_ANGLE / 2) bucket += 54;
  else									bucket += 108;

  return (bucket);
}

void update_state(struct cart_state& cs,
                  const double& force){
  /*
   * This function updates the state give current state.
   */
  double theta_ddot = get_theta_ddot(cs,force);
  double x_ddot = get_theta_ddot(cs,force);

  cs.x += cs.x_dot * INTERVAL + x_ddot * INTERVAL * INTERVAL / 2.0;
  cs.x_dot += x_ddot * INTERVAL;

  cs.theta += cs.theta_dot * INTERVAL + theta_ddot * INTERVAL * INTERVAL / 2.0;
  cs.theta_dot += theta_ddot * INTERVAL;
}

double get_theta_ddot(struct cart_state& cs, const double& force){
  double numerator = GRAV * std::sin(cs.theta) +
    std::cos(cs.theta) * ((-force - POLE_M * POLE_L *
                           cs.theta_dot * cs.theta_dot * std::sin(cs.theta))
                          / (CART_M + POLE_M));

  double denominator = POLE_L * (4.0/3 - (POLE_L * std::cos(cs.theta)
                                          * std::cos(cs.theta))/(POLE_M + CART_M));

  return numerator / denominator;
}


double get_x_ddot(struct cart_state& cs, const double& force){
  return (force + POLE_M * POLE_L * (cs.theta_dot * cs.theta_dot * std::sin(cs.theta)-
                                     get_theta_ddot(cs,force) * std::cos(cs.theta)))
    / (CART_M + POLE_M);
}

/*
 * Get states and normalize cart pole states and return an Eigen::VectorXd
 * DONE: normalize cart pole state
 */
Eigen::VectorXd normalize_state(const struct cart_state& cs){
  Eigen::VectorXd norm_state(4);
  // normalize for: (using min-max normalization)
  // 1. position
  norm_state(0) = min_max_norm(cs.x, X_MAX, X_MIN);
  // 2. velocity
  norm_state(1) = min_max_norm(cs.x_dot, X_DOT_MAX, X_DOT_MIN);
  // 3. angle
  norm_state(2) = min_max_norm(cs.theta, THETA_MAX, THETA_MIN);
  // 4. angular velocity
  norm_state(3) = min_max_norm(cs.theta_dot, THETA_DOT_MAX, THETA_DOT_MIN);
  return norm_state;
}

/*
 * Generate Fourier Basis representation for cart pole problem
 * DONE:
 * 1. Initialization (state, phi)
 */
Eigen::VectorXd get_Fourier_basis(struct cart_state& cs,
                                  const int K){
  // initialize state_phi (including normalization and assigning)
  Eigen::VectorXd state_phi = normalize_state(cs); // the result should be of length 4
  // get the Fourier basis matrix
  Eigen::MatrixXd basis = generate_fourier_multiplier(K, 4);

  // multiply Basis with state representation
  Eigen::VectorXd inter_phi = basis.dot(state_phi); // length, 4 x 4, = Length,
  inter_phi *= M_PI; // multiple by pi
  // return cosine
  return inter_phi.cos();
}

/*
 * Run TD update on cartpole policy
 * Input:
 * - weights: the weights to be updated during this episode. Should be of
 *            shape (k+1)^4 - 1
 * - K: Fourier order
 * - step_size: learning rate of TD
 */
void run_TD_cartpole(Eigen::VectorXd& weights,
                     const int K,
                     const double step_size){
  // initialize s_0
  struct cart_state cs = {0, 0, 0, 0};
  int A_t;
  double time_so_far = 0.0; // time recorded for ending episode
  double r = 0.0; // reward got at time t
  double v_s;
  double v_s_prime;
  double td_error = 0.0;

  int S_t = get_bucket(cs);
  while(S_t != -1 && time_so_far <= MAXTIME){
    // using Random policy, select random action in {LEFT, RIGHT}
    A_t = random_sample_weights(random_pi, NUM_LR);
    // before update, get the old state value function
    Eigen::VectorXd old_state_phi = get_Fourier_basis(cs, K);
    v_s = weights.dot(old_state_phi); // (L,).(L,) = scalar
    // update the underlying state, given action's force
    update_state(cs, FORCES[A_t]);
    // update time
    time_so_far += INTERVAL;
    // get current state bucket
    S_t = get_bucket(cs);
    // Get r
    if (S_t != -1 && time_so_far <= MAXTIME){ // not failing and time in horizon
      r = 1;
      v_s_prime = weights.dot(get_Fourier_basis(cs, K));
    }else{
      r = 0;
      v_s_prime = 0;
    }
    // Compute TD error, assuming gamma = 1.0
    td_error = r + v_s_prime - v_s;
    // perform update
    weights += step_size * td_error * old_state_phi; // scalar * scalar * vector;
  }
}
