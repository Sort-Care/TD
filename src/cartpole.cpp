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

/*==================================== TD ======================================*/
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
  // std::cout << cs.theta << " " << THETA_MAX << " " << THETA_MIN << std::endl;
  // std::cout << "Normalized state: " << norm_state << std::endl;
  return norm_state;
}

/*
 * Generate Fourier Basis representation for cart pole problem
 * DONE:
 * 1. Initialization (state, phi)
 */
Eigen::VectorXd get_Fourier_basis(struct cart_state& cs,
                                  const int K){
  //std::cout << "Getting Fourier basis function :" << std::endl;
  // initialize state_phi (including normalization and assigning)
  Eigen::VectorXd state_phi = normalize_state(cs); // the result should be of length 4
  // get the Fourier basis matrix, result shape ((K+1)^4, 4)
  Eigen::MatrixXd basis = generate_fourier_multiplier(K, 4);

  // multiply Basis with state representation
  //std::cout << "Basis size: "<< basis.rows() << "," << basis.cols() << std::endl;
  //std::cout << "State_phi size: "<< state_phi.rows() << "," << state_phi.cols() << std::endl;
  Eigen::VectorXd inter_phi = basis * state_phi; // length, 4 x 4, = Length,
  inter_phi *= M_PI; // multiple by pi
  // return cosine
  return inter_phi.array().cos();
}

Eigen::VectorXd get_fourier3(struct cart_state& cs){
  return get_Fourier_basis(cs, 3);
}

Eigen::VectorXd get_fourier5(struct cart_state& cs){
  return get_Fourier_basis(cs, 5);
}

Eigen::VectorXd get_fourier7(struct cart_state& cs){
  return get_Fourier_basis(cs, 7);
}

Eigen::VectorXd get_fourier9(struct cart_state& cs){
  return get_Fourier_basis(cs, 9);
}

Eigen::VectorXd get_polynomial_basis(struct cart_state& cs,
                                     const int K){
  // 3 order polynomial
  Eigen::VectorXd state_phi = normalize_state(cs);
  // result shape (5^4, 4)
  int length = pow(K+1, 4);
  Eigen::MatrixXd basis = generate_fourier_multiplier(K, 4);
  Eigen::VectorXd result(length);
  REP (i, 0, length-1){
    double prod = 1.0;
    REP(j, 0, 3){
      prod *= pow(state_phi(j), basis(i, j));
    }
    result(i) = prod;
  }
  return result;
}

Eigen::VectorXd get_poly2(struct cart_state& cs){
  return get_polynomial_basis(cs, 2);
}

Eigen::VectorXd get_poly3(struct cart_state& cs){
  return get_polynomial_basis(cs, 3);
}

Eigen::VectorXd get_poly4(struct cart_state& cs){
  return get_polynomial_basis(cs, 4);
}

Eigen::VectorXd get_poly5(struct cart_state& cs){
  return get_polynomial_basis(cs, 5);
}



/*
 * Run one TD update on cartpole policy
 * Input:
 * - weights: the weights to be updated during this episode. Should be of
 *            shape (k+1)^4 - 1
 * - K: Fourier order
 * - step_size: learning rate of TD
 * Output:
 * - double: MSE of TD errors in this episode
 * NOTE: if value functions are perfect, then TD error should all
 *       be zero.
 */
double run_TD_cartpole(Eigen::VectorXd& weights,
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
  double MSE = 0.0;

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
    td_error = r + 0.95 * v_s_prime - v_s;
    MSE += td_error * td_error;
    // perform update
    weights += step_size * td_error * old_state_phi; // scalar * scalar * vector;
  }
  //std::cout << "End Episode with MSE td-error: "<< MSE << std::endl;
  return MSE;
}

void cartpole_start_TD(const std::string filename){
  int orders[2] = {3,5};
  // initialize step sizes
  double step_sizes[10];
  double MSE_results[2][10];
  REP(i, 0, 9){
    step_sizes[i] = 0.1 / pow(10, i);
    // third order
    MSE_results[0][i] = temporal_difference(filename,
                                            100,
                                            orders[0],
                                            4,
                                            step_sizes[i],
                                            run_TD_cartpole
                                            );
    // fifth order
    MSE_results[1][i] = temporal_difference(filename,
                                            100,
                                            orders[1],
                                            4,
                                            step_sizes[i],
                                            run_TD_cartpole
                                            );
  }

  // print the MSE array to console
  REP (i, 0, 1){
    //std::cout << "Order: "<< orders[i] << std::endl;
    REP (j, 0, 9){
      std::cout << step_sizes[j] << "\t" << MSE_results[i][j] << "\t" << std::endl;
    }
  }
}

/*================================= SARSA ===========================================*/
/*
 * Run One episode update for sarsa
 */
double run_sarsa_cartpole(const int i,
                          Eigen::VectorXd& qw,
                          const double step_size,
                          const int explore_mode,
                          Eigen::VectorXd (*get_phi_s)(struct cart_state& cs)){
  // initialize first state
  struct cart_state cs = {0, 0, 0, 0};
  int A_t, A_tn, S_t, S_tn; // action a and next action a'
  double time_so_far = 0.0;
  double r = 0.0;// instant rewards
  double target = 0.0;
  double q_s, q_s_prime;
  double td_error = 0.0;
  double epsilon = 0.8/(i+1); // parameter set for epsilon greedy
  double thu = 5;
  // get bucket representation for initial state.
  S_t = get_bucket(cs);
  // get phi
  Eigen::VectorXd st_phi = get_phi_s(cs);
  Eigen::VectorXd st_phin;
  int R = st_phi.size();
  // choose action from s using policy derived from Q (greedy or softmax)
  if (explore_mode == 0){
    // using epsilon greedy
    A_t = fa_epsilon_greedy_cartpole(epsilon, qw, st_phi);
  }else if (explore_mode == 1){
    // using softmax action selection
    A_t = fa_softmax_get_action(thu, qw, st_phi);
    if (A_t == -1) A_t = 0;
  }
  // Repeat until this episode ends
  while(S_t != -1 && time_so_far <= MAXTIME){
    // Perform action a, observe r, s^'
    update_state(cs, FORCES[A_t]);
    S_tn = get_bucket(cs);
    time_so_far += INTERVAL;
    // update the reward for the time step just happened
    if (S_tn != -1 && time_so_far <= MAXTIME) {
      r = 1;
      target += r;
    }else {
      r = 0;
    }
    // Choose a' from s' using policy derived from Q
    st_phin = get_phi_s(cs); // new state representation
    if (explore_mode == 0){
      A_tn = fa_epsilon_greedy_cartpole(epsilon, qw, st_phin);
    }else{
      A_tn = fa_softmax_get_action(1, qw, st_phin);
      if (A_tn == -1) A_tn = 0;
    }
    // Update the weight
    q_s = qw.segment(A_t * R, R).dot(st_phi);
    q_s_prime = qw.segment(A_tn * R, R).dot(st_phin);
    td_error = r + 1.0 * q_s_prime - q_s;
    // std::cout << "State phi: " << st_phi << std::endl;
    // std::cout << "Step size: " << step_size << std::endl;
    qw.segment(A_t * R, R) += step_size * td_error * st_phi;
    // Assign s' to s and a to a'
    S_t = S_tn;
    st_phi = st_phin;
    A_t = A_tn;
  }
  return target;
  /*
   * Things to check because it appears that it isn't learning:
   * 1. Correctness:
   *    - The environment implementation [X]
   *    - Action selection functions     [x]
   *    - State representation           [X]
   *    - Update rule computation        [!] FIXED
   * 2. Hyper-Parameter Tunning
   *    - step size
   *    - epsilon decaying function      [ ]
   *    - Fourier order
   */
}


int fa_epsilon_greedy_cartpole(const double epsilon,
                               const Eigen::VectorXd& qw,
                               const Eigen::VectorXd& phi){
  // first sample from (epsilon, 1-epsilon)
  double e_w[2] = {epsilon, 1-epsilon};
  int sampled_index = random_sample_weights(e_w, 2);
  int result_index = 0;
  if (sampled_index == 0){
    // uniformly random sample on Action space
     result_index = random_sample_weights(random_pi, NUM_LR);
  }else{
    // perform greedy action selection
    // use Eigen's function on vector: v.segment(i, n)
    // which is a block of n elements starting at position i
    int R = phi.size();
    double q_values[NUM_LR] = {0.0};
    // the qw should be of shape (|A|*R, 1)
    REP(i, 0, NUM_LR-1){ // for each action with respect to its q values
      q_values[i] = qw.segment(i*R, R).adjoint() * phi;
    }
    int maxq = 0;// the max q value's index
    REP(i, 0, NUM_LR-1){
      if (q_values[i] > q_values[maxq]){
        maxq = i;
      }
    }// end of finding maxq
    // now maxq is containing the action index with largest q value
    result_index = maxq;
  }
  return result_index;
}

int fa_softmax_get_action(const double thu,
                          const Eigen::VectorXd& qw,
                          const Eigen::VectorXd& phi){
  int R = phi.size();
  // for each action, compute its q value
  double smq[NUM_LR] = {0.0};
  double sum = 0.0;
  // then perform softmax calculation on it
  REP(i, 0, NUM_LR-1){
    double tmp = qw.segment(i*R, R).adjoint() * phi;
    tmp *= thu;
    smq[i] = exp(tmp);
    sum += smq[i];
  }
  REP(i, 0, NUM_LR-1){
    smq[i] = smq[i] / sum;
  }
  // and sample from the result array
  int result_action = random_sample_weights(smq, NUM_LR);
  return result_action;
}

/*==================================== Q-Learning =========================================*/

double run_qlearning_cartpole(const int i, // holder, not really used
                              Eigen::VectorXd& qw, // q function weights
                              const double step_size,
                              const int explore_mode, // explore mode
                              Eigen::VectorXd (*get_phi_s)(struct cart_state& cs)){
  // function for running q learning for one episode.
  // basically very similar to sarsa
  // two main differences:
  // 1. update upon s, a, r, s' acquired.
  // 2. choose the maximum q value action.

  // initialize variables
  int S_t, S_tn, A_t;
  double time_so_far = 0.0;
  double r = 0.0;
  double target = 0.0;
  double q_s, q_sp;
  double td_error = 0.0;
  double epsilon = 0.8 / (i+1);
  double thu = 5;
  struct cart_state cs = {0.0, 0.0, 0.0, 0.0};

  S_t = get_bucket(cs);
  Eigen::VectorXd st_phi = get_phi_s(cs);
  Eigen::VectorXd st_phin;
  int R = st_phi.size();
  while(S_t != -1 && time_so_far < MAXTIME){
    // choose action using epsilon greedy
    if (explore_mode == 0){
      A_t = fa_epsilon_greedy_cartpole(epsilon, qw, st_phi);
    }else{
      A_t = fa_softmax_get_action(thu, qw, st_phi);
      if (A_t == -1) A_t = 0;
    }
    // take that action and update the environment
    update_state(cs, FORCES[A_t]);
    S_tn = get_bucket(cs);
    time_so_far += INTERVAL;
    if (S_tn != -1 && time_so_far <= MAXTIME){
      r = 1;
      target += r;
    }else{
      r = 0;
    }
    st_phin = get_phi_s(cs);
    // for updating, we need to find the maximum q value function at state s'
    int A_max = get_max_action(qw, st_phin);
    q_s = qw.segment(A_t * R, R).dot(st_phi);
    q_sp = qw.segment(A_max * R, R).dot(st_phin);
    td_error = r + 1.0 * q_sp - q_s;
    qw.segment(A_t * R, R) += step_size * td_error * st_phi;
    S_t = S_tn;
    st_phi = st_phin;
  }
  return target;
}

int get_max_action(const Eigen::VectorXd& qw,
                   const Eigen::VectorXd& phi){
  int R = phi.size();
  int maxIdx = 0;
  double maxValue = 0.0;
  REP(i, 0, NUM_LR-1){
    double tmp = qw.segment(i*R, R).dot(phi);
    if (tmp > maxValue){
      maxValue = tmp;
      maxIdx = i;
    }
  }
  return maxIdx;
}


void start_sarsa_cartpole(){
  // get called in main
  int num_episodes[2] = {100, 250};
  // search space for step_sizes
  double step_sizes[10];
  REP(i, 0, 9){
    step_sizes[i] = 0.1 / pow(10, i);
  }
  // function approximation state representation search space
  int fourier_orders[4] = {3,5,7,9};
  Eigen::VectorXd (*get_fouriers[])(struct cart_state& cs) = {
    get_fourier3,
    get_fourier5,
    get_fourier7,
    get_fourier9
  };
  int q_weight_sizes[4];
  REP(i, 0, 3){
    // the 4 here is because the cart has four fields in its state representation
    q_weight_sizes[i] = pow((fourier_orders[i] + 1), 4) * NUM_LR;
  }
  int explore_modes[2] = {0, 1};

  Eigen::MatrixXd result(100, 100);
  REP(i, 0, 99){
    result.row(i) = on_policy_function_approx(num_episodes[0],
                                              0.0003,
                                              explore_modes[1],
                                              q_weight_sizes[0],
                                              run_sarsa_cartpole,
                                              get_fouriers[0]);
  }
  std::ofstream mfile;
  mfile.open("cp_sarsa_sm.dat");
  mfile << result << '\n';
  mfile.close();
  /*
   * For action selection:
   * 1. Epsilon greedy
   *    - step_sizes[3]
   */
}

void start_qlearning_cartpole(){
  int num_episodes[2] = {100, 200};
  double step_sizes[5];
  REP(i, 0, 4){
    step_sizes[i] = 0.1 / pow(10, i);
  }
  // function approximation state representation search space
  int fourier_orders[4] = {3,5,7,9};
  Eigen::VectorXd (*get_fouriers[])(struct cart_state& cs) = {
    get_fourier3,
    get_fourier5,
    get_fourier7,
    get_fourier9
  };
  Eigen::VectorXd (*get_polys[])(struct cart_state& cs) = {
    get_poly2,
    get_poly3,
    get_poly4,
    get_poly5
  };
  int q_weight_sizes[4];
  REP(i, 0, 3){
    // the 4 here is because the cart has four fields in its state representation
    q_weight_sizes[i] = pow((fourier_orders[i] + 1), 4) * NUM_LR;
  }

  int poly_sizes[] = {
    81 * NUM_LR,
    256 * NUM_LR,
    625 * NUM_LR,
    1296 * NUM_LR
  };

  Eigen::MatrixXd result(100, 100);
  REP(i, 0, 99){
    result.row(i) = on_policy_function_approx(num_episodes[0],
                                              0.01,// 1e-4 ~ 1e-5 0.001
                                              0,
                                              q_weight_sizes[0],
                                              // poly_sizes[3],
                                              run_qlearning_cartpole,
                                              get_fouriers[0]
                                              // get_polys[3]
                                              );
  }
  std::ofstream mfile;
  mfile.open("cp_sarsa_pl.dat");
  mfile << result << '\n';
  std::cout << "Dumped to file" << std::endl;
  mfile.close();
}
