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

const int NUM_BUCKETS = 162;
const int NUM_LR = 2;


const double FORCES[2] = {-FORCE, FORCE};

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
   * 
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


double run_cartpole_on_policy(struct policy& po){

  // get the policy probability table given the policy parameters
  // that is: run the cartpole_softmax and get a returned Eigen::MatrixXd
  // which in cart pole case should have shape (2, 162)
  // where each column is a state
  Eigen::MatrixXd pi = cartpole_softmax(po,NUM_LR,NUM_BUCKETS);
    
  // initialize the environment
  // by setting x, x_dot, theta, theta_dot to zero
  // ALSO!! set time to zero!!!
  struct cart_state cs = {0.0,
                          0.0,
                          0.0,
                          0.0};

  double time_so_far = 0.0;
  double reward = 0.0;

  int S_t, S_tn, A_t;
  S_t = get_bucket(cs);// initially this should be 85
  while (S_t != -1 && time_so_far <= MAXTIME){
    // sample an action
    A_t = random_sample_eigen_vectors(pi.col(S_t));
    // update the state according
    update_state(cs,FORCES[A_t]);
    // decide if the episode is over or not
    S_tn = get_bucket(cs);
    // update time
    time_so_far += INTERVAL;
    // get reward
    //std::cout << "Bucket: " << S_t << "\tNext Bukt: " << S_tn << "\tAction: " << A_t << "\t Time: " << time_so_far << std::endl;
        
    if (S_tn != -1 && time_so_far <= MAXTIME){
      // get reward
      reward += 1;
      S_t = S_tn;
    }else{
      break;
    }
    // loop until episode ends
  }
  // then return the rewards
  return reward;
}


void run_cross_entropy_on_cartpole(){
  int cart_size = NUM_LR * NUM_BUCKETS; // 2 * 162 = 324
  Eigen::VectorXd theta;
  Eigen::MatrixXd cov;

  int K = 15;
  int E = 1;
  int N = 1;
  double epsi = 2;

  std::vector<std::future<void>> futures;
  REP(i, 0, 99){
    theta = Eigen::VectorXd::Zero(cart_size);
    cov = Eigen::MatrixXd::Constant(cart_size,
                                    cart_size,
                                    1);

    futures.push_back(std::async(std::launch::async,
                                 [&]{
                                   return cross_entropy("CCE2",
                                                        i,
                                                        cart_size,
                                                        theta,
                                                        cov,
                                                        K,
                                                        E,
                                                        N,
                                                        epsi,
                                                        eval_cart_pole_policy);
                                 }));
        
  }

  for(auto& e : futures){
    e.get();
  }

  // cross_entropy(1,
  //               cart_size,
  //               theta,
  //               cov,
  //               K,
  //               E,
  //               N,
  //               epsi,
  //               eval_cart_pole_policy);
}


/*
 * Transfer theta to pi: R^n --> [0,1]^n
 */
Eigen::MatrixXd cartpole_softmax(struct policy& po,
                                 const int rows, //NUM_ACTION
                                 const int cols){//STATE_NUM
  //first reshape
  Eigen::Map<Eigen::MatrixXd> reshaped(po.param.data(), rows, cols);
  // then apply softmax
  Eigen::MatrixXd soft = reshaped.array().exp();
  // then normalize
  Eigen::RowVectorXd col_mean = soft.colwise().sum();
  //replicate before division
  Eigen::MatrixXd repeat = col_mean.colwise().replicate(rows);
  return soft.array() / repeat.array();
}


void run_FCHC_on_cartpole(){
  int cart_size = NUM_LR * NUM_BUCKETS; // 92
    
  Eigen::VectorXd theta;

  double tau = 2;
  int N = 1;
  //std::cout << "episode" <<'\t' << "return" << std::endl;
  std::vector<std::future<void>> futures;
    
  REP (i, 0, 499){
    theta = Eigen::VectorXd::Zero(cart_size);
    futures.push_back(std::async(std::launch::async,
                                 [&]{return hill_climbing("CP",
                                                          i,
                                                          cart_size,
                                                          theta,
                                                          tau,
                                                          N,
                                                          eval_cart_pole_policy);}));
  }
}
