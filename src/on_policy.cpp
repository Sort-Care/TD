#include "on_policy.hpp"

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
                             )
{
  // initialize the weight vector
  Eigen::VectorXd qw = Eigen::VectorXd::Zero(q_weight_size);
  // data structure to record the target returns running every episode
  Eigen::RowVectorXd res = Eigen::RowVectorXd::Zero(num_episodes);
  // repeat for number of iterations
  REP(i, 0, num_episodes-1){
    // run the update for one episode
    res(i) = sarsa_fa(i, qw, step_size, explore_mode, get_phi_s);
    std::cout<< "Iteration :"<< i << "\t" << res(i) << std::endl;
  }
  // std::cout << qw << std::endl;
  return res;
}

Eigen::RowVectorXd on_policy_tabular(const int num_episodes,
                     const double step_size,
                     const int explore_mode,
                     const int q_table_size,
                     double (*sarsa_tb)(const int episode,
                                        Eigen::VectorXd& qf,
                                        const double step_size,
                                        const int explore_mode)
                     )
{
  Eigen::VectorXd qf = Eigen::VectorXd::Zero(q_table_size);
  // data structure for recording returns
  double target_return[num_episodes] = {0.0};
  Eigen::RowVectorXd res = Eigen::RowVectorXd::Zero(num_episodes);
  // repeat
  REP(i, 0, num_episodes-1){
    // run sarsa tabular update for one episode
    res(i) = sarsa_tb(i, qf, step_size, explore_mode);
    std::cout<< "Iteration :"<< i << "\t" << res(i) << std::endl;
  }
  return res;
}

