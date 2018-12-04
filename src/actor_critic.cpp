#include "actor_critic.hpp"

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
                                )
{
  Eigen::VectorXd weights = Eigen::VectorXd::Zero(weight_size);
  Eigen::VectorXd theta_param = Eigen::VectorXd::Zero(theta_size);

  Eigen::RowVectorXd res = Eigen::RowVectorXd::Zero(num_episodes);
  REP(i, 0, num_episodes-1){
    res(i) = ac_prob(i,
                     weights,
                     theta_param,
                     step_size,
                     lambda);
    std::cout << "Iteration: "<< i << "\t" << res(i) << std::endl;
  }
  return res;
}
