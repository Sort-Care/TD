#include <algorithm>
#include <Eigen/Dense>
#include <iostream>

#include "grid.hpp"
#include "cartpole.hpp"
#include "random_sampling.hpp"
#include "fourier_td.hpp"
#include "tabular_td.hpp"
#include "on_policy.hpp"

int main(int argc, char *argv[]){
  if (argc != 2){
    std::cout << "Invalid command line arguments";
    return 0;
  }
   char* opt = argv[1];
   if(strcmp(opt, "gwq") == 0){
     start_qlearning_gw();
  }else if (strcmp(opt, "gws") == 0){
     start_sarsa_gw();
   }else if (strcmp(opt, "cpq") == 0){
     start_qlearning_cartpole();
   }else if (strcmp(opt, "cps") == 0){
     start_sarsa_cartpole();
   }else{
    std::cout << "Unknown arguments, aborting." << std::endl;
    return 0;
  }

  return 0;
}
