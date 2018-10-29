#include <random>
#include <chrono>
#include <cmath>

#include "random_sampling.hpp"


int random_sample_distribution(const double distribution[], const int size){
        //check if the array adds up to 1
    double sum = 0.0;
    REP (i, 0, size-1){
        printf("%.2f ", distribution[i]);
        sum += distribution[i];
    }
    printf("\n");
    
    

    if(sum == 1.0){//valid
            // will generate [0,1]
        double rnum = random_zero_to_one();
        printf("%f\n", rnum);
        
        REP(i, 0, size-1){
            printf("%d: %.2f\t rnum: %.2f\n", i, distribution[i], rnum);
            if (rnum < distribution[i]) return i;
            else{
                rnum -= distribution[i];
            }
        }
    }else{
        return(-1);//not valid return an index that doesn't make sense.
    }
    return(-1);
}

int random_sample_weights(const double weights[], const int size){
        //first compute the sum of all weights
    double sum = 0.0;
    REP (i, 0, size-1){
        sum += weights[i];
            //printf("Adding %.2f\n", weights[i]);
    }
        //printf("%.2f\n", sum);
    double rnum = random_range(sum);
        //printf("%.2f\n", rnum);
    REP (i, 0, size-1){
        if(rnum < weights[i]) return i;
        else{
            rnum -= weights[i];
        }
    }
    return (-1);
    
}

int random_sample_eigen_vectors(const Eigen::VectorXd& vec){
    double sum = vec.sum();
    double rnum = random_range(sum);
    REP (i, 0, vec.rows() - 1){
        if(rnum < vec(i,0)) return i;
        else {
            rnum -= vec(i,0);
        }
    }
    return (-1);
}

double random_range(double range){
    std::mt19937_64 rng;
        // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
        // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, range);
        // ready to generate random numbers
    const int nSimulations = 10;
    double currentRandomNumber = unif(rng);
    return currentRandomNumber;
}

double random_zero_to_one(){
    std::mt19937_64 rng;
        // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
        // initialize a uniform distribution between 0 and 1
    std::uniform_real_distribution<double> unif(0, 1);
        // ready to generate random numbers
    const int nSimulations = 10;
    double currentRandomNumber = unif(rng);
    return currentRandomNumber;
}
