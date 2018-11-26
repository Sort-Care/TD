#include "grid.hpp"
#include "random_sampling.hpp"

//Total number of states
const int STATE_NUM = 23;
const int goal_state = 23;
const int absorbing_state = 24; //only accessible in transition table
const int NEG_INF = -100000;

//Grid World Size by rows x columns
const int row = 5;
const int column = 5;
const int NUM_ACTION = 4; //AU, AD, AL, AR
const int NUM_OUTCOME = 4;
const int X = 0;
const int Y = 1;



// Four actions
const int coordinate_change[NUM_ACTION][NUM_OUTCOME][2] = {
  {//AU
    {-1, 0}, //SUC
    {0, 0},  //STAY
    {0, -1}, //VL
    {0, 1}   //VR
  },
  {//AD
    {1, 0},
    {0, 0},
    {0, 1},
    {0, -1}
  },
  {//AL
    {0, -1},
    {0, 0},
    {1, 0},
    {-1, 0}
  },
  {//AR
    {0, 1},
    {0, 0},
    {-1, 0},
    {1, 0}
  }
};


//Grid World Model Probabilities on moving effects.
const double probs[NUM_OUTCOME] = {0.8, 0.1, 0.05, 0.05};

const double GW[row][column] = {// rewards
  {0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, 0.0, 0.0, 0.0},
  {0.0, 0.0, NEG_INF, 0.0, 0.0},
  {0.0, 0.0, NEG_INF, 0.0, 0.0},
  {0.0, 0.0, -10.0, 0.0, 10.0}
};

//map from state number to coordinates
const int state_to_coor[STATE_NUM][2] = {
  {0, 0},//1
  {0, 1},
  {0, 2},
  {0, 3},
  {0, 4},
  {1, 0},
  {1, 1},
  {1, 2},
  {1, 3},
  {1, 4},
  {2, 0},//11
  {2, 1},
  {2, 3},
  {2, 4},
  {3, 0},//15
  {3, 1},
  {3, 3},//17
  {3, 4},
  {4, 0},//19
  {4, 1},//20
  {4, 2},//21
  {4, 3},//22
  {4, 4}// 23
};

const int coor_to_state[row][column] = {
  {1,2,3,4,5},
  {6,7,8,9,10},
  {11,12,NEG_INF,13,14},
  {15,16,NEG_INF,17,18},
  {19,20,21,22,23}
};

double trans_table[(STATE_NUM+1) * NUM_ACTION][STATE_NUM+1] = {0.0};


/***** Data structures for simulating an agent in the gridworld ******/
const int NUM_EPISODES = 10000;
const double dis_gamma = 0.9;

double episode_reward[NUM_EPISODES]; //discounted reward

const double pr_actions[NUM_ACTION] = {0.25, 0.25, 0.25, 0.25};
const double d_0[STATE_NUM] = {//23 states not including the absorbing state
  1,1,1,1,1,
  1,1,1,1,1,
  1,1,1,1,
  1,1,1,1,
  1,1,1,1,1
};
const double pr_action_outcome[4] = {0.8, 0.1, 0.05, 0.05};


// optimal policy using last semester 683 value iteration python code
const int optimal_policy[STATE_NUM] = {// AU:0, AD:1, AL:2, AR:3
  3,3,3,1,1,
  3,3,3,1,1,
  0,0,1,1,
  0,0,1,1,
  0,0,3,3,0
};

int s21_cnt = 0;// estimate S_19 = 21
int s18_cnt = 0;// given S_8 = 18


void generateInput(){
  //initialize all to 0.0
  //printf("Trans table row: %d, column: %d\n", STATE_NUM*NUM_ACTION, STATE_NUM);
  REP (s, 0, STATE_NUM-1){//Genrate Transition probability row for state_i
    if(s == goal_state-1){//skip the one for goal_state
      continue;
    }
    REP (a, 0, NUM_ACTION-1 ){//for each action
      REP (o, 0, NUM_OUTCOME-1 ){//For each outcome
        /*
          Code Block for
          STATE: s
          ACTION: a
          OUTCOME: o
        */
        //compute current position, using the map
        // printf("In LOOP: For state: %d, Action: %d, Outcome: %d\n",
        //        s,
        //        a,
        //        o);
        int x, y;
        x = state_to_coor[s][X];
        y = state_to_coor[s][Y];

        int off_x, off_y;
        off_x = coordinate_change[a][o][X];
        off_y = coordinate_change[a][o][Y];

        //printf("X: %d, Y: %d, OFF_X: %d, OFF_Y: %d\n", x,y,off_x,off_y);

        //tell if the coordinates are valid or not, if not shift to original
        int new_x, new_y;
        new_x = x + off_x;
        new_y = y + off_y;
        if (new_x < 0 || new_x > 4 || new_y < 0 || new_y > 4 || GW[new_x][new_y] == NEG_INF){// invalid new position
          //reset to original, which means not moved
          new_x = x; new_y = y;
        }
        // printf("New_x: %d\t New_y: %d \n",
        //        new_x,
        //        new_y);
        //compute probability
        int result_state = coor_to_state[new_x][new_y];

        //update trans_table[s * NUM]
        trans_table[s * NUM_ACTION + a][result_state-1] += probs[o];
        // printf("Updating transition table at (%d, %d) adding prob %f\n",
        //        s*NUM_ACTION+a,
        //        result_state-1,
        //        probs[o]);
      }
      //print trans_table[s*a][ALL]
    }
  }

  //modify the transition table according to the terminal state

  //first modify the goals transition to always go to absorbing state
  REP (i, 0, NUM_ACTION-1){
    trans_table[(goal_state-1)*NUM_ACTION + i][absorbing_state-1] = 1.0;
    trans_table[(absorbing_state-1)*NUM_ACTION + i][absorbing_state-1] = 1.0;
  }

  //print_normal();
}



void print_normal(){
  printf("%d\n", STATE_NUM+1);//first line number of state
  REP (i, 0, STATE_NUM-1){//the next STATE_NUM line: rewards for entering each state
    printf("%.1f\n", GW[state_to_coor[i][X]][state_to_coor[i][Y]]);
  }
  printf("%.2f\n", 0.00);//absorbing_state
  REP (i, 0, (STATE_NUM+1)*NUM_ACTION-1){// next STATE_NUM * NUM_ACTION lines: Transition table
    REP (j, 0, STATE_NUM){
      //            printf("(%d, %d): ",i, j);
      printf("%.2f, ", trans_table[i][j]);
    }
    printf("\n");
  }
}


void print_for_py(){
  int action_order[4] = {2, 0, 3, 1};//U, D, L, R  0, 1, 2, 3, want L, U, R, D
  //print to run value iteration with a python file
  printf("%d\n", STATE_NUM+1);
  REP (i, 0, STATE_NUM-1){//the next STATE_NUM line: rewards for entering each state
    printf("%.1f\n", GW[state_to_coor[i][X]][state_to_coor[i][Y]]);
  }

  printf("%.2f\n", 0.00);
  REP (i, 0, NUM_ACTION-1){// 0, 1, 2, 3  No L, U, R, D
    //for each actions print for every state, here it should be printing
    // 24 lines
    REP (j, 0, STATE_NUM){ //0, 1, ..., 23
      //print trans_table[(j-1)*NUM_ACTION+i][all]
      REP(k, 0, STATE_NUM){
        printf("%.2f", trans_table[j*NUM_ACTION+action_order[i]][k]);
        if (k != STATE_NUM) printf(", ");
        else{
          printf("\n");
        }
      }
    }
  }
}


double simulate_random(){
  //get s0 as a start state
  int start_state = random_sample_weights(d_0, 23);
  int S_t, A_t, S_tn;
  double discounted_reward = 0.0;
  int cnt = 0;

  S_t = start_state;
  while (S_t != absorbing_state-1){// nor absorbing state
    //sample an action randomly
    //printf("Current State: %d\t", S_t);
    A_t = random_sample_weights(pr_actions, 4);
    //printf("Picked Action: %d\t", A_t);
    //sample the next state given
    S_tn = random_sample_weights(trans_table[S_t * NUM_ACTION + A_t], STATE_NUM+1);
    //printf("Next State: %d\n", S_tn);
    //Get reward
    discounted_reward += pow(dis_gamma, cnt) * GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]];
    cnt ++;
    //printf("discounted: %f\t Reward:%.2f\n", pow(dis_gamma, cnt),
    //GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]]);

    //update the discounted reward
    S_t = S_tn;
  }

  return discounted_reward;
  //printf("%.8f\n", discounted_reward);
}


double estimate_quantity(){
  REP(i, 0, NUM_EPISODES-1){
    printf("\rEpisode: %d", i);
    int start_state = random_sample_weights(d_0, 23);
    int S_t, A_t, S_tn;
    int cnt = 0;
    bool valid_trial = false;
    S_t = start_state;
    while (S_t != absorbing_state-1){// nor absorbing state
      //sample an action randomly
      //printf("Current State: %d\t", S_t);
      A_t = random_sample_weights(pr_actions, 4);
      //printf("Picked Action: %d\t", A_t);
      //sample the next state given
      S_tn = random_sample_weights(trans_table[S_t * NUM_ACTION + A_t], STATE_NUM+1);
      //printf("Next State: %d\n", S_tn);
      //Get reward
      //discounted_reward += pow(dis_gamma, cnt) * GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]];
      cnt ++;

      if(cnt == 8 && S_tn == 17){
        //                printf("\n adding one to s18");
        s18_cnt++;
        valid_trial = true;
      }
      if(valid_trial && cnt == 19 && S_tn == 20){
        //                printf("\nadding one to s21 given s18");
        s21_cnt++;
      }

      //printf("discounted: %f\t Reward:%.2f\n", pow(dis_gamma, cnt),
      //GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]]);

      //update the discounted reward
      S_t = S_tn;
    }
  }
  printf("\ns21_cnt: %d, s18_cnt: %d\n", s21_cnt, s18_cnt);
  return ((double)s21_cnt / s18_cnt);
}


double simulate_optimal(){
  //get s0 as a start state
  int start_state = random_sample_weights(d_0, 23);
  int S_t, A_t, S_tn;
  double discounted_reward = 0.0;
  int cnt = 0;

  S_t = start_state;
  while (S_t != absorbing_state-1){// nor absorbing state
    //sample an action randomly
    //printf("Current State: %d\t", S_t);
    A_t = optimal_policy[S_t];
    //printf("Picked Action: %d\t", A_t);
    //sample the next state given
    S_tn = random_sample_weights(trans_table[S_t * NUM_ACTION + A_t], STATE_NUM+1);
    //printf("Next State: %d\n", S_tn);
    //Get reward
    discounted_reward += pow(dis_gamma, cnt) * GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]];
    cnt ++;
    //printf("discounted: %f\t Reward:%.2f\n", pow(dis_gamma, cnt),
    //GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]]);

    //update the discounted reward
    S_t = S_tn;
  }

  return discounted_reward;
  //printf("%.8f\n", discounted_reward);
}

void run_simulation_with_strategy(double (*f)()){
  printf("Episode: \n");
  REP (i, 0, NUM_EPISODES-1){
    printf("\r %d", i);
    episode_reward[i] = (*f)();
  }
  printf("\n");
}

void get_array_statistics(const double array[],const int size){
  //get mean, standard deviation, maximum, and minimum of the observed discounted returns.

  double amean, adevia, amax, amin;
  amax = *std::max_element(array,array+size);
  amin = *std::min_element(array,array+size);
  get_standard_deviation(array, size, &amean, &adevia);
  printf("Max: %.5f, Min: %.5f, Mean: %.5f, Devia: %.5f\n",
         amax, amin,
         amean, adevia);
}

void get_standard_deviation(const double array[],
                            const int size,
                            double *amean,
                            double *adevia){
  double sum = 0.0, mean, deviation = 0.0;

  REP(i, 0, size-1){//compute the sum of the array
    sum += array[i];
  }

  //get the mean
  mean = sum / size;
  *amean = mean;
  REP(i, 0, size-1){
    deviation += pow(array[i]-mean, 2);
  }

  *adevia = sqrt(deviation / size);
}

/*
 * Run TD update of grid world.
 */
double run_TD_gridworld(Eigen::VectorXd& vf,
                        const double step_size){
  // initialize s_0
  int S_t, A_t, S_tn;
  int cnt = 0;
  double r = 0.0;
  double td_error;
  double MSE = 0.0;
  S_t = random_sample_weights(d_0, 23);
  while (S_t != absorbing_state-1){
    A_t = random_sample_weights(pr_actions, 4);
    S_tn = random_sample_weights(trans_table[S_t * NUM_ACTION + A_t], STATE_NUM + 1);
    //std::cout << "S_tn: " << S_tn << std::endl;
    if (S_tn == absorbing_state-1){
      r = 0;
      td_error = r - vf[S_t];
    }else{
      r = GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]];
      td_error = r + dis_gamma * vf[S_tn] - vf[S_t];
      //std::cout << "TD_error: "<< r << "+" << dis_gamma <<"*" << vf[S_tn] << "-" << vf[S_t] << std::endl;
    }
    vf[S_t] += step_size * td_error;
    //std::cout << "VF: " << vf[S_t] << "\t "<< "TD: " << td_error << std::endl;
    MSE += td_error * td_error;
    S_t = S_tn;
  }
  // using Random policy
  // Compute TD error
  // perform update
  return MSE;
}

double gw_start_TD(){
  generateInput();
  double step_sizes[10];
  double MSES[10];
  REP(i, 0, 9){
    step_sizes[i] = 0.1 / pow(10, i);
    MSES[i] = tabular_TD(100, 23, step_sizes[i], run_TD_gridworld);
  }
  REP(i, 0, 9){
    std::cout << step_sizes[i] << "\t" << MSES[i] << "\t" << std::endl;
  }
}


/*=============================== SARSA ====================================*/
double run_sarsa_gridworld(const int episode,
                           Eigen::VectorXd& qf,// size 23 * 4 (23 position, 4 actions each)
                           const double step_size,
                           const int explore_mode){
  int S_t, A_t, S_tn, A_tn;
  double r = 0.0;
  double target = 0.0;
  int step_cnt = 0;
  double td_error = 0.0;
  double epsilon = 0.6 / (episode + 1);
  double thu = 5;
  double qs, qsp;

  // S_t = random_sample_weights(d_0, 23);
  S_t = 0;
  // use epsilon greedy or softmax to select an action
  if (explore_mode == 0){
    // use epsilon greedy case
    A_t = tabular_epsilon_greedy(epsilon, qf, S_t);
  }else{
    // use Softmax action selection
    A_t = tabular_softmax(thu, qf, S_t);
  }
  while (S_t != absorbing_state - 1){
    // take action
    S_tn = random_sample_weights(trans_table[S_t *  NUM_ACTION + A_t], STATE_NUM +1);
    if (S_tn == absorbing_state-1){
      r = 0;
      qsp = 0;
      qs = qf(S_t * NUM_ACTION + A_t);
      td_error = r + dis_gamma * qsp - qs;
    }else{
      r = GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]];
      target += pow(dis_gamma, step_cnt) * r;
      // choose action for S_tn
      if (explore_mode == 0){
        // use epsilon greedy case
        A_tn = tabular_epsilon_greedy(epsilon, qf, S_tn);
      }else{
        // use Softmax action selection
        A_tn = tabular_softmax(thu, qf, S_tn);
      }
      td_error = r + dis_gamma * qf(S_tn * NUM_ACTION + A_tn) - qf(S_t * NUM_ACTION + A_t);
    }
    qf(S_t * NUM_ACTION + A_t) += step_size * td_error;
    S_t = S_tn;
    A_t = A_tn;
    step_cnt ++;
  }
  return target;
}

int tabular_epsilon_greedy(const double epsilon,
                           const Eigen::VectorXd& qf,
                           const int S_t){
  double mode_weights[2] = {epsilon, 1-epsilon};
  int mode_index = random_sample_weights(mode_weights, 2);
  int action = 0;
  if (mode_index == 0){
    // uniformly random
    double random_weights[NUM_ACTION] = {1,1,1,1};
    action = random_sample_weights(random_weights, 4);
  }else{
    // pick the best among qf.segment(S_t * NUM_ACTION, NUM_ACTION)
    int start_index = S_t * NUM_ACTION;
    int max_index = 0;
    double maxq = 0.0;
    REP(i, 0, NUM_ACTION - 1){
      if (qf(start_index + i) > maxq){
        maxq = qf(start_index + i);
        max_index = i;
      }
    }
    action = max_index;
  }
  return action;
}

int tabular_softmax(const double thu,
                    const Eigen::VectorXd& qf,
                    const int S_t){
  // perform softmax action selection.
  Eigen::VectorXd q_st = qf.segment(S_t * NUM_ACTION, NUM_ACTION);
  q_st *= thu; // q(s,a) * thu
  q_st = q_st.array().exp(); // exp(q(s,a)*thu)
  double sum  = q_st.sum();  // denominator
  q_st /= sum;
  int action = random_sample_eigen_vectors(q_st);
  return action;
}

/*=============================== q LEARNING ===============================*/
double run_qlearning_gridworld(const int episode,
                               Eigen::VectorXd& qf,
                               const double step_size,
                               const int explore_mode){
  // Q-Learning Tabular
  int S_t, A_t, S_tn;
  double r = 0.0;
  double target = 0.0;
  int step_cnt = 0;
  double td_error = 0.0;
  double epsilon = 0.8 / (episode + 1);
  double thu = 5;
  double qs, qsp;

  // S_t = random_sample_weights(d_0, 23);
  S_t = 0;
  while (S_t != absorbing_state -1 ){
    if (explore_mode == 0){
      A_t = tabular_epsilon_greedy(epsilon, qf, S_t);
    }else{
      A_t = tabular_softmax(thu, qf, S_t);
    }
    S_tn = random_sample_weights(trans_table[S_t * NUM_ACTION + A_t], STATE_NUM + 1);
    if (S_tn == absorbing_state -1 ){
      r = 0;
      qs = qf(S_t * NUM_ACTION + A_t);
      qsp = 0;
      td_error = r + dis_gamma * qsp - qs;
    }else{
      r = GW[state_to_coor[S_tn][X]][state_to_coor[S_tn][Y]];
      target += pow(dis_gamma, step_cnt) * r;
      int A_tn = get_best_action(qf, S_tn);
      td_error = r + dis_gamma * qf(S_tn * NUM_ACTION + A_tn) - qf(S_t * NUM_ACTION + A_t);
    }
    qf(S_t * NUM_ACTION + A_t) += step_size * td_error;
    S_t = S_tn;
    step_cnt ++;
  }
  return target;
}

int get_best_action(const Eigen::VectorXd& qf,
                    const int S_t){
  int start_index = S_t * NUM_ACTION;
  int max_index = 0;
  double maxq = 0.0;
  REP(i, 0, NUM_ACTION-1){
    if (qf(start_index + i) > maxq){
      maxq = qf(start_index+i);
      max_index = i;
    }
  }
  return max_index;
}

/*============================ FUNCTIONS CALLED by MAIN =============================*/
void start_sarsa_gw(){
  int num_episodes = 100;
  double step_size = 0.09;
  int explore_mode = 0;
  int q_table_size = STATE_NUM * NUM_ACTION;

  Eigen::MatrixXd result(100, 100);
  generateInput();

  REP(i, 0, 99){
    result.row(i) = on_policy_tabular(num_episodes,
                                      step_size,
                                      explore_mode,
                                      q_table_size,
                                      run_sarsa_gridworld);
  }
  std::ofstream mfile;
  mfile.open("gw_sarsa_ep.dat");
  mfile << result << '\n';
  mfile.close();
}

void start_qlearning_gw(){
  int num_episodes = 100;
  double step_size = 0.1;
  int explore_mode = 0;
  int q_table_size = STATE_NUM * NUM_ACTION;

  Eigen::MatrixXd result(100, 100);
  generateInput();
  REP(i, 0, 99){
    result.row(i) = on_policy_tabular(num_episodes,
                                      step_size,
                                      explore_mode,
                                      q_table_size,
                                      run_qlearning_gridworld);
  }
  std::ofstream mfile;
  mfile.open("gw_ql_ep.dat");
  mfile << result << '\n';
  mfile.close();
}
