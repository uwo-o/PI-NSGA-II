#include <iostream>
#include "common.hpp"

int Config::POP_SIZE = 300;
int Config::MAX_GEN = 300;
int Config::N_DOMAIN = 2000;
int Config::N_BOUNDARY = 500;
double Config::ERC_SIGMA = 0.20;
int Config::MAX_TREE_DEPTH = 6;
double Config::CROSSOVER_PROB = 0.85;
double Config::MUTATION_PROB = 0.35;
int Config::TOURNAMENT_SIZE = 3;
double Config::STOP_THRESHOLD = 1e-7;
int Config::RAR_INTERVAL = 25;
int Config::RAR_CANDIDATES = 500;
double Config::RAR_ADAPTIVE_RATIO = 0.10;
int Config::RAR_ELITE_COUNT = 10;
double Config::RAR_RANDOM_RATIO = 0.25;
int Config::CORES = 1;

int main() {
    std::cout << "CROSSOVER: " << Config::CROSSOVER_PROB << std::endl;
    std::cout << "MUTATION: " << Config::MUTATION_PROB << std::endl;
    return 0;
}
