#include <iostream>
#include <fstream>
#include <istream>
#include <random>
#include <set>
#include <map>
#include <vector>
#include <sstream>
#include <utility> 
#include <algorithm>
#include <chrono> 
#include "CollaborativeFiltering.h"

static std::default_random_engine engine(std::random_device{}());

bool toss_coin(double probability) {
    std::bernoulli_distribution distribution(probability);
    return distribution(engine);
}

void split_set(const std::map<std::pair<int, int>, double>& fullMap,
              std::map<std::pair<int, int>, double>& trainMap,
              std::map<std::pair<int, int>, double>& testMap,
              double trainSplitRatio = 0.8) {

    std::vector<std::pair<int, int>> keys;
    for (const auto& element : fullMap) {
        keys.push_back(element.first);
    }

    std::shuffle(keys.begin(), keys.end(), engine);
    size_t trainSize = static_cast<size_t>(keys.size() * trainSplitRatio);

    for (size_t i = 0; i < keys.size(); ++i) {
        if (i < trainSize) {
            trainMap[keys[i]] = fullMap.at(keys[i]);
        } else {
            testMap[keys[i]] = fullMap.at(keys[i]);
        }
    }
}

std::map<std::pair<int, int>, double> read_ratings(const std::string& filename, double samplingRate = 1.0) {
    std::ifstream file(filename);
    std::map<std::pair<int, int>, double> ratings;
    std::string line;

    if (!file) {
        std::cerr << "Unable to open file" << std::endl;
        return ratings;
    }

    // Skip the first line (headers)
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (!toss_coin(samplingRate)) continue;

        std::istringstream iss(line);
        std::string token;
        int user, item;
        double rating;

        std::getline(iss, token, ',');
        user = std::stoi(token);

        std::getline(iss, token, ',');
        item = std::stoi(token);

        std::getline(iss, token, ',');
        rating = std::stod(token);

        ratings.emplace(std::make_pair(user, item), rating);
    }

    return ratings;
}

int main() {

    std::ifstream file("ratings.csv");
    std::string line;
    std::map<std::pair<int, int>, double> train_set, test_set;
    
    int k = 15; // number of latent dimensions
    int n = 300000; // upper bound for number of items
    int m = 10000; // upper bound number of users
    
    double lambda = 1e-3; // regularization parameter
    double eta = 1e-4; // learning rate
    double decay = 0.9; // decay rate
    int n_iterations = 15; // number of iterations for the gradient descent

    double train_set_size = 0.8; // percentage of the data will be used for training

    std::string filename = "ratings.csv";
    std::map<std::pair<int, int>, double> ratings = read_ratings(filename, 1);
    std::cout << "Finish Reading File" << std::endl;


    split_set(ratings, train_set, test_set, train_set_size); // split the data into train and test sets
    std::cout << "Train Set Size: " << train_set.size() << std::endl;
    std::cout << "Test Set Size: " << test_set.size() << std::endl;

    CollaborativeFiltering cf(train_set, k, lambda, n_iterations, eta, decay, m, n, true);
    cf.fit();

    // calculate the mean absolute error
    double mae = 0;
    double mae_3 = 0; // mean absolute error if we were to guess 3 for every rating

    for (auto &x : test_set) {
        int i = x.first.first, j = x.first.second;
        double prediction = cf.predict(i, j);

        if (prediction < 0.5) 
            prediction = 1;
        else if (prediction > 5) 
            prediction = 5;

        mae += abs(x.second - prediction);
        mae_3 += abs(3 - x.second);
    }

    mae /= test_set.size();
    mae_3 /= test_set.size();

    std::cout << "Mean Absolute Error: " << mae << std::endl;
    std::cout << "Mean Absolute Error of Always Guessing 3: " << mae_3 << std::endl;

    return 0;
}

