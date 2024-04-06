/*
    This experiment removes ratings between 2.5 and 3.5 from the dataset and trains the model on the remaining ratings.
    The goal is to just predict whether a user will like an item or not. That is, distinguish between 0-2 and 4-5 ratings.
*/

#include <iostream>
#include <fstream>
#include <istream>
#include <random>
#include <set>
#include <map>
#include <vector>
#include <sstream>
#include <algorithm>
#include "CollaborativeFiltering.h"

static std::default_random_engine engine(std::random_device{}());

bool toss_coin(double probability) {
    std::bernoulli_distribution distribution(probability);
    return distribution(engine);
}

void split_set(const std::map<std::pair<int, int>, double>& full_set,
              std::map<std::pair<int, int>, double>& train_set,
              std::map<std::pair<int, int>, double>& test_set,
              double train_split_ratio = 0.8) {

    for (auto &x : full_set) {
        if (toss_coin(train_split_ratio)) {
            train_set.emplace(x.first, x.second);
        } else {
            test_set.emplace(x.first, x.second);
        }
    }
}

std::map<std::pair<int, int>, double> read_ratings(const std::string& filename, double samplingRate = 1) {
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

        if (rating < 2.5) {
            ratings.emplace(std::make_pair(user, item), 0);
        } else if (rating > 3.5) {
            ratings.emplace(std::make_pair(user, item), 1);
        }
    }

    return ratings;
}

int main() {

    std::ifstream file("ratings.csv");
    std::string line;
    std::map<std::pair<int, int>, double> train_set, test_set;
    
    int k = 15; // number of latent dimensions
    int n = 250000; // upper bound for number of items
    int m = 165000; // upper bound number of users
    
    double lambda = 1e-2; // regularization parameter
    double eta = 1e-5; // learning rate
    double decay = 0.9; // decay rate
    int n_iterations = 15; // number of iterations for the gradient descent

    double train_set_size = 0.8; // percentage of the data will be used for training

    std::string filename = "ratings.csv";
    std::map<std::pair<int, int>, double> ratings = read_ratings(filename, 0.5);
    std::cout << "Finish Reading File" << std::endl;

    split_set(ratings, train_set, test_set, train_set_size); // split the data into train and test sets
    std::cout << "Train Set Size: " << train_set.size() << std::endl;
    std::cout << "Test Set Size: " << test_set.size() << std::endl;

    CollaborativeFiltering cf(train_set, k, lambda, n_iterations, eta, decay, m, n, true);
    cf.fit();

    // calculate the mean absolute error
    double accuracy = 0;

    for (auto &x : test_set) {
        int i = x.first.first, j = x.first.second;
        double prediction = cf.predict(i, j);

        if ((prediction < 0.5 && x.second == 0) || (prediction > 0.5 && x.second == 1)) {
            accuracy += 1;
        }
    }

    accuracy /= test_set.size();

    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}

