#include <iostream>
#include <fstream>
#include <istream>
#include <random>
#include <set>
#include <map>
#include <vector>
#include <sstream>

using namespace std;

static std::default_random_engine shared_engine(std::random_device{}());

bool toss_coin(double p) {
    std::bernoulli_distribution distribution(p);
    return distribution(shared_engine);
}

double generate_uniform_random_number() {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(shared_engine);
}

// Initialize a vector to random values between 0 and 1
void init_matrix(std::vector<std::vector<double>> &matrix) {
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            matrix[i][j] = generate_uniform_random_number();
        }
    }
}

double dot_product(std::vector<double> &v1, std::vector<double> &v2) {
    double result = 0;
    for (int i = 0; i < v1.size(); i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

int main() {

    std::ifstream file("ratings.csv");
    std::string line;
    std::map<std::pair<int, int>, double> ratings;
    std::map<std::pair<int, int>, double> test_set;
    std::map<int, std::set<int>> users_items; 
    std::map<int, std::set<int>> items_users;
    std::set<int> users;
    std::set<int> items;
    
    int K = 15; // number of latent dimensions
    int m = 7000; // upper bound for number of users
    int n = 300000; // upper bound number of items
    
    double test_set_size = 0.2; // percentage of the data will be used for testing
    double lambda = 1e-3; // regularization parameter
    double eta = 1e-4; // learning rate
    double decay = 0.9; // decay rate
    int n_iterations = 15; // number of iterations for the gradient descent

    if (file.is_open()) {
        std::getline(file, line); // skip the first line

        while (std::getline(file, line)) {

            if (toss_coin(0.6)) {
                // toss a coin to skip some data points
                continue;
            }

            std::istringstream iss(line);
            std::string token;
            // read user, song, and rating
            std::getline(iss, token, ',');
            int user = std::stol(token);
            std::getline(iss, token, ',');
            int item = std::stol(token);
            std::getline(iss, token, ',');
            double rating = std::stod(token);

            if (toss_coin(1 - test_set_size)) {
                // if the coin toss is true, add the rating to the training set
                ratings[std::make_pair(user, item)] = rating;
                users_items[user].insert(item); // add song to user's list of songs
                items_users[item].insert(user); // add user to song's list of users
            } else {
                // if the coin toss is false, add the rating to the test set
                test_set[std::make_pair(user, item)] = rating;
            }
             
            // keep track of users and movies that have been added
            // the IDs might be larger than the number of users and movies
            users.insert(user); 
            items.insert(item);
        }

        file.close();
    } else {
        std::cout << "Unable to open file" << std::endl;
    }

    std::cout << "Finish Reading File" << std::endl;

    // initialize U and V for the collaborative filtering
    std::vector<std::vector<double>> U(m, std::vector<double>(K, 0));
    std::vector<std::vector<double>> V(n, std::vector<double>(K, 0));

    // initialize U and V with random values
    init_matrix(U);
    init_matrix(V);

    for (int t = 0; t < n_iterations; t++) {
        eta = eta * decay;
        vector<vector<double>> U_new (m, vector<double>(K, 0));
        vector<vector<double>> V_new (n, vector<double>(K, 0));

        for (int i : users) {
            for (int k = 0; k < K; k++) {
                double sum = 0;
                for (int j : users_items[i]) {
                    double eij = dot_product(U[i], V[j]) - ratings[std::make_pair(i, j)];
                    sum += eij * V[j][k];
                }
                U_new[i][k] = U[i][k] - eta * (sum + lambda * U[i][k]);
            }
        }

        for (int j : items) {
            for (int k = 0; k < K; k++) {
                double sum = 0;
                for (int i : items_users[j]) {
                    double eij = dot_product(U[i], V[j]) - ratings[std::make_pair(i, j)];
                    sum += eij * U[i][k];
                }
                V_new[j][k] = V[j][k] - eta * (sum + lambda * V[j][k]);
            }
        }

        U = U_new;
        V = V_new;

        cout << "Finished iteration " << t << endl;
    }

    std::cout << "Finish Gradient Descent" << std::endl;
    
    // calculate the mean absolute error
    double mae = 0;
    double mae_3 = 0; // mean absolute error if we were to guess 3 for every rating

    for (auto &x : test_set) {
        int i = x.first.first, j = x.first.second;
        double r = x.second, prediction = dot_product(U[i], V[j]);

        if (prediction < 0.5) {
            prediction = 1;
        } else if (prediction > 5) {
            prediction = 5;
        }

        mae += abs(dot_product(U[i], V[j]) - r);
        mae_3 += abs(3 - r);
    }

    mae = static_cast<double>(mae / test_set.size());
    mae_3 = static_cast<double>(mae_3 / test_set.size());
    std::cout << "Mean Absolute Error: " << mae << std::endl;
    std::cout << "Mean Absolute Error Random Guess: " << mae_3 << std::endl;

    return 0;
}

