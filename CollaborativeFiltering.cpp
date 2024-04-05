// Implementation of CollaborativeFiltering class
#include <iostream>
#include "CollaborativeFiltering.h"
#include <random>
#include <set>
#include <map>

CollaborativeFiltering::CollaborativeFiltering(std::map<std::pair<int, int>, double> &ratings,
                                                int latent_dim,
                                                double lambda,
                                                int num_iterations,
                                                double eta,
                                                double decay,
                                                int num_users,
                                                int num_items,
                                                bool verbose) {
    this->ratings = ratings;
    this->latent_dim = latent_dim;
    this->lambda = lambda;
    this->num_iterations = num_iterations;
    this->eta = eta;
    this->m = num_users;
    this->n = num_items;
    this->decay = decay;
    this->verbose = verbose;

    U = std::vector<std::vector<double>>(m, std::vector<double>(latent_dim, 0));
    V = std::vector<std::vector<double>>(n, std::vector<double>(latent_dim, 0));
}

double CollaborativeFiltering::generate_uniform_random_number() {
    static std::default_random_engine engine(std::random_device{}());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(engine);
}

double CollaborativeFiltering::dot_product(std::vector<double> &v1, std::vector<double> &v2) {
    double result = 0;
    for (int i = 0; i < v1.size(); i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

void CollaborativeFiltering::fit() {
    // Fit the model using the training data.
    std::map<int, std::set<int>> users_items;
    std::map<int, std::set<int>> items_users;

    std::cout << "Fitting the model on " << ratings.size() << " ratings" << std::endl;

    for (auto &rating : ratings) {
        int user = rating.first.first;
        int item = rating.first.second;
        
        users_items[user].insert(item);
        items_users[item].insert(user);
    }

    // Initialize the latent factors randomly
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < latent_dim; j++) {
            U[i][j] = generate_uniform_random_number();
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < latent_dim; j++) {
            V[i][j] = generate_uniform_random_number();
        }
    }

    // Gradient Descent
    for (int iteration = 0; iteration < num_iterations; iteration++) {
        eta = eta * decay;

        for (int i = 1; i < m; i++) {
            for (int j : users_items[i]) {    
                double eij = dot_product(U[i], V[j]) - ratings[std::make_pair(i, j)];
                for (int k = 0; k < latent_dim; k++) {
                    U[i][k] = U[i][k] - eta * (2 * eij * V[j][k] + lambda * U[i][k]);
                }
            }
        }

        for (int j = 0; j < n; j++) {
            for (int i : items_users[j]) {
                double eij = dot_product(U[i], V[j]) - ratings[std::make_pair(i, j)];
                for (int k = 0; k < latent_dim; k++) {
                    V[j][k] = V[j][k] - eta * (2 * eij * U[i][k] + lambda * V[j][k]);
                }
            }
        }
        
        if (verbose) {
            std::cout << "Iteration: " << iteration << std::endl;
        }
    }
}

double CollaborativeFiltering::predict(int &user, int &item) {
    // Predict the rating of a user for an item.
    return dot_product(U[user], V[item]);
}

std::vector<std::vector<double>> CollaborativeFiltering::get_U() const {
    return U;
}

std::vector<std::vector<double>> CollaborativeFiltering::get_V() const {
    return V;
}