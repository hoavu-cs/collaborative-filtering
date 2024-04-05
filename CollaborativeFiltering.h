#ifndef CollaborativeFiltering_H
#define CollaborativeFiltering_H

#include <vector>
#include <set>
#include <map>

class CollaborativeFiltering {
private:
    std::map<std::pair<int, int>, double> ratings;
    double eta, decay, lambda;
    bool verbose;
    int m, n, num_iterations,latent_dim;
    std::vector<std::vector<double>> U;
    std::vector<std::vector<double>> V;

    double generate_uniform_random_number(); // generate a random number between 0 and 1
    double dot_product(std::vector<double> &v1, std::vector<double> &v2); // dot product of two vectors

public:
    CollaborativeFiltering(std::map<std::pair<int, int>, double> &ratings, 
                                int latent_dim, 
                                double lambda, 
                                int num_iterations, 
                                double eta,
                                double decay,
                                int num_users,
                                int num_items,
                                bool verbose = false);
    /*
        Constructor for the collaborative filtering class.
        ratings: set of pairs (user, song) with the rating given by the user to the song
        latent_dim: number of latent dimensions
        lambda: regularization parameter
        n_iterations: number of iterations for the gradient descent
        eta: learning rate
        m: number of users
        n: number of items
    */
    
    void fit(); 
    /*
        Fit the model using the training data.
    */
    
    double predict(int &user, int &item);
    /*
        Predict the rating of a user for a song.
        user: user id
        song: song id
        returns: predicted rating
    */
   
    std::vector<std::vector<double>> get_U() const;
    /*
        Get the embedding matrix U of the users.
    */

    std::vector<std::vector<double>> get_V() const;
    /*
        Get the embedding matrix V of the items.
    */
};

#endif