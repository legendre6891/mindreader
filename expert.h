#pragma once

#include "game.h"
#include <cmath>
#include <vector>

template <typename A, typename Y> struct Expert {
    double log_weight;
    double eta;
    Game g;

    Expert(double eta, Game<A, Y> g, double log_weight = 0.0)
        : log_weight{log_weight}, eta{eta}, g{g} {}

    virtual A predict() const = 0;
    virtual void update(const A &a, const Y &y);

    void update_weights(const A &a, const Y &y) {
        double loss = g.loss(a, y);
        log_weight -= eta * loss;
    }

    double weight() const { return std::exp(log_weight); }
    double score() const { return log_weight; }
};

unsigned int arg_softmax(std::vector<double> scores) { return 0; }

template <typename A, typename Y> struct ExpertWeights {
  public:
    std::vector<Expert> experts;
    ExpertWeights(const std::vector<Expert> &experts) experts{experts} {}

    A predict() {
        std::vector<double> scores;
        for (const auto &expert : experts) {
            scores.push_back(expert.score());
        }

        auto i = arg_softmax(scores);
        A action = experts[i].predict();

        for (auto &e : experts) {
            e.update()
        }
    }

    void update(const A)
};
