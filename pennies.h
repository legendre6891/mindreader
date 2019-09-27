#include "util.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

    // initialize original index locations
    std::vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

    return idx;
}

template <typename A, typename Y>
using Expert =
    std::function<A(const std::vector<A> &, const std::vector<Y> &, int)>;

template <typename A, typename Y>
using LossFunction = std::function<double(A, Y)>;

template <typename A, typename Y> struct ExpertAdvice {
    std::vector<A> predictions;
    std::vector<Y> outcomes;
    std::vector<A> advice;

    LossFunction<A, Y> loss_function;
    const int nrounds;
    const double eta;

    std::vector<double> scores;
    std::vector<Expert<A, Y>> experts;
    std::vector<std::string> labels;

    std::vector<double> m_pct_weights;
    std::vector<size_t> m_indices;
    std::map<A, double> m_action_pct_weights;

    int round_counter;
    double cumulative_loss;

    ExpertAdvice(LossFunction<A, Y> loss_function, int nrounds,
                 std::vector<Expert<A, Y>> experts,
                 std::vector<std::string> labels)
        : loss_function{loss_function}, nrounds{nrounds}, experts{experts},
          labels{labels}, round_counter{0}, cumulative_loss{0.0},
          eta{std::sqrt(2.0 * std::log(experts.size()) / nrounds)} {

        auto n_experts = experts.size();
        advice.resize(n_experts);
        labels.resize(n_experts);
        scores.resize(n_experts);

        m_pct_weights.resize(n_experts);
        m_indices.resize(n_experts);

        reset();
    }

    void reset() {
        round_counter = 0;
        cumulative_loss = 0.0;
        predictions.clear();
        outcomes.clear();

        auto n_experts = experts.size();
        for (unsigned int i = 0; i < n_experts; i++) {
            scores[i] = 0.0;
            advice[i] = experts[i](predictions, outcomes, round_counter);
        }

        update_debug();
    }

    bool gameover() const { return !(round_counter < nrounds); }

    void update(A prediction, Y outcome) {
        auto n = experts.size();

        outcomes.push_back(outcome);
        predictions.push_back(prediction);
        cumulative_loss += loss_function(prediction, outcome);
        round_counter++;

        for (auto i = 0u; i < n; i++) {
            scores[i] -= loss_function(advice[i], outcome);
            advice[i] = experts[i](predictions, outcomes, round_counter);
        }

        update_debug();
    }

    A predict() { return advice[softmax_sample(scores, eta)]; }

  private:
    void update_debug() {
        double M = scores[0];
        for (unsigned i = 0; i < scores.size(); ++i) {
            M = M >= scores[i] ? M : scores[i];
        }

        auto w = scores;
        for (unsigned i = 0; i < scores.size(); ++i) {
            w[i] = std::exp((w[i] - M) * eta);
        }

        double sum = std::accumulate(w.begin(), w.end(), 0.0);

        for (unsigned i = 0; i < scores.size(); ++i) {
            w[i] = 100.0 * w[i] / sum;
        }

        m_pct_weights = w;
        m_indices = sort_indexes(w);

        m_action_pct_weights.clear();
        for (unsigned int i = 0; i < advice.size(); i++) {
            m_action_pct_weights[advice[i]] += w[i];
        }
    }
};

double zero_one_loss(int p, int y) {
    if (p == y)
        return 0.0;
    else
        return 1.0;
}

struct ProportionExpert {
    double p;
    ProportionExpert(double p = 0.5) : p{p} {}

    int operator()(const std::vector<int> &predictions,
                   const std::vector<int> &outcomes, int n) {
        if (runif() <= p)
            return -1;
        else
            return 1;
    }
};

struct CorrelatedExpert {
    double p;
    CorrelatedExpert(double p = 0.5) : p{p} {}

    int operator()(const std::vector<int> &predictions,
                   const std::vector<int> &outcomes, int n) {
        auto r = runif();

        if (outcomes.empty()) {
            if (r <= 0.5)
                return -1;
            else
                return 1;
        }

        if (r <= p) {
            return outcomes.back();
        } else {
            return -outcomes.back();
        }
    }
};

struct StreakExpert {
    double p;
    StreakExpert(double p = 0.5) : p{p} {}

    int operator()(const std::vector<int> &predictions,
                   const std::vector<int> &outcomes, int n) {
        auto r = runif();

        if (outcomes.empty()) {
            if (r <= 0.5)
                return -1;
            else
                return 1;
        }

        if (r <= p) {
            return outcomes.back() * predictions.back();
        } else {
            return -outcomes.back() * predictions.back();
        }
    }
};

struct ExponentialExpert {
    double beta;
    ExponentialExpert(double beta) : beta{beta} {}

    int operator()(const std::vector<int> &predictions,
                   const std::vector<int> &outcomes, int n) {
        auto r = runif();

        if (outcomes.empty()) {
            if (r <= 0.5)
                return -1;
            else
                return 1;
        }

        double accum = 0;
        double weight = 0;
        for (auto o : outcomes) {
            accum = accum * beta;
            accum = accum + o;

            weight = weight * beta;
            weight += 1;
        }
        accum = accum / weight;
        accum = (accum + 1) / 2.0;

        if (r <= accum) {
            return -1;
        } else {
            return 1;
        }
    }
};

struct CosineExpert {
    double omega, phi;
    CosineExpert(double omega, double phi) : omega{omega}, phi{phi} {}

    int operator()(const std::vector<int> &predictions,
                   const std::vector<int> &outcomes, int n) {
        auto r = runif();

        if (outcomes.empty()) {
            if (r <= 0.5)
                return -1;
            else
                return 1;
        }

        double accum = 0;
        double total = 0;

        for (unsigned int i = 0; i < outcomes.size(); i++) {
            accum += outcomes[i] * std::cos(omega * i + phi);
            total += 1 * std::cos(omega * i + phi);
        }

        accum = accum / total;
        accum = (accum + 1) / 2.0;

        if (r <= accum) {
            return -1;
        } else {
            return 1;
        }
    }
};

struct LengthTwoExpert {
    double a, b, c, d;
    LengthTwoExpert(double a, double b, double c, double d)
        : a{a}, b{b}, c{c}, d{d} {}

    int operator()(const std::vector<int> &predictions,
                   const std::vector<int> &outcomes, int n) {
        auto r = runif();

        if (outcomes.size() <= 1) {
            if (r <= 0.5)
                return -1;
            else
                return 1;
        }

        int x = outcomes[outcomes.size() - 2];
        int y = outcomes[outcomes.size() - 1];

        if (x == -1 && y == -1)
            return (r <= a) ? -1 : 1;
        if (x == -1 && y == 1)
            return (r <= b) ? -1 : 1;
        if (x == 1 && y == -1)
            return (r <= c) ? -1 : 1;
        if (x == 1 && y == 1)
            return (r <= d) ? -1 : 1;
    }
};
