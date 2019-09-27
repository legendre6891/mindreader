#include <GLFW/glfw3.h>
#include <stdio.h>
#include <random>
#include <vector>
#include <numeric>
#include "util.h"

void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int InitializeOnce() {
    glfwSetErrorCallback(glfw_error_callback);
    int success = glfwInit();

    if (!success)
        return success;

#if __APPLE__
    // GL 3.2 + GLSL 150
    const char *glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // 3.0+ only
#endif

    return 0;
}

double runif() {
    static std::random_device rd;
    static std::mt19937 twister(rd());
    static std::uniform_real_distribution<> dist(0, 1);
    return dist(twister);
}

unsigned int sample(const std::vector<double>& v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double U = runif();
    double u = U * sum;

    double accum = 0.0;

    for (unsigned int i = 0; i < v.size(); i++) {
        accum += v[i];
        if (u <= accum) return i;
    }
    return v.size() - 1;
}

unsigned int softmax_sample(const std::vector<double>& v, double eta) {
    double M = v[0];
    for (unsigned i = 0; i < v.size(); ++i) {
        M = M >= v[i] ? M : v[i];
    }

    auto w = v;
    for (unsigned i = 0; i < v.size(); ++i) {
        w[i] = std::exp((w[i] - M) * eta);
    }
    return sample(w); 
}