#pragma once

#include <vector>

int InitializeOnce();
void glfw_error_callback(int error, const char *description);

double runif();
unsigned int sample(const std::vector<double>& v);
unsigned int softmax_sample(const std::vector<double>& v, double eta);