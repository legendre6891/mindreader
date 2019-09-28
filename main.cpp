// #include "lib/abseil-cpp/absl/strings/str_cat.h"
// #include "lib/abseil-cpp/absl/strings/str_format.h"

#include "lib/fmt/include/fmt/format.h"
#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <iostream>
#include <stdio.h>
#include <vector>

#include <GL/glew.h> // Initialize with glewInit()

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include "util.h"

#define PI 3.14159265358979323846

#if __APPLE__
const char *glsl_version = "#version 150";
#else
const char *glsl_version = "#version 130";
#endif

static void HelpMarker(const char *desc) {
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

#include "pennies.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>

int main() {
    std::vector<double> grid;
    for (double x = 0.0; x <= 1; x += 0.05) {
        grid.push_back(x);
    }
    std::vector<double> grid2 = {0.1, 0.25, 0.4, 0.6, 0.75, 0.9};
    std::vector<double> grid3 = {0.1, 0.3, 0.5, 0.7, 0.9};

    std::vector<Expert<int, int>> experts;
    std::vector<std::string> labels;

    for (auto g : grid) {
        experts.push_back(ProportionExpert(g));
        labels.push_back(fmt::format("Proportion[{:.2f}]", g));
    }

    for (auto g : {0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5}) {
        experts.push_back(ExponentialExpert(g));
        labels.push_back(fmt::format("Exponential[{:.2f}]", g));
    }

    for (auto g : grid2) {
        experts.push_back(StreakExpert(g));
        labels.push_back(fmt::format("Streak[{:.2f}]", g));
    }

    for (auto g : grid2) {
        experts.push_back(CorrelatedExpert(g));
        labels.push_back(fmt::format("Correlated[{:.2f}]", g));
    }

    std::vector<double> omegas = {0.0, 0.5, 1.0, 1.5, 2.,  2.5, 3.,
                                  3.5, 4.,  4.5, 5.,  5.5, 6.};
    std::vector<double> phis = {-PI, -0.5 * PI, 0., 0.5 * PI, PI};

    for (auto w : omegas) {
        for (auto phi : phis) {
            experts.push_back(CosineExpert(2 * PI / w, phi));
            labels.push_back(fmt::format("Cosine[{:.2f} {:.2f}]", w, phi));
        }
    }

    for (auto a : grid3) {
        for (auto b : grid3) {
            for (auto c : grid3) {
                for (auto d : grid3) {
                    if (runif() <= 2.2) {
                        experts.push_back(LengthTwoExpert(a, b, c, d));
                        labels.push_back(fmt::format(
                            "LengthTwo[{:.2f} {:.2f} {:.2f} {:.2f}]", a, b,
                            c, d));
                    }
                }
            }
        }
    }

    int n_rounds = 101;
    int n_experts = experts.size();
    auto E = ExpertAdvice<int, int>(zero_one_loss, n_rounds, experts, labels);

    InitializeOnce();

    // Create window with graphics context
    GLFWwindow *window = glfwCreateWindow(700, 450, "Mindreader", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    bool err = glewInit() != GLEW_OK;

    if (err) {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    int screen_width, screen_height;
    glfwGetFramebufferSize(window, &screen_width, &screen_height);
    glViewport(0, 0, screen_width, screen_height);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);
    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    int human_score = 0;
    int cpu_score = 0;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        int y = 0;

        // feed inputs to dear imgui, start new frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Mindreader", NULL);
        ImGui::Text("You");
        ImGui::SameLine(100);
        ImGui::Text("Mindreader");
        ImGui::Text("%d", human_score);
        ImGui::SameLine(100);
        ImGui::Text("%d", cpu_score);

        ImGui::Separator();
        ImGui::Spacing();
        if (ImGui::Button("New Game")) {
            E.reset();
            human_score = 0;
            cpu_score = 0;
        }
        ImGui::Text("Use the Left and Right arrow keys");
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Text("Round %d out of %d", E.round_counter, E.nrounds);

        bool gameover = E.gameover() == false && cpu_score <= E.nrounds / 2 &&
                        human_score <= E.nrounds / 2;

        if (E.round_counter >= 1 && gameover) {
            if (E.outcomes.back() == E.predictions.back()) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.3f, 1.0f),
                                   "LOST last round");
            } else {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.3f, 1.0f),
                                   "WON last round");
            }
        }
        if (gameover) {
            if (ImGui::IsKeyPressed(262)) {
                y = 1;
            } else if (ImGui::IsKeyPressed(263)) {
                y = -1;
            }
            if (y != 0) {
                auto p = E.predict();
                E.update(p, y);
                cpu_score = E.round_counter - (int)E.cumulative_loss;
                human_score = (int)E.cumulative_loss;
            }
        } else {
            ImGui::Spacing();

            if (cpu_score > human_score) {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.3f, 1.0f),
                                   "You lost :(");
            } else if (cpu_score < human_score) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.3f, 1.0f),
                                   "You won :)!");
            } else {
                ImGui::Text("Tied game - not bad!");
            }
        }

        ImGui::End();

        ImGui::Begin("Expert Weights", NULL);

        ImGui::Text("Weight (%%)");
        ImGui::SameLine(100);
        ImGui::Text("Expert");
        ImGui::SameLine(170);
        ImGui::Text("[No. of Experts: %d]", n_experts);

        ImGui::Separator();
        ImGui::Spacing();

        for (auto i = 0; i < n_experts; i++) {
            // for (auto i = 0; i < std::min(n_experts, 20); i++) {
            auto j = E.m_indices[i];
            ImGui::Text("%1.2f", E.m_pct_weights[j]);
            ImGui::SameLine(100);
            ImGui::Text("%s", E.labels[j].c_str());
        }
        ImGui::End();

        ImGui::Begin("Next Prediction", NULL);

        if (E.m_action_pct_weights[-1] >= E.m_action_pct_weights[1]) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.3f, 1.0f),
                               "LEFT: %2.1f%%", E.m_action_pct_weights[-1]);
            ImGui::SameLine(120);
            ImGui::Text("RIGHT: %2.1f%%", E.m_action_pct_weights[1]);
        } else {
            ImGui::Text("LEFT: %2.1f%%", E.m_action_pct_weights[-1]);
            ImGui::SameLine(120);
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.3f, 1.0f),
                               "RIGHT: %2.1f%%", E.m_action_pct_weights[1]);
        }

        ImGui::End();

        // ImGui::ShowDemoWindow();

        // if (show_app_style_editor) {
        // ImGui::Begin("Style Editor", &show_app_style_editor);
        // ImGui::ShowStyleEditor();
        // ImGui::End();
        // }

        // Render dear imgui into screen
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
