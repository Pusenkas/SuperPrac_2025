#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

const double PI = M_PI;
const int Lx = 1, Ly = 1, Lz = 1;
const double tau = 0.01;
const int N = 16;
const int K = 10 + 1;
const double h = double(Lx) / N;
double a_square = 1.0f / 4.0f;

using Grid = std::array<std::array<std::array<double, N + 1>, N + 1>, N + 1>;
using TimeGrid = std::vector<Grid>;

struct Error {
    double abs_error;
    double rel_error;
};

static double func(double t, double x, double y, double z) {
    auto a = (PI / 2) * std::sqrt(1 / (Lx * Lx) + 4 / (Ly * Ly) + 9 / (Lz * Lz));
    return std::sin(PI * x / Lx) * std::sin(2 * PI * y / Ly) * std::sin(3 * PI * z / Lz) * std::cos(a * t) * a;
}

static double func_at(int t, int x, int y, int z) {
    return func(t * tau, x * h, y * h, z * h);
}

static void fill_start(TimeGrid& data) {
    for (int t = 0; t <= 1; t++) {
        for (int i = 0; i < N + 1; i++) {
            for (int j = 0; j < N + 1; j++) {
                for (int k = 0; k < N + 1; k++) {
                    data[t][i][j][k] = func_at(t, i, j, k);
                }
            }
        }
    }
}

static double laplace(const TimeGrid& data, const int t, int i, int j, int k) {
    double first = data[t][i - 1][j][k] - 2 * data[t][i][j][k] + data[t][i + 1][j][k];
    double second = data[t][i][j - 1][k] - 2 * data[t][i][j][k] + data[t][i][j + 1][k];
    double third = data[t][i][j][k - 1] - 2 * data[t][i][j][k] + data[t][i][j][k + 1];
    return (first + second + third) / (h * h);
}

static std::pair<double, double> test_point(const TimeGrid& data, int t, int i, int j, int k) {
    return {data[t][i][j][k], func_at(t, i, j, k)};
}

static std::pair<double, double> calculate_error(const TimeGrid& data, int t) {
    const double EPSILON = 0.0001;
    double abs_error = 0;
    double rel_error = 0;
    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            for (int k = 0; k < N + 1; k++) {
                double tmp_abs_error = std::abs(data[t][i][j][k] - func_at(t, i, j, k));
                if (tmp_abs_error > abs_error) {
                    abs_error = tmp_abs_error;
                }

                double tmp_rel_error = std::abs(data[t][i][j][k] - func_at(t, i, j, k)) / std::abs(func_at(t, i, j, k));
                if (std::abs(func_at(t, i, j, k)) < EPSILON && std::abs(data[t][i][j][k] - func_at(t, i, j, k)) < EPSILON) {
                    continue;
                }
                if (tmp_rel_error > rel_error) {
                    // std::cout << t << " " << i << " " << j << " " << k << "|||" << data[t][i][j][k] << " " << func_at(t, i, j, k) << std::endl;
                    rel_error = tmp_rel_error;
                }
            }
        }
    }
    return {abs_error, rel_error};
}

void print_data(const TimeGrid& data, int t) {
    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < N + 1; j++) {
            for (int k = 0; k < N + 1; k++) {
                std::cout << data[t][i][j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

void print_data_all(const TimeGrid& data) {
    for (int t = 0; t < K; t++) {
        for (int i = 0; i < N + 1; i++) {
            for (int j = 0; j < N + 1; j++) {
                for (int k = 0; k < N + 1; k++) {
                    std::cout << std::fixed << std::setprecision(4) << data[t][i][j][k] << " ";
                }
            }
        }
    }
    std::cout << std::endl;
}

void print_error_rel(const TimeGrid& data) {
    const double EPSILON = 0.0001;
    for (int t = 0; t < K; t++) {
        for (int i = 0; i < N + 1; i++) {
            for (int j = 0; j < N + 1; j++) {
                for (int k = 0; k < N + 1; k++) {
                    double tmp_rel_error = std::abs(data[t][i][j][k] - func_at(t, i, j, k)) / std::abs(func_at(t, i, j, k));
                    if ((std::abs(func_at(t, i, j, k)) < 0.001 && std::abs(data[t][i][j][k] - func_at(t, i, j, k)) < EPSILON) || tmp_rel_error > 1) {
                        std::cout << std::fixed << std::setprecision(4) << 0.000 << " ";
                    } else {
                        std::cout << std::fixed << std::setprecision(4) << tmp_rel_error << " ";
                    }
                }
            }
        }
    }
    std::cout << std::endl;
}

void print_error_abs(const TimeGrid& data) {
    for (int t = 0; t < K; t++) {
        for (int i = 0; i < N + 1; i++) {
            for (int j = 0; j < N + 1; j++) {
                for (int k = 0; k < N + 1; k++) {
                    std::cout << std::fixed << std::setprecision(4) << std::abs(data[t][i][j][k] - func_at(t, i, j, k)) << " ";
                }
            }
        }
    }
    std::cout << std::endl;
}

void print_data_all_analitical() {
    for (int t = 0; t < K; t++) {
        for (int i = 0; i < N + 1; i++) {
            for (int j = 0; j < N + 1; j++) {
                for (int k = 0; k < N + 1; k++) {
                    std::cout << std::fixed << std::setprecision(4) << func_at(t, i, j, k) << " ";
                }
            }
        }
    }
    std::cout << std::endl;
}

int main() {
    TimeGrid data(K);
    fill_start(data);

    for (int t = 2; t < K; t++) {
        for (int i = 1; i < N; i++) {
            for (int j = 1; j < N; j++) {
                for (int k = 1; k < N; k++) {
                    double l = laplace(data, t - 1, i, j, k);
                    data[t][i][j][k] = tau * tau * a_square * l - data[t - 2][i][j][k] + 2 * data[t - 1][i][j][k];
                }
            }

            for (int k = 1; k < N; k++) {
                double tmp = (data[t][i][1][k] + data[t][i][N - 1][k]) / 2.0f;
                data[t][i][0][k] = tmp;
                data[t][i][N][k] = tmp;
            }
        }
    }

    auto [abs_error, rel_error] = calculate_error(data, 8);
    // std::cout << "Abs errpr: " << abs_error << "| Rel error: " << rel_error << std::endl;

    // print_data_all_analitical();
    // print_data_all(data);
    // print_error_abs(data);
    print_error_rel(data);

    // std::cout << "Final" << data[K - 1][4][12][7] << std::endl;

    return 0;
}
