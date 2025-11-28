#include <omp.h>

#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

const double PI = M_PI;
const int Lx = 1, Ly = 1, Lz = 1;
const double tau = 0.001;
const int N = 256;
const int K = 30 + 1;
const double h = double(Lx) / N;
const double a_square = 1 / 4;
const int THREAD_COUNT = 8;

using Grid = std::array<std::array<std::array<double, N + 1>, N + 1>, N + 1>;
using TimeGrid = std::vector<Grid>;

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

struct Error {
    double abs_error;
    double rel_error;
};

static Error calculate_error(const TimeGrid& data, int t) {
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
                    std::cout << t << " " << i << " " << j << " " << k << "|||" << data[t][i][j][k] << " " << func_at(t, i, j, k) << std::endl;
                    rel_error = tmp_rel_error;
                }
            }
        }
    }
    return {abs_error, rel_error};
}

int main() {
    omp_set_num_threads(THREAD_COUNT);
    TimeGrid data(K);
    fill_start(data);

    auto time1 = omp_get_wtime();
    for (int t = 2; t < K; t++) {
#pragma omp parallel for default(none) shared(data, t) collapse(2) schedule(dynamic, 16)
        for (int i = 1; i < N; i++) {
            for (int j = 1; j < N; j++) {
                for (int k = 1; k < N; k++) {
                    double l = laplace(data, t - 1, i, j, k);
                    data[t][i][j][k] = tau * tau * a_square * l - data[t - 2][i][j][k] + 2 * data[t - 1][i][j][k];
                }
            }
        }

        // for (int i = 1; i < N; i++) {
        //     for (int k = 1; k < N; k++) {
        //         double tmp = (data[t][i][1][k] + data[t][i][N - 1][k]) / 2.0f;
        //         data[t][i][0][k] = tmp;
        //         data[t][i][N][k] = tmp;
        //     }
        // }
    }
    auto time2 = omp_get_wtime();

    std::cout << "Execution time: " << time2 - time1 << std::endl;

    Error error = calculate_error(data, K - 1);
    std::cout << "Abs errpr: " << error.abs_error << "| Rel error: " << error.rel_error << std::endl;
    return 0;
}
