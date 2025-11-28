#include <mpi.h>

#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

const double PI = M_PI;
const int Lx = 1, Ly = 1, Lz = 1;
const double tau = 0.001;
const int N = 127;
const int K = 20 + 1;
const double h = double(Lx) / N;
const double a_square = 1.0f / 4.0f;

int WORLD_SIZE;  // 8
int SIZE_X;      // 4
int SIZE_Y;      // 2

using Grid = std::array<std::array<std::array<double, N + 1>, N + 1>, N + 1>;

int TripleToSingle(int x, int y, int z) {
    const int small_grid_size_x = (N + 1) / SIZE_X;
    const int small_grid_size_y = (N + 1) / SIZE_Y;
    const int small_grid_size_z = (N + 1);
    return x * small_grid_size_y * small_grid_size_z + y * small_grid_size_z + z;
}

static double func(double t, double x, double y, double z) {
    auto a = (PI / 2) * std::sqrt(1 / (Lx * Lx) + 4 / (Ly * Ly) + 9 / (Lz * Lz));
    return std::sin(PI * x / Lx) * std::sin(2 * PI * y / Ly) * std::sin(3 * PI * z / Lz) * std::cos(a * t) * a;
}

static double func_at(int t, int x, int y, int z) {
    return func(t * tau, x * h, y * h, z * h);
}

static void fill_start(std::vector<double>& data_t0, std::vector<double>& data_t1, int process_coords_x, int process_coords_y) {
    for (int i = 0; i < (N + 1) / SIZE_X; i++) {
        for (int j = 0; j < (N + 1) / SIZE_Y; j++) {
            for (int k = 0; k < N + 1; k++) {
                int pos = TripleToSingle(i, j, k);
                data_t0[pos] = func_at(0, i, j, k);
                data_t1[pos] = func_at(1, i, j, k);
            }
        }
    }
}

static double laplace(const std::vector<double>& data, const std::vector<double>& left, const std::vector<double>& right,
                     const std::vector<double>& bottom, const std::vector<double>& top, int i, int j, int k,
                     const int small_grid_size_x, const int small_grid_size_y, const int small_grid_size_z, int rank) {
    double first, second;
    if (i == 0) {
        int pos1 = TripleToSingle(small_grid_size_x - 1, j, k);
        int pos2 = TripleToSingle(i, j, k);
        int pos3 = TripleToSingle(i + 1, j, k);
        first = left[pos1] - 2 * data[pos2] + data[pos3];
    } else if (i == small_grid_size_x - 1) {
        int pos1 = TripleToSingle(i - 1, j, k);
        int pos2 = TripleToSingle(i, j, k);
        int pos3 = TripleToSingle(0, j, k);
        first = data[pos1] - 2 * data[pos2] + right[pos3];
    } else {
        int pos1 = TripleToSingle(i - 1, j, k);
        int pos2 = TripleToSingle(i, j, k);
        int pos3 = TripleToSingle(i + 1, j, k);
        first = data[pos1] - 2 * data[pos2] + data[pos3];
    }

    if (j == 0) {
        int pos1 = TripleToSingle(i, small_grid_size_y - 1, k);
        int pos2 = TripleToSingle(i, j, k);
        int pos3 = TripleToSingle(i, j + 1, k);
        second = bottom[pos1] - 2 * data[pos2] + data[pos3];
    } else if (j == small_grid_size_y - 1) {
        int pos1 = TripleToSingle(i, j - 1, k);
        int pos2 = TripleToSingle(i, j, k);
        int pos3 = TripleToSingle(i, 0, k);
        second = data[pos1] - 2 * data[pos2] + top[pos3];
    } else {
        int pos1 = TripleToSingle(i, j - 1, k);
        int pos2 = TripleToSingle(i, j, k);
        int pos3 = TripleToSingle(i, j + 1, k);
        second = data[pos1] - 2 * data[pos2] + data[pos3];
    }
    int pos1 = TripleToSingle(i, j, k - 1);
    int pos2 = TripleToSingle(i, j, k);
    int pos3 = TripleToSingle(i, j, k + 1);
    double third = data[pos1] - 2 * data[pos2] + data[pos3];
    return (first + second + third) / (h * h);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int old_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &old_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    WORLD_SIZE = world_size;
    SIZE_X = WORLD_SIZE / 2;
    SIZE_Y = 2;

    if (WORLD_SIZE % 2 != 0) {
        std::cerr << "Error: " << WORLD_SIZE % 2 << std::endl;
        MPI_Finalize();
        return 1;
    }

    MPI_Comm cart_comm;
    std::array<int, 2> dims = {SIZE_Y, SIZE_X};
    std::array<int, 2> period = {0, 0};
    bool reorder = true;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), period.data(), reorder, &cart_comm);

    if (cart_comm == MPI_COMM_NULL) {
        std::cout << "OPS" << std::endl;
    }

    int rank;
    MPI_Comm_rank(cart_comm, &rank);

    std::array<int, 2> coords;
    MPI_Cart_coords(cart_comm, rank, 2, coords.data());

    std::cout << "Rank=" << rank << " coord_y=" << coords[0] << " coord_x=" << coords[1] << std::endl;
    int coord_y = coords[0];
    int coord_x = coords[1];

    const int small_grid_size_x = (N + 1) / SIZE_X;
    const int small_grid_size_y = (N + 1) / SIZE_Y;
    const int small_grid_size_z = (N + 1);

    const int small_grid_size = small_grid_size_x * small_grid_size_y * small_grid_size_z;
    auto data_minus_2 = std::vector<double>(small_grid_size);
    auto data_minus_1 = std::vector<double>(small_grid_size);
    auto data = std::vector<double>(small_grid_size);

    fill_start(data_minus_2, data_minus_1, coord_x, coord_y);

    auto start_time = MPI_Wtime();
    MPI_Barrier(cart_comm);
    for (int t = 2; t < K; t++) {
        std::fill(data.begin(), data.end(), 0);
        std::vector<double> left = data_minus_1;
        std::vector<double> right = data_minus_1;
        std::vector<double> top = data_minus_1;
        std::vector<double> bottom = data_minus_1;

        int left_rank, right_rank, bottom_rank, top_rank;
        MPI_Cart_shift(cart_comm, 0, 1, &left_rank, &right_rank);
        MPI_Cart_shift(cart_comm, 1, 1, &bottom_rank, &top_rank);
        int res1 = MPI_Sendrecv(left.data(), small_grid_size, MPI_DOUBLE,
                                left_rank, 1,
                                right.data(), small_grid_size, MPI_DOUBLE,
                                right_rank, 1,
                                cart_comm, MPI_STATUS_IGNORE);

        int res2 = MPI_Sendrecv(bottom.data(), small_grid_size, MPI_DOUBLE,
                                bottom_rank, 2,
                                top.data(), small_grid_size, MPI_DOUBLE,
                                top_rank, 2,
                                cart_comm, MPI_STATUS_IGNORE);
        int from_x = 0;
        int to_x = small_grid_size_x;
        int from_y = 0;
        int to_y = small_grid_size_y;
        if (coord_x == 0) {
            from_x += 1;
        }
        if (coord_x == SIZE_X - 1) {
            to_x -= 1;
        }
        if (coord_y == 0) {
            from_y += 1;
        }
        if (coord_y == SIZE_Y - 1) {
            to_y -= 1;
        }

        for (int i = from_x; i < to_x; i++) {
            for (int j = from_y; j < to_y; j++) {
                for (int k = 1; k < small_grid_size_z - 1; k++) {
                    double l = laplace(data_minus_1, left, right, bottom, top, i, j, k, small_grid_size_x, small_grid_size_y, small_grid_size_z, rank);
                    int pos = TripleToSingle(i, j, k);
                    data[pos] = tau * tau * a_square * l - data_minus_2[pos] + 2 * data_minus_1[pos];
                }
            }
        }
        data_minus_2.swap(data_minus_1);
        data_minus_1.swap(data);
        MPI_Barrier(cart_comm);
    }
    auto end_time = MPI_Wtime();

    auto full_data = new Grid();
    std::vector<double> fake_data((N + 1) * (N + 1) * (N + 1));
    MPI_Gather(data_minus_1.data(), small_grid_size, MPI_DOUBLE, fake_data.data(), small_grid_size, MPI_DOUBLE, 0, cart_comm);

    if (rank == 0) {
        int pos = 0;
        for (int b = 0; b < WORLD_SIZE; b++) {
            for (int i = 0; i < small_grid_size_x; i++) {
                for (int j = 0; j < small_grid_size_y; j++) {
                    for (int k = 0; k < small_grid_size_z; k++) {
                        (*full_data)[i][j][k] = fake_data[pos];
                        pos += 1;
                    }
                }
            }
        }
    }

    if (rank == 0) {
        std::cout << "Execution time: " << end_time - start_time << std::endl;
        std::cout << "Final check " << (*full_data)[4][12][7] << std::endl;
    }
    MPI_Finalize();
    return 0;
}
