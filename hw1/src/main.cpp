#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <omp.h>

class net_solver
{
public:
    net_solver(int N, const std::function<double(double, double)> &fun_f,
               const std::function<double(double, double)> &fun_g) : N(N), NB(std::ceil((double)N / BS)),
                                                                     h((double)1 / (N + 1))
    {
        u = std::vector<std::vector<double>>(N + 2, std::vector<double>(N + 2, 0));
        f = std::vector<std::vector<double>>(N + 2, std::vector<double>(N + 2, 0));
        dm = std::vector<double>(NB, 0);

        for (int i = 1; i < N + 1; ++i)
            for (int j = 1; j < N + 1; ++j)
                f[i][j] = fun_f(i * h, j * h);

        for (int i = 0; i < N + 1; ++i)
        {
            u[i][0] = fun_g(i * h, 0);
            u[0][i + 1] = fun_g(0, (i + 1) * h);
            u[i + 1][N + 1] = fun_g((i + 1) * h, (N + 1) * h);
            u[N + 1][i] = fun_g((N + 1) * h, i * h);
        }
    }

    std::vector<std::vector<double>> solve()
    {
        double dmax = 0;
        do
        {
            dmax = 0;
            for (int nx = 0; nx < NB; ++nx)
            {
                dm[nx] = 0;
#pragma omp parallel for shared(nx)
                for (int i = 0; i < nx + 1; ++i)
                {
                    int j = nx - i;
                    double d = process_block(i, j);
                    dm[i] = std::max(dm[i], d);
                }
            }
            for (int nx = NB - 2; nx >= 0; --nx)
            {
#pragma omp parallel for shared(nx)
                for (int i = 1; i < nx + 1; ++i)
                {
                    int j = 2 * (NB - 1) - nx - i;
                    double d = process_block(i, j);
                    dm[i] = std::max(dm[i], d);
                }
            }
            dmax = std::max(dmax, *std::max_element(dm.begin(), dm.end()));
        } while (dmax > eps);
        return u;
    }

private:
    const int N;
    const int BS = 32;
    const int NB;

    const double h;
    const double eps = 0.1;

    std::vector<std::vector<double>> u;
    std::vector<std::vector<double>> f;
    std::vector<double> dm;

    double process_block(int i_block, int j_block)
    {
        int beg_i = i_block * BS + 1;
        int beg_j = j_block * BS + 1;

        int end_i = std::min(beg_i + BS, N);
        int end_j = std::min(beg_j + BS, N);

        double dm = 0;
        for (int i = beg_i; i < end_i; ++i)
        {
            for (int j = beg_j; j < end_j; ++j)
            {
                double temp = u[i][j];
                u[i][j] = 0.25 * std::abs(u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] -
                                          h * h * f[i][j]);
                double d = std::abs(temp - u[i][j]);
                dm = std::max(dm, d);
            }
        }
        return dm;
    }
};

int main()
{
    auto fun_f = [](double x, double y)
    { return 0; };

    auto fun_g = [](double x, double y)
    {
        if (x == 0)
            return 100 - 200 * y;
        else if (y == 0)
            return 100 - 200 * x;
        else if (x == 1)
            return -100 + 200 * y;
        return -100 + 200 * x;
    };

    std::vector<int> threads = {1, 4, 8};
    std::vector<int> Ns = {100, 200, 300, 500, 1000, 2000, 3000};
    for (const auto th : threads)
    {
        omp_set_num_threads(th);
        for (const auto N : Ns)
        {
            net_solver net(N, fun_f, fun_g);
            auto start_time = omp_get_wtime();
            net.solve();
            auto end_time = omp_get_wtime();
            std::cout << "threads = " << th << " net size = " << N
                      << " time = " << end_time - start_time << std::endl;
        }
    }
}