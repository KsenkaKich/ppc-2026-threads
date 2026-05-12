#include "kichanova_k_lin_system_by_conjug_grad/all/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace kichanova_k_lin_system_by_conjug_grad {

namespace {

double DotProductOMP(const std::vector<double> &a, const std::vector<double> &b, int n) {
  if (n <= 0) {
    return 0.0;
  }

  double result = 0.0;
#pragma omp parallel for reduction(+ : result) schedule(static) default(none) shared(a, b, n)
  for (int i = 0; i < n; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

void MatrixVectorProductOMP(const std::vector<double> &a, const std::vector<double> &v, std::vector<double> &result,
                            int n) {
  if (n <= 0) {
    return;
  }

  const auto stride = static_cast<size_t>(n);

  if (n < 500) {
    for (int i = 0; i < n; ++i) {
      const double *row = &a[i * stride];
      result[i] = std::inner_product(row, row + n, v.begin(), 0.0);
    }
    return;
  }

#pragma omp parallel for schedule(dynamic, 64) default(none) shared(a, v, result, n, stride)
  for (int i = 0; i < n; ++i) {
    const double *row = &a[i * stride];
    double sum = 0.0;

    int j = 0;
    for (; j <= n - 8; j += 8) {
      sum += row[j] * v[j];
      sum += row[j + 1] * v[j + 1];
      sum += row[j + 2] * v[j + 2];
      sum += row[j + 3] * v[j + 3];
      sum += row[j + 4] * v[j + 4];
      sum += row[j + 5] * v[j + 5];
      sum += row[j + 6] * v[j + 6];
      sum += row[j + 7] * v[j + 7];
    }
    for (; j < n; ++j) {
      sum += row[j] * v[j];
    }
    result[i] = sum;
  }
}

void UpdateSolutionAndResidualOMP(std::vector<double> &x, std::vector<double> &r, const std::vector<double> &p,
                                  const std::vector<double> &ap, double alpha, int n) {
  if (n <= 0) {
    return;
  }

#pragma omp parallel for schedule(static) default(none) shared(x, r, p, ap, alpha, n)
  for (int i = 0; i < n; ++i) {
    x[i] += alpha * p[i];
    r[i] -= alpha * ap[i];
  }
}

void UpdateDirectionOMP(std::vector<double> &p, const std::vector<double> &r, double beta, int n) {
  if (n <= 0) {
    return;
  }

#pragma omp parallel for schedule(static) default(none) shared(p, r, beta, n)
  for (int i = 0; i < n; ++i) {
    p[i] = r[i] + (beta * p[i]);
  }
}

void InitializeVectorsOMP(std::vector<double> &r, std::vector<double> &p, const std::vector<double> &b, int n) {
  std::copy(b.begin(), b.begin() + n, r.begin());

#pragma omp parallel for schedule(static) default(none) shared(p, r, n)
  for (int i = 0; i < n; ++i) {
    p[i] = r[i];
  }
}

}  // namespace

KichanovaKLinSystemByConjugGradALL::KichanovaKLinSystemByConjugGradALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool KichanovaKLinSystemByConjugGradALL::ValidationImpl() {
  const InType &input_data = GetInput();
  if (input_data.n <= 0) {
    return false;
  }
  if (input_data.A.size() != static_cast<size_t>(input_data.n) * static_cast<size_t>(input_data.n)) {
    return false;
  }
  if (input_data.b.size() != static_cast<size_t>(input_data.n)) {
    return false;
  }
  return true;
}

bool KichanovaKLinSystemByConjugGradALL::PreProcessingImpl() {
  GetOutput().assign(GetInput().n, 0.0);
  return true;
}

bool KichanovaKLinSystemByConjugGradALL::RunImpl() {
  const InType &input_data = GetInput();
  OutType &x = GetOutput();

  const int n = input_data.n;
  if (n == 0) {
    return false;
  }

  const std::vector<double> &a = input_data.A;
  const std::vector<double> &b = input_data.b;
  const double epsilon = input_data.epsilon;

  std::vector<double> r(n);
  std::vector<double> p(n);
  std::vector<double> ap(n);

  InitializeVectorsOMP(r, p, b, n);

  double rr_old = DotProductOMP(r, r, n);
  double residual_norm = std::sqrt(rr_old);

  if (residual_norm < epsilon) {
    return true;
  }

  const int max_iter = n * 1000;

  for (int iter = 0; iter < max_iter; ++iter) {
    MatrixVectorProductOMP(a, p, ap, n);

    const double p_ap = DotProductOMP(p, ap, n);
    if (std::abs(p_ap) < 1e-30) {
      break;
    }

    const double alpha = rr_old / p_ap;
    UpdateSolutionAndResidualOMP(x, r, p, ap, alpha, n);

    const double rr_new = DotProductOMP(r, r, n);
    residual_norm = std::sqrt(rr_new);

    if (residual_norm < epsilon) {
      break;
    }

    const double beta = rr_new / rr_old;
    UpdateDirectionOMP(p, r, beta, n);

    rr_old = rr_new;
  }

  return true;
}

bool KichanovaKLinSystemByConjugGradALL::PostProcessingImpl() {
  return true;
}

}  // namespace kichanova_k_lin_system_by_conjug_grad
