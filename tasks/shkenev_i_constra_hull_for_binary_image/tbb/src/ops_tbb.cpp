#include "shkenev_i_constra_hull_for_binary_image/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <queue>
#include <ranges>
#include <vector>

namespace shkenev_i_constra_hull_for_binary_image {

namespace {

constexpr uint8_t kThreshold = 128;

constexpr std::array<std::pair<int, int>, 4> kDirs = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

int64_t Cross(const Point &a, const Point &b, const Point &c) {
  return (int64_t(b.x - a.x) * (c.y - b.y)) - (int64_t(b.y - a.y) * (c.x - b.x));
}

inline bool InBounds(int x, int y, int w, int h) {
  return x >= 0 && x < w && y >= 0 && y < h;
}

}  // namespace

ShkenevIConstrHullTBB::ShkenevIConstrHullTBB(const InType &in) : work_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShkenevIConstrHullTBB::ValidationImpl() {
  const auto &in = GetInput();
  return in.width > 0 && in.height > 0 && in.pixels.size() == size_t(in.width) * size_t(in.height);
}

bool ShkenevIConstrHullTBB::PreProcessingImpl() {
  work_ = GetInput();
  work_.components.clear();
  work_.convex_hulls.clear();

  ThresholdImage();
  return true;
}

void ShkenevIConstrHullTBB::ThresholdImage() {
  auto &p = work_.pixels;

  tbb::parallel_for(tbb::blocked_range<size_t>(0, p.size()), [&](const auto &r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      p[i] = (p[i] > kThreshold) ? 255 : 0;
    }
  });
}

void ShkenevIConstrHullTBB::FindComponents() {
  const int w = work_.width;
  const int h = work_.height;

  std::vector<uint8_t> visited(size_t(w) * h, 0);

  work_.components.clear();

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      size_t idx = Index(x, y, w);

      if (visited[idx] || work_.pixels[idx] == 0) {
        continue;
      }

      std::vector<Point> comp;
      std::queue<Point> q;

      q.emplace(x, y);
      visited[idx] = 1;

      while (!q.empty()) {
        auto cur = q.front();
        q.pop();

        comp.push_back(cur);

        for (auto [dx, dy] : kDirs) {
          int nx = cur.x + dx;
          int ny = cur.y + dy;

          if (!InBounds(nx, ny, w, h)) {
            continue;
          }

          size_t nidx = Index(nx, ny, w);

          if (visited[nidx] || work_.pixels[nidx] == 0) {
            continue;
          }

          visited[nidx] = 1;
          q.emplace(nx, ny);
        }
      }

      work_.components.push_back(std::move(comp));
    }
  }
}

bool ShkenevIConstrHullTBB::RunImpl() {
  FindComponents();

  auto &comps = work_.components;
  auto &hulls = work_.convex_hulls;

  hulls.resize(comps.size());

  tbb::parallel_for(tbb::blocked_range<size_t>(0, comps.size()), [&](const auto &r) {
    for (size_t i = r.begin(); i < r.end(); ++i) {
      const auto &c = comps[i];

      if (c.size() <= 2) {
        hulls[i] = c;
      } else {
        hulls[i] = BuildHull(c);
      }
    }
  });

  GetOutput() = work_;
  return true;
}

std::vector<Point> ShkenevIConstrHullTBB::BuildHull(const std::vector<Point> &pts_in) {
  if (pts_in.size() <= 2) {
    return pts_in;
  }

  std::vector<Point> pts = pts_in;

  std::sort(pts.begin(), pts.end(), [](auto &a, auto &b) { return (a.x != b.x) ? (a.x < b.x) : (a.y < b.y); });

  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

  if (pts.size() <= 2) {
    return pts;
  }

  std::vector<Point> lower, upper;
  lower.reserve(pts.size());
  upper.reserve(pts.size());

  for (auto &p : pts) {
    while (lower.size() >= 2 && Cross(lower[lower.size() - 2], lower.back(), p) <= 0) {
      lower.pop_back();
    }
    lower.push_back(p);
  }

  for (auto &p : std::ranges::reverse_view(pts)) {
    while (upper.size() >= 2 && Cross(upper[upper.size() - 2], upper.back(), p) <= 0) {
      upper.pop_back();
    }
    upper.push_back(p);
  }

  lower.pop_back();
  upper.pop_back();

  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

size_t ShkenevIConstrHullTBB::Index(int x, int y, int w) {
  return size_t(y) * size_t(w) + size_t(x);
}

bool ShkenevIConstrHullTBB::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_constra_hull_for_binary_image
