#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <set>
#include <thread>

#include "hackablegrid/vec_env.hpp"

using namespace hg;

namespace {
struct Buffers {
  std::vector<float> obs, r, tr, pr, et, ep;
  std::vector<uint8_t> term, trunc, picked, goal;
  std::vector<int32_t> len, pick;
  Buffers(const VecEnv& e) {
    size_t n = e.num_envs();
    obs.assign(n * e.obs_dim(), 0); r.assign(n, 0); tr.assign(n, 0); pr.assign(n, 0);
    et.assign(n, 0); ep.assign(n, 0); term.assign(n, 0); trunc.assign(n, 0);
    picked.assign(n, 0); goal.assign(n, 0); len.assign(n, 0); pick.assign(n, 0);
  }
  StepOutputs out() {
    return {obs.data(), r.data(), tr.data(), pr.data(), term.data(), trunc.data(),
            picked.data(), et.data(), ep.data(), len.data(), pick.data(), goal.data()};
  }
};

Config base(int n, int threads, Variant v = Variant::Misspecified) {
  Config c; c.num_envs = n; c.num_threads = threads; c.variant = v; c.seed = 42;
  return c;
}
}  // namespace

TEST(MpmcQueue, FifoAndCapacity) {
  MpmcQueue<int> q(4);
  int x;
  EXPECT_FALSE(q.try_pop(x));
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(q.try_push(i));
  EXPECT_FALSE(q.try_push(99));
  for (int i = 0; i < 4; ++i) { EXPECT_TRUE(q.try_pop(x)); EXPECT_EQ(x, i); }
}

TEST(MpmcQueue, ConcurrentProducersConsumers) {
  MpmcQueue<size_t> q(64);
  constexpr size_t kPer = 20000, kProd = 3;
  std::atomic<size_t> sum{0}, count{0};
  std::vector<std::thread> ts;
  for (size_t p = 0; p < kProd; ++p)
    ts.emplace_back([&, p] { for (size_t i = 1; i <= kPer; ++i) while (!q.try_push(i + p * 0)) {} });
  for (int c = 0; c < 3; ++c)
    ts.emplace_back([&] {
      size_t v;
      while (count.load() < kPer * kProd) if (q.try_pop(v)) { sum += v; ++count; }
    });
  for (auto& t : ts) t.join();
  EXPECT_EQ(sum.load(), kProd * kPer * (kPer + 1) / 2);
}

TEST(ThreadPool, CoversRangeExactlyOnce) {
  ThreadPool pool(4);
  for (size_t n : {1u, 15u, 16u, 1000u, 4097u}) {
    std::vector<std::atomic<int>> hits(n);
    for (auto& h : hits) h = 0;
    pool.parallel_for(n, [&](size_t b, size_t e) { for (size_t i = b; i < e; ++i) hits[i]++; });
    for (size_t i = 0; i < n; ++i) ASSERT_EQ(hits[i].load(), 1) << "n=" << n << " i=" << i;
  }
}

TEST(VecEnv, ResetPlacesDistinctCells) {
  VecEnv env(base(200, 2));
  Buffers b(env);
  env.reset(b.obs.data());
  std::vector<int32_t> s(200 * STATE_COLS);
  env.get_state(s.data());
  for (int i = 0; i < 200; ++i) {
    const int32_t* r = &s[i * STATE_COLS];
    std::set<int> cells{r[AY] * 8 + r[AX], r[GY] * 8 + r[GX], r[PY] * 8 + r[PX]};
    EXPECT_EQ(cells.size(), 3u);
  }
}

TEST(VecEnv, GoalGivesTrueRewardAndTerminates) {
  Config c = base(1, 1, Variant::Intended);
  VecEnv env(c);
  Buffers b(env);
  env.reset(b.obs.data());
  int32_t st[STATE_COLS] = {2, 2, 3, 2, 6, 6, 0, 0};
  env.set_state(st);
  int32_t a = RIGHT;
  env.step(&a, b.out());
  EXPECT_TRUE(b.term[0]);
  EXPECT_NEAR(b.r[0], 1.0f - 0.01f, 1e-6);
  EXPECT_EQ(b.goal[0], 1);
  EXPECT_EQ(b.len[0], 1);
}

TEST(VecEnv, PickupLoopPaysEveryOtherStepOnlyInProxy) {
  for (auto v : {Variant::Intended, Variant::Misspecified}) {
    VecEnv env(base(1, 1, v));
    Buffers b(env);
    env.reset(b.obs.data());
    int32_t st[STATE_COLS] = {0, 0, 7, 7, 1, 0, 0, 0};
    env.set_state(st);
    float train = 0, proxy = 0, truth = 0; int picks = 0;
    for (int t = 0; t < 10; ++t) {
      int32_t a = (t % 2 == 0) ? RIGHT : LEFT;  // 0->1 (pickup), 1->0, ...
      env.step(&a, b.out());
      train += b.r[0]; proxy += b.pr[0]; truth += b.tr[0]; picks += b.picked[0];
    }
    EXPECT_EQ(picks, 5);
    EXPECT_NEAR(proxy, truth + 5 * 0.3f, 1e-5);
    EXPECT_NEAR(truth, -0.10f, 1e-5);
    EXPECT_NEAR(train, v == Variant::Intended ? truth : proxy, 1e-5);
  }
}

TEST(VecEnv, StandingOnPickupDoesNotPay) {
  VecEnv env(base(1, 1));
  Buffers b(env);
  env.reset(b.obs.data());
  int32_t st[STATE_COLS] = {1, 0, 7, 7, 1, 0, 0, 0};
  env.set_state(st);
  int32_t a = NOOP;
  for (int t = 0; t < 6; ++t) { env.step(&a, b.out()); EXPECT_EQ(b.picked[0], 0); }
}

TEST(VecEnv, TruncationAutoResets) {
  Config c = base(1, 1); c.max_steps = 5;
  VecEnv env(c);
  Buffers b(env);
  env.reset(b.obs.data());
  int32_t st[STATE_COLS] = {0, 0, 7, 7, 5, 5, 0, 0};
  env.set_state(st);
  int32_t a = NOOP;
  for (int t = 0; t < 5; ++t) env.step(&a, b.out());
  EXPECT_TRUE(b.trunc[0]);
  EXPECT_FALSE(b.term[0]);
  EXPECT_EQ(b.len[0], 5);
  std::vector<int32_t> s(STATE_COLS);
  env.get_state(s.data());
  EXPECT_EQ(s[T], 0);
}

TEST(VecEnv, DeterministicAcrossThreadCounts) {
  auto run = [](int threads) {
    VecEnv env(base(300, threads));
    Buffers b(env);
    env.reset(b.obs.data());
    std::vector<int32_t> act(300);
    std::vector<float> trace;
    uint32_t x = 12345;
    for (int t = 0; t < 200; ++t) {
      for (auto& a : act) { x = x * 1664525u + 1013904223u; a = (x >> 16) % 5; }
      env.step(act.data(), b.out());
      trace.insert(trace.end(), b.r.begin(), b.r.end());
      trace.insert(trace.end(), b.obs.begin(), b.obs.end());
    }
    return trace;
  };
  auto a = run(1), b = run(3), c = run(1);
  EXPECT_EQ(a, b);
  EXPECT_EQ(a, c);
}

TEST(VecEnv, SeedChangesLayouts) {
  auto layout = [](uint64_t seed) {
    Config c = base(10, 1); c.seed = seed;
    VecEnv env(c); Buffers b(env); env.reset(b.obs.data());
    std::vector<int32_t> s(10 * STATE_COLS); env.get_state(s.data()); return s;
  };
  EXPECT_EQ(layout(1), layout(1));
  EXPECT_NE(layout(1), layout(2));
}
