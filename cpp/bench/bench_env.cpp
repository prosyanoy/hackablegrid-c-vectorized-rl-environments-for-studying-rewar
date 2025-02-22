#include <benchmark/benchmark.h>

#include "hackablegrid/vec_env.hpp"

using namespace hg;

static void BM_VecEnvStep(benchmark::State& state) {
  Config c;
  c.num_envs = static_cast<int>(state.range(0));
  c.num_threads = static_cast<int>(state.range(1));
  c.variant = Variant::Misspecified;
  VecEnv env(c);
  size_t n = c.num_envs;
  std::vector<float> obs(n * env.obs_dim()), r(n), tr(n), pr(n), et(n), ep(n);
  std::vector<uint8_t> term(n), trunc(n), picked(n), goal(n);
  std::vector<int32_t> len(n), pick(n), act(n);
  StepOutputs o{obs.data(), r.data(), tr.data(), pr.data(), term.data(), trunc.data(),
                picked.data(), et.data(), ep.data(), len.data(), pick.data(), goal.data()};
  env.reset(obs.data());
  uint32_t x = 1;
  for (auto& a : act) { x = x * 1664525u + 1013904223u; a = (x >> 16) % 5; }
  for (auto _ : state) env.step(act.data(), o);
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK(BM_VecEnvStep)->ArgsProduct({{1024, 4096}, {1, 2, 4}})->UseRealTime();

BENCHMARK_MAIN();
