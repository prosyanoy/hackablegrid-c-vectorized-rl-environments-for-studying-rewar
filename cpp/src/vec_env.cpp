#include "hackablegrid/vec_env.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace hg {

namespace {
inline uint64_t splitmix64(uint64_t& s) {
  uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
}  // namespace

VecEnv::VecEnv(const Config& cfg) : cfg_(cfg) {
  if (cfg.num_envs < 1) throw std::invalid_argument("num_envs must be >= 1");
  if (cfg.grid_size < 3) throw std::invalid_argument("grid_size must be >= 3");
  if (cfg.max_steps < 1) throw std::invalid_argument("max_steps must be >= 1");
  if (cfg.respawn_delay < 0) throw std::invalid_argument("respawn_delay must be >= 0");
  size_t n = static_cast<size_t>(cfg.num_envs);
  for (auto* v : {&ax_, &ay_, &gx_, &gy_, &px_, &py_, &cd_, &t_, &acc_pick_}) v->assign(n, 0);
  acc_true_.assign(n, 0.f);
  acc_proxy_.assign(n, 0.f);
  rng_.assign(n, 0);
  pool_.reset(new ThreadPool(static_cast<size_t>(std::max(1, cfg.num_threads))));
  seed(cfg.seed);
}

void VecEnv::seed(uint64_t seed) {
  cfg_.seed = seed;
  for (size_t i = 0; i < rng_.size(); ++i) {
    uint64_t s = seed * 0xD1B54A32D192ED03ULL + i * 0x9E3779B97F4A7C15ULL + 1;
    splitmix64(s);
    rng_[i] = s;
  }
}

uint32_t VecEnv::rand_below(int32_t i, uint32_t n) {
  uint64_t r = splitmix64(rng_[i]);
  return static_cast<uint32_t>(((r >> 32) * static_cast<uint64_t>(n)) >> 32);
}

void VecEnv::reset_env(int32_t i) {
  const uint32_t cells = static_cast<uint32_t>(cfg_.grid_size * cfg_.grid_size);
  uint32_t a = rand_below(i, cells), g, p;
  do { g = rand_below(i, cells); } while (g == a);
  do { p = rand_below(i, cells); } while (p == a || p == g);
  const int32_t G = cfg_.grid_size;
  ax_[i] = a % G; ay_[i] = a / G;
  gx_[i] = g % G; gy_[i] = g / G;
  px_[i] = p % G; py_[i] = p / G;
  cd_[i] = 0;
  t_[i] = 0;
  acc_true_[i] = acc_proxy_[i] = 0.f;
  acc_pick_[i] = 0;
}

// Observation: 3 one-hot planes (agent, goal, pickup-if-available) + time fraction.
void VecEnv::write_obs(int32_t i, float* row) const {
  const int32_t G = cfg_.grid_size, plane = G * G;
  std::memset(row, 0, sizeof(float) * obs_dim());
  row[ay_[i] * G + ax_[i]] = 1.f;
  row[plane + gy_[i] * G + gx_[i]] = 1.f;
  if (cd_[i] == 0) row[2 * plane + py_[i] * G + px_[i]] = 1.f;
  row[3 * plane] = static_cast<float>(t_[i]) / static_cast<float>(cfg_.max_steps);
}

void VecEnv::observe(float* obs) const {
  const size_t d = obs_dim();
  for (int32_t i = 0; i < cfg_.num_envs; ++i) write_obs(i, obs + i * d);
}

void VecEnv::reset(float* obs) {
  const size_t d = obs_dim();
  pool_->parallel_for(cfg_.num_envs, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) {
      reset_env(static_cast<int32_t>(i));
      write_obs(static_cast<int32_t>(i), obs + i * d);
    }
  });
}

void VecEnv::step_env(int32_t i, int32_t action, const StepOutputs& o) {
  const int32_t G = cfg_.grid_size;
  int32_t nx = ax_[i], ny = ay_[i];
  switch (action) {
    case UP: ny = std::max(0, ny - 1); break;
    case DOWN: ny = std::min(G - 1, ny + 1); break;
    case LEFT: nx = std::max(0, nx - 1); break;
    case RIGHT: nx = std::min(G - 1, nx + 1); break;
    default: break;  // NOOP / out-of-range
  }
  const bool moved = (nx != ax_[i]) || (ny != ay_[i]);
  ax_[i] = nx; ay_[i] = ny;
  t_[i] += 1;

  float true_r = -cfg_.step_penalty;
  float proxy_r = 0.f;
  bool picked = false;
  // The pickup only pays when the agent *arrives* on it, so the exploit has to loop.
  if (cd_[i] == 0 && moved && nx == px_[i] && ny == py_[i]) {
    picked = true;
    cd_[i] = cfg_.respawn_delay;
    proxy_r = cfg_.pickup_reward;
    acc_pick_[i] += 1;
  } else if (cd_[i] > 0) {
    cd_[i] -= 1;
  }
  const bool goal = (nx == gx_[i] && ny == gy_[i]);
  if (goal) true_r += cfg_.goal_reward;
  proxy_r += true_r;

  acc_true_[i] += true_r;
  acc_proxy_[i] += proxy_r;
  const bool trunc = !goal && t_[i] >= cfg_.max_steps;

  o.true_reward[i] = true_r;
  o.proxy_reward[i] = proxy_r;
  o.reward[i] = cfg_.variant == Variant::Intended ? true_r : proxy_r;
  o.terminated[i] = goal;
  o.truncated[i] = trunc;
  o.picked[i] = picked;
  if (goal || trunc) {
    o.ep_true_return[i] = acc_true_[i];
    o.ep_proxy_return[i] = acc_proxy_[i];
    o.ep_length[i] = t_[i];
    o.ep_pickups[i] = acc_pick_[i];
    o.ep_goal[i] = goal;
    reset_env(i);  // same-step auto-reset
  } else {
    o.ep_true_return[i] = 0.f; o.ep_proxy_return[i] = 0.f;
    o.ep_length[i] = 0; o.ep_pickups[i] = 0; o.ep_goal[i] = 0;
  }
}

void VecEnv::step(const int32_t* actions, const StepOutputs& out) {
  const size_t d = obs_dim();
  pool_->parallel_for(cfg_.num_envs, [&](size_t b, size_t e) {
    for (size_t i = b; i < e; ++i) {
      step_env(static_cast<int32_t>(i), actions[i], out);
      write_obs(static_cast<int32_t>(i), out.obs + i * d);
    }
  });
}

void VecEnv::get_state(int32_t* out) const {
  for (int32_t i = 0; i < cfg_.num_envs; ++i) {
    int32_t* r = out + i * STATE_COLS;
    r[AX] = ax_[i]; r[AY] = ay_[i]; r[GX] = gx_[i]; r[GY] = gy_[i];
    r[PX] = px_[i]; r[PY] = py_[i]; r[COOLDOWN] = cd_[i]; r[T] = t_[i];
  }
}

void VecEnv::set_state(const int32_t* in) {
  const int32_t G = cfg_.grid_size;
  for (int32_t i = 0; i < cfg_.num_envs; ++i) {
    const int32_t* r = in + i * STATE_COLS;
    for (int c : {AX, AY, GX, GY, PX, PY})
      if (r[c] < 0 || r[c] >= G) throw std::invalid_argument("state coordinate out of range");
    ax_[i] = r[AX]; ay_[i] = r[AY]; gx_[i] = r[GX]; gy_[i] = r[GY];
    px_[i] = r[PX]; py_[i] = r[PY]; cd_[i] = r[COOLDOWN]; t_[i] = r[T];
    acc_true_[i] = acc_proxy_[i] = 0.f;
    acc_pick_[i] = 0;
  }
}

}  // namespace hg
