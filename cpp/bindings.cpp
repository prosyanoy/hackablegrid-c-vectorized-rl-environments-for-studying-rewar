// pybind11 bindings. The Python-visible NativeVecEnv owns persistent NumPy output
// buffers that the engine writes into directly (zero-copy); the arrays returned by
// step()/reset() are views that are overwritten by the next call.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "hackablegrid/vec_env.hpp"

namespace py = pybind11;
using hg::Config;
using hg::VecEnv;

namespace {

template <class T>
py::array_t<T> make(std::vector<py::ssize_t> shape) {
  py::array_t<T> a(shape);
  std::memset(a.mutable_data(), 0, a.nbytes());
  return a;
}

class NativeVecEnv {
 public:
  NativeVecEnv(int num_envs, int grid_size, int max_steps, int variant, int num_threads,
               uint64_t seed, float goal_reward, float step_penalty, float pickup_reward,
               int respawn_delay)
      : env_(make_config(num_envs, grid_size, max_steps, variant, num_threads, seed, goal_reward,
                         step_penalty, pickup_reward, respawn_delay)) {
    const py::ssize_t n = num_envs, d = env_.obs_dim();
    obs_ = make<float>({n, d});
    reward_ = make<float>({n});
    true_reward_ = make<float>({n});
    proxy_reward_ = make<float>({n});
    terminated_ = make<uint8_t>({n});
    truncated_ = make<uint8_t>({n});
    picked_ = make<uint8_t>({n});
    ep_true_ = make<float>({n});
    ep_proxy_ = make<float>({n});
    ep_len_ = make<int32_t>({n});
    ep_pick_ = make<int32_t>({n});
    ep_goal_ = make<uint8_t>({n});
  }

  static Config make_config(int num_envs, int grid_size, int max_steps, int variant,
                            int num_threads, uint64_t seed, float goal_reward,
                            float step_penalty, float pickup_reward, int respawn_delay) {
    if (variant != 0 && variant != 1) throw std::invalid_argument("variant must be 0 or 1");
    Config c;
    c.num_envs = num_envs; c.grid_size = grid_size; c.max_steps = max_steps;
    c.variant = static_cast<hg::Variant>(variant);
    c.num_threads = num_threads; c.seed = seed; c.goal_reward = goal_reward;
    c.step_penalty = step_penalty; c.pickup_reward = pickup_reward;
    c.respawn_delay = respawn_delay;
    return c;
  }

  int num_envs() const { return env_.num_envs(); }
  int obs_dim() const { return env_.obs_dim(); }
  int grid_size() const { return env_.config().grid_size; }
  int max_steps() const { return env_.config().max_steps; }
  int variant() const { return static_cast<int>(env_.config().variant); }
  int num_threads() const { return env_.config().num_threads; }

  void seed(uint64_t s) { env_.seed(s); }

  py::array_t<float> reset(py::object seed) {
    if (!seed.is_none()) env_.seed(seed.cast<uint64_t>());
    py::gil_scoped_release rel;
    env_.reset(obs_.mutable_data());
    return obs_;
  }

  py::tuple step(py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions) {
    if (actions.ndim() != 1 || actions.shape(0) != env_.num_envs())
      throw std::invalid_argument("actions must have shape (num_envs,)");
    hg::StepOutputs o{obs_.mutable_data(),      reward_.mutable_data(),
                      true_reward_.mutable_data(), proxy_reward_.mutable_data(),
                      terminated_.mutable_data(),  truncated_.mutable_data(),
                      picked_.mutable_data(),      ep_true_.mutable_data(),
                      ep_proxy_.mutable_data(),    ep_len_.mutable_data(),
                      ep_pick_.mutable_data(),     ep_goal_.mutable_data()};
    {
      py::gil_scoped_release rel;
      env_.step(actions.data(), o);
    }
    py::dict info;
    info["true_reward"] = true_reward_;
    info["proxy_reward"] = proxy_reward_;
    info["picked"] = picked_;
    info["ep_true_return"] = ep_true_;
    info["ep_proxy_return"] = ep_proxy_;
    info["ep_length"] = ep_len_;
    info["ep_pickups"] = ep_pick_;
    info["ep_goal"] = ep_goal_;
    return py::make_tuple(obs_, reward_, terminated_, truncated_, info);
  }

  py::array_t<int32_t> get_state() const {
    py::array_t<int32_t> a({(py::ssize_t)env_.num_envs(), (py::ssize_t)hg::STATE_COLS});
    env_.get_state(a.mutable_data());
    return a;
  }

  py::array_t<float> set_state(py::array_t<int32_t, py::array::c_style | py::array::forcecast> s) {
    if (s.ndim() != 2 || s.shape(0) != env_.num_envs() || s.shape(1) != hg::STATE_COLS)
      throw std::invalid_argument("state must have shape (num_envs, 8)");
    env_.set_state(s.data());
    env_.observe(obs_.mutable_data());
    return obs_;
  }

 private:
  VecEnv env_;
  py::array_t<float> obs_, reward_, true_reward_, proxy_reward_, ep_true_, ep_proxy_;
  py::array_t<uint8_t> terminated_, truncated_, picked_, ep_goal_;
  py::array_t<int32_t> ep_len_, ep_pick_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "HackableGrid native batched gridworld engine";
  py::class_<NativeVecEnv>(m, "NativeVecEnv")
      .def(py::init<int, int, int, int, int, uint64_t, float, float, float, int>(),
           py::arg("num_envs"), py::arg("grid_size") = 8, py::arg("max_steps") = 64,
           py::arg("variant") = 0, py::arg("num_threads") = 1, py::arg("seed") = 0,
           py::arg("goal_reward") = 1.0f, py::arg("step_penalty") = 0.01f,
           py::arg("pickup_reward") = 0.3f, py::arg("respawn_delay") = 1)
      .def_property_readonly("num_envs", &NativeVecEnv::num_envs)
      .def_property_readonly("obs_dim", &NativeVecEnv::obs_dim)
      .def_property_readonly("grid_size", &NativeVecEnv::grid_size)
      .def_property_readonly("max_steps", &NativeVecEnv::max_steps)
      .def_property_readonly("variant", &NativeVecEnv::variant)
      .def_property_readonly("num_threads", &NativeVecEnv::num_threads)
      .def("seed", &NativeVecEnv::seed)
      .def("reset", &NativeVecEnv::reset, py::arg("seed") = py::none())
      .def("step", &NativeVecEnv::step)
      .def("get_state", &NativeVecEnv::get_state)
      .def("set_state", &NativeVecEnv::set_state);
}
