// Persistent worker pool. parallel_for splits [0, n) into chunks that are
// distributed through a lock-free MPMC queue; the calling thread helps drain it.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "mpmc_queue.hpp"

namespace hg {

class ThreadPool {
 public:
  // num_threads counts the caller too: 1 means fully inline execution.
  explicit ThreadPool(size_t num_threads)
      : num_threads_(num_threads < 1 ? 1 : num_threads), queue_(4096) {
    for (size_t i = 1; i < num_threads_; ++i) workers_.emplace_back([this] { worker_loop(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_.store(true);
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  size_t size() const { return num_threads_; }

  // fn(begin, end) is invoked on disjoint sub-ranges covering [0, n).
  void parallel_for(size_t n, const std::function<void(size_t, size_t)>& fn) {
    if (n == 0) return;
    if (num_threads_ == 1) {
      fn(0, n);
      return;
    }
    size_t target_chunks = num_threads_ * 4;
    size_t chunk = (n + target_chunks - 1) / target_chunks;
    if (chunk < 16) chunk = 16;
    size_t nchunks = (n + chunk - 1) / chunk;
    if (nchunks == 1) {
      fn(0, n);
      return;
    }
    job_ = &fn;
    n_ = n;
    chunk_ = chunk;
    remaining_.store(nchunks, std::memory_order_relaxed);
    queued_.fetch_add(nchunks);
    for (size_t c = 0; c < nchunks; ++c) {
      while (!queue_.try_push(c)) std::this_thread::yield();
    }
    { std::lock_guard<std::mutex> lk(m_); }
    cv_.notify_all();
    size_t c;
    while (remaining_.load(std::memory_order_acquire) != 0) {
      if (queue_.try_pop(c)) {
        queued_.fetch_sub(1);
        run_chunk(c);
      } else {
        cpu_relax();
      }
    }
  }

 private:
  static void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
  }

  void run_chunk(size_t c) {
    size_t b = c * chunk_;
    size_t e = b + chunk_ < n_ ? b + chunk_ : n_;
    (*job_)(b, e);
    remaining_.fetch_sub(1, std::memory_order_release);
  }

  void worker_loop() {
    size_t c;
    int spins = 0;
    for (;;) {
      if (queue_.try_pop(c)) {
        queued_.fetch_sub(1);
        run_chunk(c);
        spins = 0;
        continue;
      }
      if (stop_.load()) return;
      if (++spins < 4000) {
        cpu_relax();
        continue;
      }
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [this] { return stop_.load() || queued_.load() > 0; });
      spins = 0;
    }
  }

  size_t num_threads_;
  MpmcQueue<size_t> queue_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_;
  std::atomic<bool> stop_{false};
  std::atomic<size_t> queued_{0};
  std::atomic<size_t> remaining_{0};
  const std::function<void(size_t, size_t)>* job_ = nullptr;
  size_t n_ = 0;
  size_t chunk_ = 1;
};

}  // namespace hg
