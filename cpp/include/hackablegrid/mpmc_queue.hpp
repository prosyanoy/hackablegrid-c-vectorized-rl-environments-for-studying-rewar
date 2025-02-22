// Bounded lock-free multi-producer/multi-consumer queue (Vyukov's algorithm).
#pragma once
#include <atomic>
#include <cstddef>
#include <memory>

namespace hg {

template <class T>
class MpmcQueue {
 public:
  // capacity is rounded up to a power of two (>= 2).
  explicit MpmcQueue(size_t capacity) {
    size_t cap = 2;
    while (cap < capacity) cap <<= 1;
    mask_ = cap - 1;
    cells_.reset(new Cell[cap]);
    for (size_t i = 0; i < cap; ++i) cells_[i].seq.store(i, std::memory_order_relaxed);
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
  }

  size_t capacity() const { return mask_ + 1; }

  bool try_push(const T& v) {
    Cell* cell;
    size_t pos = tail_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &cells_[pos & mask_];
      size_t seq = cell->seq.load(std::memory_order_acquire);
      intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
      if (dif == 0) {
        if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (dif < 0) {
        return false;  // full
      } else {
        pos = tail_.load(std::memory_order_relaxed);
      }
    }
    cell->data = v;
    cell->seq.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool try_pop(T& out) {
    Cell* cell;
    size_t pos = head_.load(std::memory_order_relaxed);
    for (;;) {
      cell = &cells_[pos & mask_];
      size_t seq = cell->seq.load(std::memory_order_acquire);
      intptr_t dif = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
      if (dif == 0) {
        if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) break;
      } else if (dif < 0) {
        return false;  // empty
      } else {
        pos = head_.load(std::memory_order_relaxed);
      }
    }
    out = cell->data;
    cell->seq.store(pos + mask_ + 1, std::memory_order_release);
    return true;
  }

 private:
  struct Cell {
    std::atomic<size_t> seq;
    T data;
  };
  std::unique_ptr<Cell[]> cells_;
  size_t mask_;
  alignas(64) std::atomic<size_t> head_;
  alignas(64) std::atomic<size_t> tail_;
};

}  // namespace hg
