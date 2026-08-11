/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "blocking_counter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>

#include "util/threadpool.h"

namespace xllm {

TEST(BlockingCounterTest, BasicTest) {
  BlockingCounter counter(1);
  counter.decrement_count();
  counter.wait();
  EXPECT_TRUE(counter.wait_for(std::chrono::milliseconds(0)));
}

TEST(BlockingCounterTest, TwoThreadTest) {
  ThreadPool threadpool(1);
  BlockingCounter counter(2);

  std::atomic<int32_t> called{0};
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  called.fetch_add(1, std::memory_order_relaxed);
  counter.decrement_count();
  counter.wait();
  EXPECT_EQ(2, called.load(std::memory_order_relaxed));
}

TEST(BlockingCounterTest, MultiThreadTest) {
  ThreadPool threadpool(4);
  BlockingCounter counter(5);

  std::atomic<int32_t> called{0};
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });
  called.fetch_add(1, std::memory_order_relaxed);
  counter.decrement_count();

  counter.wait();
  EXPECT_EQ(5, called.load(std::memory_order_relaxed));
}

TEST(BlockingCounterTest, WaitTimeoutTest) {
  ThreadPool threadpool(2);
  BlockingCounter counter(3);

  std::atomic<int32_t> called{0};
  threadpool.schedule([&counter, &called]() {
    called.fetch_add(1, std::memory_order_relaxed);
    counter.decrement_count();
  });

  called.fetch_add(1, std::memory_order_relaxed);
  counter.decrement_count();

  const std::chrono::milliseconds timeout(100);
  EXPECT_FALSE(counter.wait_for(timeout));
  EXPECT_EQ(2, called.load(std::memory_order_relaxed));
}

}  // namespace xllm
