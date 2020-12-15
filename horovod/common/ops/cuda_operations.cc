// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "gpu_operations.h"
#include "cuda/cuda_kernels.h"
#include "../message.h"
#include "../logging.h"
#include <sys/time.h>

#include <thread>

namespace horovod {
namespace common {

class GPUContext::impl {
public:
  cudaError_t GetGpuEvent(cudaEvent_t* event) {
    int device;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    auto& mutex = cuda_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto& queue = cuda_events[device];
      if (!queue.empty()) {
        *event = queue.front();
        queue.pop();
        return cudaSuccess;
      }
    }

    return cudaEventCreateWithFlags(event, cudaEventDisableTiming);
  }

  cudaError_t ReleaseGpuEvent(cudaEvent_t event) {
    int device;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
      return status;
    }

    auto& mutex = cuda_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto& queue = cuda_events[device];
      queue.push(event);
    }

    return cudaSuccess;
  }

  void ErrorCheck(std::string op_name, cudaError_t cuda_result) {
    if (cuda_result != cudaSuccess) {
      throw std::logic_error(std::string(op_name) + " failed: " + cudaGetErrorString(cuda_result));
    }
  }

  void RecordEvent(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue, std::string name, cudaStream_t& stream) {
    cudaEvent_t event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event));
    ErrorCheck("cudaEventRecord", cudaEventRecord(event, stream));
    event_queue.emplace(name, event);
  }

  void WaitForEvents(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
      const std::vector<TensorTableEntry>& entries, Timeline& timeline,
      const std::function<void()>& error_check_callback) {
    while (!event_queue.empty()) {
      std::string name;
      cudaEvent_t event;
      std::tie(name, event) = event_queue.front();
      event_queue.pop();

      struct timeval start_time;
      struct timeval end_time;
      unsigned long time_taken;
      std::stringstream ss;

      if (name != "") {
        gettimeofday(&start_time, NULL);
        timeline.ActivityStartAll(entries, name);
      }

      // Check for async (networking) errors while waiting for the event to complete
      // 当一个event完成，timeline记录完成时间
      cudaError_t cuda_result;
      while (true) {
        cuda_result = cudaEventQuery(event);
        if (cuda_result == cudaSuccess) {
          break;
        }

        if (cuda_result != cudaErrorNotReady) {
          throw std::logic_error(std::string("cudaEventQuery failed: ") + cudaGetErrorString(cuda_result));
        }

        if (error_check_callback) {
          error_check_callback();
        }
        std::this_thread::yield();
      }

      if (name != "") {
        if (name == "NCCL_ALLREDUCE" || name == "NCCL_BCAST") {
          gettimeofday(&end_time, NULL);
          time_taken = 1000 * 1000 * (end_time.tv_sec - start_time.tv_sec)
                  + (end_time.tv_usec - start_time.tv_usec);
          long long size = 0;
          for (auto entry : entries) {
            size += entry.tensor->size();
          }

          LOG(TRACE) << "iietest: size in entries " << size;

          double num_of_MB = 1.5 * size / (1024 * 1024);
          double avg = 1000 * 1000 * num_of_MB * 8 / time_taken;
          ss << "iietest: Processing " << entries.size()
            << " tensors, total size:" << num_of_MB << "MB"
            << ", 执行" << name << "耗时" << time_taken * 1.0 / 1000 << "ms"
            << ", avg: " <<  avg << "Mbps.";
          LOG(TRACE) << ss.str() << std::endl;
        }
       
        timeline.ActivityEndAll(entries);
      }
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void StreamCreate(cudaStream_t *stream) {
    int greatest_priority;
    ErrorCheck("cudaDeviceGetStreamPriorityRange",
        cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    ErrorCheck("cudaStreamCreateWithPriority",
        cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority));
  }

  void StreamSynchronize(cudaStream_t stream) {
    ErrorCheck("cudaStreamSynchronize", cudaStreamSynchronize(stream));
  }

  int GetDevice() {
    int device;
    ErrorCheck("cudaGetDevice", cudaGetDevice(&device));
    return device;
  }

  void SetDevice(int device) {
    ErrorCheck("cudaSetDevice", cudaSetDevice(device));
  }

  void MemcpyAsyncD2D(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
  }

  void MemcpyAsyncH2D(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
  }

  void MemcpyAsyncD2H(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
  }

  void ScaleBufferImpl(const void* fused_input_data, void* buffer_data, int64_t num_elements,
                       double scale_factor, DataType dtype, cudaStream_t stream) {
    ScaleBufferCudaImpl(fused_input_data, buffer_data, num_elements, scale_factor, dtype, stream);

    // TODO: https://github.com/horovod/horovod/issues/2230
    //ErrorCheck("ScaleBufferCudaImpl", cudaGetLastError());
  }

private:
  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex cuda_events_mutex;
};

#include "gpu_context_impl.cc"

} // namespace common
} // namespace horovod
