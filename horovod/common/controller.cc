// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "controller.h"

#include <sys/time.h>

#include <atomic>
#include <map>
#include <queue>
#include <set>
#include <unordered_set>
#include <string>
#include <sstream>

#include "global_state.h"
#include "logging.h"
#include "operations.h"

#if HAVE_CUDA
#include "ops/cuda/cuda_kernels.h"
#endif


namespace horovod {
namespace common {


void Controller::SynchronizeParameters() {
  ParameterManager::Params param;
  if (is_coordinator_) {
    param = parameter_manager_.GetParams();
  }

  void* buffer = (void*)(&param);
  size_t param_size = sizeof(param);
  Bcast(buffer, param_size, 0, Communicator::GLOBAL);

  if (!is_coordinator_) {
    parameter_manager_.SetParams(param);
  }
  parameter_manager_.Reset();
}

// tensor_queue应该存放的是算出来的梯度 
Controller::Controller(ResponseCache& response_cache, TensorQueue& tensor_queue,
                       Timeline& timeline, ParameterManager& parameter_manager)
    : stall_inspector_(response_cache), tensor_queue_(tensor_queue),
      timeline_(timeline), response_cache_(response_cache),
      parameter_manager_(parameter_manager) {}

void Controller::Initialize() {
  response_cache_.clear();

  // Initialize concrete implementations.
  DoInitialization();
}

//为profiling增加的函数定义
template <typename Container>
void Print_Response_Info(std::string prefix_str, Container& responses, int rank) {
  std::stringstream ss;
  int64_t total_size = 0;

  ss << prefix_str << rank;
  for (auto& response : responses) {
    int total_size_in_response = 0;
    for (auto& size : response.tensor_sizes()) { //size表示一个tensor中有多少个元素
      total_size_in_response += size;
    }
    total_size += total_size_in_response;
    // ss << "; num of tensors in a response: " << response.tensor_sizes().size()
    //    << "; Total Tensor Size in this response: " << total_size_in_response << ". ";
  }

  ss << "size of response queue: " << responses.size()
     << "; Total Tensor Size in response queue: " << total_size << ".";
  LOG(TRACE) << ss.str() << std::endl;
}

//coordination阶段的通信（通过response cache或worker-coordinator直接的request/response）
//都在该函数中完成
ResponseList Controller::ComputeResponseList(std::atomic_bool& shut_down,
                                             HorovodGlobalState& state) {
  //为profiling增加的变量定义
  struct timeval start_time;
  struct timeval end_time;
  unsigned long time_taken;
  int total_size;  
  std::stringstream ss;

  // Update cache capacity if autotuning is active.
  if (parameter_manager_.IsAutoTuning()) {
    response_cache_.set_capacity((int)parameter_manager_.CacheEnabled() *
                                 cache_capacity_);
  }

  // Copy the data structures out from parameters.
  // However, don't keep the lock for the rest of the loop, so that
  // enqueued stream callbacks can continue.
  CacheCoordinator cache_coordinator(response_cache_.num_active_bits());

  // message queue used only in this cycle
  // 从tensor_queue_中将message_queue_tmp填充

  // JOIN 是什么操作？
  std::deque<Request> message_queue_tmp;
  tensor_queue_.PopMessagesFromQueue(message_queue_tmp);  //tensor_queue_是算出来的梯度
  int num_hit = 0, num_invalid = 0;

  for (auto& message : message_queue_tmp) {
    if (message.request_type() == Request::JOIN) {
      state.joined = true;
      cache_coordinator.set_uncached_in_queue(true);
      continue;
    }

    // Keep track of cache hits，把命中cache的request记录到cache coordinator中
    if (response_cache_.capacity() > 0) {
      //判断request对应的response是否在cache中
      auto cache_ = response_cache_.cached(message);
      if (cache_ == ResponseCache::CacheState::HIT) {   //如果在
        uint32_t cache_bit = response_cache_.peek_cache_bit(message);
        cache_coordinator.record_hit(cache_bit);
        num_hit++;
        // Record initial time cached tensor is encountered in queue.
        stall_inspector_.RecordCachedTensorStart(message.tensor_name());

      } else {                                         //如果不在
        if (cache_ == ResponseCache::CacheState::INVALID) {
          uint32_t cache_bit = response_cache_.peek_cache_bit(message);
          cache_coordinator.record_invalid_bit(cache_bit);
          num_invalid++;
        }
        cache_coordinator.set_uncached_in_queue(true);

        // Remove timing entry if uncached or marked invalid.
        stall_inspector_.RemoveCachedTensor(message.tensor_name());
      }
    }
  }
  LOG(TRACE) << "iietest: num of request hit cache:" << num_hit
             << ", cache invalid:" << num_invalid;

  // join集合在一块操作？ 如果state.joined 将response_cache_中的request所对应的bit全部放入cache_coordinator
  if (state.joined && response_cache_.capacity() > 0) {
    for (uint32_t bit : response_cache_.list_all_bits()) {
      cache_coordinator.record_hit(bit);
    }
  }

  // Flag indicating that the background thread should shut down.
  bool should_shut_down = shut_down;

  // Check for stalled tensors.
  if (stall_inspector_.ShouldPerformCheck()) {
    if (is_coordinator_) {
      should_shut_down |= stall_inspector_.CheckForStalledTensors(size_);
    }

    if (response_cache_.capacity() > 0) {
      stall_inspector_.InvalidateStalledCachedTensors(cache_coordinator);
    }
    stall_inspector_.UpdateCheckTime();
  }

  cache_coordinator.set_should_shut_down(should_shut_down);

  if (response_cache_.capacity() > 0) {
    // Obtain common cache hits and cache invalidations across workers. Also,
    // determine if any worker has uncached messages in queue or requests
    // a shutdown. This function removes any invalid cache entries, if they
    // exist.
    //---------------在这里，worker之间通过bitvector的allreduce操作完成协调过程------------
    gettimeofday(&start_time, NULL);
    CoordinateCacheAndState(cache_coordinator);
    gettimeofday(&end_time, NULL);
    time_taken = 1000 * (end_time.tv_sec-start_time.tv_sec)
                 + (end_time.tv_usec-start_time.tv_usec) / 1000;
    LOG(TRACE) << "iietest: " << "ResponseCache同步，耗时："
               << time_taken << "ms"; 
       
    // Remove uncommon cached tensors from queue and replace to state
    // queue for next cycle. Skip adding common cached tensors to
    // queue as they are handled separately.

    // 经过CoordinateCacheAndState(cache_coordinator) 之后，现在 cache_coordinator
    // 中访存的都是common cache hits and cache invalidations ,下面将不在更新后的
    // cache_coordinator中的Request处理出来，然后放入tensor_queue_中，下一轮进行处理

    std::deque<Request> messages_to_replace;
    size_t num_messages = message_queue_tmp.size();
    for (size_t i = 0; i < num_messages; ++i) {
      auto& message = message_queue_tmp.front();
      if (response_cache_.cached(message) == ResponseCache::CacheState::HIT) {  //request命中cache
        uint32_t cache_bit = response_cache_.peek_cache_bit(message);
        //uncommon cached tensors(不是在所有rank上都准备好的tensor)，放入messages_to_replace
        if (cache_coordinator.cache_hits().find(cache_bit) ==
            cache_coordinator.cache_hits().end()) { 
          // Try to process again in next cycle.
          messages_to_replace.push_back(std::move(message));
        } else {       //common tensor，可以进行allreduce了，从stall_inspector里移出
          // Remove timing entry for messages being handled this cycle.
          // 这一轮进行处理的话就不需要进行监控
          stall_inspector_.RemoveCachedTensor(message.tensor_name());
        }
      } else {                                                                //request未命中cache
        // Remove timing entry for messages being handled this cycle.
        stall_inspector_.RemoveCachedTensor(message.tensor_name());
        message_queue_tmp.push_back(std::move(message));               //放入队尾
      }
      message_queue_tmp.pop_front();
    }
    tensor_queue_.PushMessagesToQueue(messages_to_replace);
  }

  // message_queue_tmp 是这轮需要协调的所有Request
  if (!message_queue_tmp.empty()) {
    LOG(TRACE, rank_) << "iietest: " << "Sent " << message_queue_tmp.size()
                      << " messages to coordinator.";
  }
  
  ResponseList response_list;
  response_list.set_shutdown(cache_coordinator.should_shut_down());

  bool need_communication = true;
  if (response_cache_.capacity() > 0 &&
      !cache_coordinator.uncached_in_queue()) {
    // if cache is enabled and no uncached new message coming in, no need for
    // additional communications
    need_communication = false;
    
    // If no messages to send, we can simply return an empty response list;
    if (cache_coordinator.cache_hits().empty()) {
      return response_list;
    }
    // otherwise we need to add cached messages to response list.
  }

  if (!need_communication) {  //worker和coordinator不需要通信即可完成协商(ResponseCache的作用)
    // If all messages in queue have responses in cache, use fast path with
    // no additional coordination.
    // 如果所有在queue中的request 在 responses cache中都存在，则不需要communication
    std::deque<Response> responses;
    // Convert cache hits to responses. Populate so that least
    // recently used responses get priority. All workers call the code
    // here so we use the get method here to consistently update the cache
    // order.
    // LRU list
    for (auto bit : cache_coordinator.cache_hits()) {
      responses.push_back(response_cache_.get_response(bit));
    }

    // Fuse responses as normal.
    // FuseResponses 用来将满足要求的responses合并
    // 合并前的responses与合并后的 response_list 进行比较
    Print_Response_Info("iietest: no need comm. before FuseResponses:", responses, rank_);
    
    gettimeofday(&start_time, NULL);
    response_list = FuseResponses(responses, state);
    gettimeofday(&end_time, NULL);
    time_taken = 1000 * (end_time.tv_sec - start_time.tv_sec)
                 + (end_time.tv_usec - start_time.tv_usec) / 1000;
    LOG(TRACE) << "iietest: FuseResponses耗时：" << time_taken;
    
    Print_Response_Info("iietest: no need comm. after FuseResponses:", response_list.responses(), rank_);
    response_list.set_shutdown(cache_coordinator.should_shut_down());
  } else {  //worker和coordinator通信的部分。request和response的通信都在下面这段代码中
    // There are uncached messages coming in, need communication to figure out
    // whether those are ready to be reduced.

    // Collect all tensors that are ready to be reduced. Record them in the
    // tensor count table (rank zero) or send them to rank zero to be
    // recorded (everyone else).

    // 需要进行communication
    // message_queue_tmp是这轮需要协调的所有 Request
    std::vector<std::string> ready_to_reduce;

    if (is_coordinator_) {  //rank0的工作
      LOG(TRACE, rank_) << "iietest: Adding messages from rank 0";

      //这个循环首先处理的是rank0本身ready的Tensor,接收其它worker的request并处理在该循环之后，同样的代码逻辑
      while (!message_queue_tmp.empty()) {      
        // Pop the first available message
        Request message = message_queue_tmp.front();
        message_queue_tmp.pop_front();

        if (message.request_type() == Request::JOIN) {
          state.joined_size++;
          continue;
        }

        //判断message包含的tensor是否可以进行allreduce了（在所有节点上都ready）
        bool reduce = IncrementTensorCount(message, state.joined_size);    

        // Record initial time for an uncached tensor is encountered in queue.
        stall_inspector_.RecordUncachedTensorStart(             
            message.tensor_name(), message.request_rank(), size_);
        if (reduce) {
          ready_to_reduce.push_back(message.tensor_name());
        }
      }

      // Receive ready tensors from other ranks
      // 先处理rank0 上的，然后从其他rank同步 ready_tensor ，然后在处理其他tensor上的
      std::vector<RequestList> ready_list;

      //该方法在controller的子类中实现,ready_to_reduce 在mpi实现里并没有用上
      //通过mpi gather接收其他rank发来的RequestList（按 rank 顺序放入 ready_list）
      // 记录同步前时间
      gettimeofday(&start_time, NULL);
      RecvReadyTensors(ready_to_reduce, ready_list);
      gettimeofday(&end_time, NULL);
      time_taken = 1000 * (end_time.tv_sec - start_time.tv_sec)
                   + (end_time.tv_usec - start_time.tv_usec) / 1000;
      LOG(TRACE) << "iietest: rank0接收请求耗时：" << time_taken;   
      
      // Process messages. 循环从1开始，不包括rank0本身。处理收到的其它worker的request，判断相应tensor能否reduce
      for (int i = 1; i < size_; ++i) {
        LOG(TRACE) << "Adding messages from rank " << i;
        auto received_message_list = ready_list[i];
        for (auto& received_message : received_message_list.requests()) {
          auto& received_name = received_message.tensor_name();

          if (received_message.request_type() == Request::JOIN) {
            state.joined_size++;
            continue;
          }

          bool reduce = IncrementTensorCount(received_message, state.joined_size);
          stall_inspector_.RecordUncachedTensorStart(
              received_message.tensor_name(), received_message.request_rank(),
              size_);
          if (reduce) {
            ready_to_reduce.push_back(received_name);
          }
        }
        if (received_message_list.shutdown()) {
          // Received SHUTDOWN request from one of the workers.
          should_shut_down = true;
        }
      }

      // Check if tensors from previous ticks are ready to reduce after Joins. 为什么是previous tick?
      // Join具体代表什么操作？
      if (state.joined_size > 0) {
        for (auto& table_iter : message_table_) {
          int count = (int)table_iter.second.size();
          // 如果所有的 rank 都发来了这个 tensor 的 request，且这个 tensor 不在 ready_to_reduce 列表里
          if (count == (size_ - state.joined_size) &&
              std::find(ready_to_reduce.begin(), ready_to_reduce.end(),
                        table_iter.first) == ready_to_reduce.end()) {
            state.timeline.NegotiateEnd(table_iter.first);
            ready_to_reduce.push_back(table_iter.first);
          }
        }
      }

      // At this point, rank zero should have a fully updated tensor count
      // table and should know all the tensors that need to be reduced or
      // gathered, and everyone else should have sent all their information
      // to rank zero. We can now do reductions and gathers
      // rank zero will choose which ones and in what order, and will 
      // notify the other ranks before doing each reduction.
      std::deque<Response> responses;

      if (response_cache_.capacity() > 0) {
        // Prepopulate response list with cached responses. Populate so that
        // least recently used responses get priority. Since only the
        // coordinator rank calls this code, use peek instead of get here to
        // preserve cache order across workers.
        // No need to do this when all ranks did Join.
        if (state.joined_size < size_) {
          for (auto bit : cache_coordinator.cache_hits()) {
            responses.push_back(response_cache_.peek_response(bit));
          }
        }
      }

      for (auto& tensor_name : ready_to_reduce) {  //为每个要reduce的tensor构造一个response消息
        Response response = ConstructResponse(tensor_name, state.joined_size);
        responses.push_back(std::move(response));
      }
      if (state.joined_size == size_) {
        // All ranks did Join(). Send the response, reset joined size.
        Response join_response;
        join_response.set_response_type(Response::JOIN);
        join_response.add_tensor_name(JOIN_TENSOR_NAME);
        responses.push_back(std::move(join_response));
        state.joined_size = 0;
      }
      
      Print_Response_Info("iietest: need comm. before FuseResponses:", responses, rank_);
            
      //把符合条件的response合并，降低通信开销
      // ready_to_reduce 转换为responses，然后再经过FuseResponses 转换为最终的response_list
      gettimeofday(&start_time, NULL);
      response_list = FuseResponses(responses, state);
      gettimeofday(&end_time, NULL);
      time_taken = 1000 * (end_time.tv_sec - start_time.tv_sec)
                   + (end_time.tv_usec - start_time.tv_usec) / 1000;
      
      LOG(TRACE) << "iietest: FuseResponses耗时：" << time_taken;
      Print_Response_Info("iietest: need comm. after FuseResponses:", response_list.responses(), rank_);
      
      response_list.set_shutdown(should_shut_down);
      
      // Broadcast final results to other ranks.在controller子类中实现
      gettimeofday(&start_time, NULL);
      SendFinalTensors(response_list);
      gettimeofday(&end_time, NULL);
      time_taken = 1000 * (end_time.tv_sec - start_time.tv_sec)
                   + (end_time.tv_usec - start_time.tv_usec) / 1000;
      LOG(TRACE) << "iietest: rank0广播耗时：" << time_taken;
    } else {         //worker的工作
      RequestList message_list;
      message_list.set_shutdown(should_shut_down);
      while (!message_queue_tmp.empty()) {
        message_list.add_request(message_queue_tmp.front());
        message_queue_tmp.pop_front();
      }

      // Send ready tensors to rank zero
      //把message_list中的request通过一个MPI_GATHER操作发给rank0

      // 记录 worker 传送给 coordinator 的数据以及时间
      // 数据量通过massage_list获取
      // request_type()
      // root_rank()
      // tensor_name()
      // tensor_shape()
      LOG(TRACE) << "iietest: before " << rank_ <<" MPI_GATHER. ";
      for (auto& request : message_list.requests()) {
        ss.str("");
        ss << "iietest: Request type:" << request.RequestType_Name(request.request_type())
           << ",Root rank:" << request.root_rank()
           << ",Tensor shape:<";

        for (auto& size : request.tensor_shape()) {
           ss << size <<",";
        }
        ss << ">.";
        LOG(TRACE) << ss.str() << std::endl;
      }

      
      gettimeofday(&start_time, NULL);
      SendReadyTensors(message_list);
      gettimeofday(&end_time, NULL);
      time_taken = 1000 * (end_time.tv_sec - start_time.tv_sec)
                   + (end_time.tv_usec - start_time.tv_usec) / 1000;
      LOG(TRACE) << "iietest: MPI_GATHER耗时：" << time_taken;

      // Receive final tensors to be processed from rank zero
      gettimeofday(&start_time, NULL);
      RecvFinalTensors(response_list);
      gettimeofday(&end_time, NULL);
      time_taken = 1000 * (end_time.tv_sec - start_time.tv_sec)
                   + (end_time.tv_usec - start_time.tv_usec) / 1000;
      LOG(TRACE) << "iietest: 接收Response耗时：" << time_taken;
      // 记录接收完成之后的数据
      // 通过response_list获取
      // response_type()
      // tensor_type()
      // tensor_names_string()
      // error_message()
      // tensor_sizes()
      /*
      LOG(TRACE) << "iietest: 从coordinator接受的具体信息为";      
      for (auto& response : response_list.responses()) {
        total_size = 0;        
        for (auto& size : response.tensor_sizes()) {
          total_size += size;          
        }

        LOG(TRACE) << "iietest: " << "-------------------------------------";
        LOG(TRACE) << "iietest:  Response Type:" << response.ResponseType_Name(response.response_type())
                   << "; Tensor Name:" << response.tensor_names_string()        
                   << "; Total tensor size:" << total_size;        
        LOG(TRACE) << "iietest: " << "-------------------------------------";
      }
      */
    }
  }


  //至此，coordination阶段的通信全部完成。
  /*-------------------------------------------------------------------------------*/
 
   // If need_communication is false, meaning no uncached message coming in,
  // thus no need to update cache.
  // 将此次得到的response 放入response_cache_ 中
  if (need_communication && response_cache_.capacity() > 0) {
    // All workers add supported responses to cache. This updates the cache
    // order consistently across workers.
    for (auto& response : response_list.responses()) {
      if ((response.response_type() == Response::ResponseType::ALLREDUCE ||
           response.response_type() == Response::ResponseType::ADASUM ||
           response.response_type() == Response::ResponseType::ALLTOALL) &&
          (int)response.devices().size() == size_) {
        response_cache_.put(response, tensor_queue_, state.joined);
      }
    }
  }

  // Reassign cache bits based on current cache order.
  response_cache_.update_cache_bits();

  return response_list;
}

int64_t Controller::TensorFusionThresholdBytes() {
  int64_t proposed_fusion_threshold =
      parameter_manager_.TensorFusionThresholdBytes();

  // If the cluster is homogeneous,
  // adjust buffer size to make sure it is divisible by local_size to improve
  // performance for operations that perform local reductions by default such as Adasum.
  if (is_homogeneous_) {
    // Assume the worst-case data type float64, since if it is divisible with
    // float64, it will be divisible for other types too.

    // Ensuring that fusion buffer can hold a number of elements divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for performance
    int double_size = GetTypeSize(HOROVOD_FLOAT64);
    int64_t div = local_size_ * double_size * FUSION_BUFFER_ATOMIC_UNIT;
    return ((proposed_fusion_threshold + div - 1) / div) * div;
  }
  return proposed_fusion_threshold;
}

Response Controller::ConstructResponse(std::string& name, int joined_size) {
  bool error = false;
  auto it = message_table_.find(name);
  assert(it != message_table_.end());

  std::vector<Request>& requests = it->second;
  assert(!requests.empty());

  std::ostringstream error_message_stream;

  // Check that all data types of tensors being processed
  // are identical.
  auto data_type = requests[0].tensor_type();
  for (unsigned int i = 1; i < requests.size(); ++i) {
    auto request_type = requests[i].tensor_type();
    if (data_type != request_type) {
      error = true;
      error_message_stream << "Mismatched data types: One rank had type "
                           << DataType_Name(data_type)
                           << ", but another rank had type "
                           << DataType_Name(request_type) << ".";
      break;
    }
  }

  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  for (unsigned int i = 1; i < requests.size(); ++i) {
    if (error) {
      break;
    }

    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream << "Mismatched operations: One rank did an "
                           << Request::RequestType_Name(message_type)
                           << ", but another rank did an "
                           << Request::RequestType_Name(request_type) << ".";
      break;
    }
  }

  // If we are doing an allreduce or broadcast, check that all tensor shapes are
  // identical.
  if (message_type == Request::ALLREDUCE ||
      message_type == Request::ADASUM ||
      message_type == Request::BROADCAST) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape != request_shape) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of shape "
            << tensor_shape.DebugString()
            << ", but another rank sent a tensor of shape "
            << request_shape.DebugString() << ".";
        break;
      }
    }
  }

  // If we are doing an allreduce, check that prescaling and postscaling factors
  // are identical across ranks.
  double prescale_factor;
  double postscale_factor;
  if (message_type == Request::ALLREDUCE ||
      message_type == Request::ADASUM) {
    prescale_factor = requests[0].prescale_factor();
    postscale_factor = requests[0].postscale_factor();

    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }
      double request_prescale_factor = requests[i].prescale_factor();
      double request_postscale_factor = requests[i].postscale_factor();

      if (prescale_factor != request_prescale_factor ||
          postscale_factor != request_postscale_factor) {
        error = true;
        error_message_stream
            << "Mismatched prescale and/or postscale factors: "
            << "One rank sent factors (" << prescale_factor
            << ", " << postscale_factor << "), but another rank "
            << "sent factors (" << request_prescale_factor
            << ", " << request_postscale_factor << ").";
        break;
      }
    }
  }

  std::vector<int64_t> tensor_sizes;
  if (message_type == Request::ALLGATHER ||
      message_type == Request::ALLTOALL) {
    if (joined_size > 0) {
      error = true;
      if (message_type == Request::ALLGATHER) {
        error_message_stream << "Allgather is not supported with Join at this time. "
                             << "Specify sparse_to_dense=True if using DistributedOptimizer";
      } else if (message_type == Request::ALLTOALL) {
        error_message_stream << "Alltoall is not supported with Join at this time.";
      }
    }

    // If we are doing an allgather/alltoall, make sure all but the first dimension are
    // the same. The first dimension may be different and the output tensor is
    // the sum of the first dimension. Collect the sizes by rank for allgather only.
    tensor_sizes.resize(requests.size());
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }

    if (tensor_shape.dims() == 0) {
      error = true;
      error_message_stream << "Rank zero tried to "
                           << Request::RequestType_Name(message_type)
                           << " a rank-zero tensor.";
    } else {
      tensor_sizes[requests[0].request_rank()] = tensor_shape.dim_size(0);
    }

    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape.dims() != request_shape.dims()) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of rank "
            << tensor_shape.dims()
            << ", but another rank sent a tensor of rank "
            << request_shape.dims() << ".";
        break;
      }

      bool dim_mismatch = false;
      for (int dim = 1; dim < tensor_shape.dims(); ++dim) {
        if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
          error = true;
          error_message_stream
              << "Mismatched " << Request::RequestType_Name(message_type)
              << " tensor shapes: One rank sent a tensor with dimension " << dim
              << " equal to " << tensor_shape.dim_size(dim)
              << ", but another rank sent a tensor with dimension " << dim
              << " equal to " << request_shape.dim_size(dim) << ".";
          dim_mismatch = true;
          break;
        }
      }
      if (dim_mismatch) {
        break;
      }

      // Collect first dimension sizes for allgather to use for fusion and allgather op.
      if (message_type == Request::ALLGATHER) {
        tensor_sizes[requests[i].request_rank()] = request_shape.dim_size(0);
      }
    }
  }

  if (message_type == Request::ALLREDUCE || message_type == Request::ADASUM) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    tensor_sizes.push_back(tensor_shape.num_elements()); //num_elements()返回tensor中有多少个元素
  }

  if (message_type == Request::BROADCAST) {
    if (joined_size > 0) {
      error = true;
      error_message_stream << "Broadcast is not supported with Join at this time.";
    }

    // If we are doing a broadcast, check that all root ranks are identical.
    int first_root_rank = requests[0].root_rank();
    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      int this_root_rank = requests[i].root_rank();
      if (first_root_rank != this_root_rank) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " root ranks: One rank specified root rank " << first_root_rank
            << ", but another rank specified root rank " << this_root_rank
            << ".";
        break;
      }
    }
  }

  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); ++i) {
    if (error) {
      break;
    }

    bool this_device_is_cpu = requests[i].device() == CPU_DEVICE_ID;
    if (first_device_is_cpu != this_device_is_cpu) {
      error = true;
      error_message_stream
          << "Mismatched " << Request::RequestType_Name(message_type)
          << " CPU/GPU device selection: One rank specified device "
          << (first_device_is_cpu ? "CPU" : "GPU")
          << ", but another rank specified device "
          << (this_device_is_cpu ? "CPU" : "GPU") << ".";
      break;
    }
  }
  std::vector<int32_t> devices(requests.size());
  for (auto& request : requests) {
    devices[request.request_rank()] = request.device();
  }

  Response response;
  response.add_tensor_name(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(Response::ERROR);
    response.set_error_message(error_message);
  } else if (message_type == Request::ALLGATHER) {
    response.set_response_type(Response::ALLGATHER);
    for (auto dim : tensor_sizes) {
      response.add_tensor_size(dim);
    }
  } else if (message_type == Request::ALLREDUCE) {
    response.set_response_type(Response::ALLREDUCE);
    for (auto dim : tensor_sizes) {
      response.add_tensor_size(dim);
    }
    response.set_tensor_type(data_type);
    response.set_prescale_factor(prescale_factor);
    response.set_postscale_factor(postscale_factor);
  } else if (message_type == Request::BROADCAST) {
    response.set_response_type(Response::BROADCAST);
  } else if (message_type == Request::ALLTOALL) {
    response.set_response_type(Response::ALLTOALL);
  } else if (message_type == Request::ADASUM) {
    response.set_response_type(Response::ADASUM);
    for (auto dim : tensor_sizes) {
      response.add_tensor_size(dim);
    }
    response.set_tensor_type(data_type);
    response.set_prescale_factor(prescale_factor);
    response.set_postscale_factor(postscale_factor);
  }
  response.set_devices(devices);

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed response.
  message_table_.erase(it);
  stall_inspector_.RemoveUncachedTensor(name);

  return response;
}

void Controller::CoordinateCacheAndState(CacheCoordinator& cache_coordinator) {
  // Sync cache and state information across workers. 完成实际的协调过程
  cache_coordinator.sync(shared_from_this(), timeline_enabled_);

  // If invalid cache entries exist, erase associated entries.
  if (!cache_coordinator.invalid_bits().empty()) {
    for (auto bit : cache_coordinator.invalid_bits()) {
      response_cache_.erase_response(bit);
    }
  }

  if (timeline_enabled_) {
    // Start/continue negotiation phase on timeline bit entries.
    for (auto bit : cache_coordinator.timeline_bits()) {
      auto& response = response_cache_.peek_response(bit);
      timeline_.NegotiateStart(response.tensor_names()[0],
                               (Request::RequestType)response.response_type());
    }

    // End negotiation phase for synced cache hit set entries.
    for (auto bit : cache_coordinator.cache_hits()) {
      auto& response = response_cache_.peek_response(bit);
      timeline_.NegotiateEnd(response.tensor_names()[0]);
    }
  }
}

//把多个response里的内容放入一个response对象(需满足如下条件)，再把这些response对象放入ResponseList返回
//1.执行的操作类型相同
//2.在同一个设备上
//3.tensor类型相同
//4.tensor大小之和不超过tensor fusion buffer大小
//5.prescale_factor和postscale_factor相同
ResponseList Controller::FuseResponses(std::deque<Response>& responses,
                                       HorovodGlobalState& state) {
  ResponseList response_list;
  while (!responses.empty()) {

    auto response = responses.front();
    assert(response.tensor_names().size() == 1);
    responses.pop_front();
    int64_t tensor_size = 0;
    if (response.response_type() == Response::ResponseType::ALLREDUCE ||
        response.response_type() == Response::ResponseType::ADASUM) {
      // Attempt to add more responses to this fused response.
      //因为一个response里可能会包含多个tensor的信息，所以tensor_sizes是个vector
      //执行response fuse前，response里只包含一个tensor的信息
      tensor_size = response.tensor_sizes()[0] * GetTypeSize(response.tensor_type());
#if HAVE_CUDA
      if (state.batch_d2d_memcopies) {
        // Add 16 byte pad for batched memcpy op
        tensor_size = BATCHED_D2D_PADDING * ((tensor_size + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
      }
#endif
      std::deque<Response> skipped_responses;
      int64_t skipped_size = 0;
      while (!responses.empty()) {
        auto& new_response = responses.front();
        assert(new_response.tensor_names().size() == 1);

        int64_t new_tensor_size = new_response.tensor_sizes().empty()
                                      ? 0
                                      : new_response.tensor_sizes()[0] *
                                        GetTypeSize(new_response.tensor_type());

#if HAVE_CUDA
        if (state.batch_d2d_memcopies) {
          // Add 16 byte pad for batched memcpy op
          new_tensor_size = BATCHED_D2D_PADDING * ((new_tensor_size + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
        }
#endif

        if (response.response_type() == new_response.response_type() &&
            response.devices() == new_response.devices() &&
            response.tensor_type() == new_response.tensor_type() &&
            tensor_size + new_tensor_size <= TensorFusionThresholdBytes() &&
            response.prescale_factor() == new_response.prescale_factor() &&
            response.postscale_factor() == new_response.postscale_factor()) {
          // These tensors will fuse together well.
          tensor_size += new_tensor_size;
          response.add_tensor_name(std::move(new_response.tensor_names()[0]));
          response.add_tensor_size(new_response.tensor_sizes()[0]);
          responses.pop_front();
        } else {  //不能fuse
          // In general, don't try to fuse additional tensors since they are
          // usually computed in order of requests and skipping tensors may
          // mean that the batch will have to wait longer while skipped
          // tensors could be reduced at that time. However, mixed-precision
          // training may yield requests of various dtype in a mixed-up
          // sequence causing breakups in fusion. To counter this some look
          // ahead is allowed.

          skipped_size += new_tensor_size;
          if (tensor_size + skipped_size <= TensorFusionThresholdBytes()) {
            // Skip response and look ahead for more to fuse.
            skipped_responses.push_back(std::move(new_response));
            responses.pop_front();
          } else {
            break;
          }
        }
      }
      // Replace any skipped responses.
      while (!skipped_responses.empty()) {
        responses.push_front(std::move(skipped_responses.back()));
        skipped_responses.pop_back();
      }

    } else if (response.response_type() == Response::ResponseType::ALLGATHER) {
      // Attempt to add more responses to this fused response.
      const auto& entry =
          tensor_queue_.GetTensorEntry(response.tensor_names()[0]);

      // This is size of first dimension.
      int64_t total_byte_size_of_output =
          TotalByteSizeOfAllgatherOutput(response.tensor_sizes(), entry);

      std::deque<Response> skipped_responses;
      int64_t skipped_size = 0;
      while (!responses.empty()) {

        auto& new_response = responses.front();
        assert(new_response.tensor_names().size() == 1);
        const auto& new_entry =
            tensor_queue_.GetTensorEntry(new_response.tensor_names()[0]);

        int64_t new_total_byte_size_of_output = TotalByteSizeOfAllgatherOutput(
            new_response.tensor_sizes(), new_entry);

        if (response.response_type() == new_response.response_type() &&
            response.devices() == new_response.devices() &&
            entry.tensor->dtype() == new_entry.tensor->dtype() &&
            total_byte_size_of_output + new_total_byte_size_of_output <=
                TensorFusionThresholdBytes()) {

          // These tensors will fuse together well.
          total_byte_size_of_output += new_total_byte_size_of_output;
          response.add_allgather_response(new_response);
          responses.pop_front();

        } else {
          // In general, don't try to fuse additional tensors since they are
          // usually computed in order of requests and skipping tensors may
          // mean that the batch will have to wait longer while skipped
          // tensors could be reduced at that time. However, mixed-precision
          // training may yield requests of various dtype in a mixed-up
          // sequence causing breakups in fusion. To counter this some look
          // ahead is allowed.

          skipped_size += new_total_byte_size_of_output;
          if (total_byte_size_of_output + skipped_size <=
              TensorFusionThresholdBytes()) {
            // Skip response and look ahead for more to fuse.
            skipped_responses.push_back(std::move(new_response));
            responses.pop_front();
          } else {
            break;
          }
        }
      }

      // Replace any skipped responses.
      while (!skipped_responses.empty()) {
        responses.push_front(std::move(skipped_responses.back()));
        skipped_responses.pop_back();
      }
    }

    response_list.add_response(std::move(response));
    LOG(TRACE) << "Created response of size " << tensor_size;
  }
  return response_list;
}

int64_t Controller::TotalByteSizeOfAllgatherOutput(
    const std::vector<int64_t>& tensor_sizes, const TensorTableEntry& entry) {
  int64_t total_dimension_size = 0;
  for (auto sz : tensor_sizes) {
    total_dimension_size += sz;
  }
  // Every tensor participating in Allgather operation may have
  // different first dimension size, but the rest of dimensions are same
  // for all tensors.  Here we get shape of tensor sliced by first
  // dimension. Allgather output will have shape of: (sum of first
  // dimension of every tensor) x (tensor slice shape).
  int64_t total_count_of_output_entries = total_dimension_size;
  for (int i = 1; i < entry.tensor->shape().dims(); ++i) {
    total_count_of_output_entries *= entry.tensor->shape().dim_size(i);
  }
  int element_size = GetTypeSize(entry.tensor->dtype());
  int64_t total_byte_size_of_output =
      total_count_of_output_entries * element_size;

  return total_byte_size_of_output;
}

int Controller::GetLocalSizeAtCrossRank(int i) {
  return local_sizes_for_cross_rank_[i];
}

/*判断Request中的tensor是否在所有的rank上都准备好了(可以进行allreduce)，该方法只在coordinator上运行
  message_table_是一个map，保存了coordinator收到的所有request消息。key是Tensor name, value是一个vector，
  长度是rank的个数，其中的元素是每个rank发来的request。当这个vector填满的时候，对应的tensor就可以进行allreduce了
*/
bool Controller::IncrementTensorCount(const Request& msg, int joined_size) {
  auto& name = msg.tensor_name();
  auto table_iter = message_table_.find(name);
  if (table_iter == message_table_.end()) {               //msg不在message_table_中，则加入
    std::vector<Request> messages = {msg};
    messages.reserve(static_cast<unsigned long>(size_));  //size_是rank的个数
    message_table_.emplace(name, std::move(messages));
    table_iter = message_table_.find(name);
    timeline_.NegotiateStart(name, msg.request_type());   //对应Tensor的negotiation过程开始
  } else {                                               //msg已经在message_table_中，则把msg加入对应的vector   
    std::vector<Request>& messages = table_iter->second;
    messages.push_back(msg);
  }

  timeline_.NegotiateRankReady(name, msg.request_rank());

  std::vector<Request>& messages = table_iter->second;
  int count = (int)messages.size();
  bool ready_to_reduce = count == (size_ - joined_size);  //处于Join状态的rank不参与allreduce(等待状态)
  if (ready_to_reduce) {                                //Tensor可以进行allreduce，coordination过程结束
    timeline_.NegotiateEnd(name);
  }
  return ready_to_reduce;
}

void Controller::SetTimelineEnabled(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_pending_ = value;
  timeline_enabled_ = value;
}

void Controller::SetTimelineEnabledPending(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_pending_ = value;
}

void Controller::SetMarkCyclesInTimelinePending(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  mark_cycles_in_timeline_pending_ = value;
}

void Controller::SynchronizeTimelineEnabled() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_ = timeline_enabled_pending_;
}

bool Controller::TimeLineEnabled() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return timeline_enabled_;
}

bool Controller::TimelineEnabledPending() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return timeline_enabled_pending_;
}

bool Controller::MarkCyclesInTimelinePending() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return mark_cycles_in_timeline_pending_;
}
} // namespace common
} // namespace horovod
