// Isolated CUDA qualification of the production registration planner/registry.
// 64 MiB host / 16 MiB device only; no model, weights, or performance measurement.
#include "core/Parameter_Server/host_registration.h"
#include <cuda_runtime_api.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace hr = batchgen::host_registration;
constexpr std::size_t MiB = 1024 * 1024;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(bool value) {
    if (!value) throw std::runtime_error("registration qualification assertion failed");
}

int main() {
    try {
        check(cudaSetDevice(0));
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, 0));
        require(props.major == 10 && props.minor == 3);
        constexpr std::size_t bytes = 64 * MiB, chunk = 16 * MiB;
        const std::vector<hr::Range> tensors = {
            {0,12*MiB},{12*MiB,10*MiB},{22*MiB,10*MiB},{32*MiB,16*MiB},{48*MiB,16*MiB}};
        const auto ranges = hr::tensor_aware_plan(bytes, chunk, sysconf(_SC_PAGESIZE), tensors);
        require(ranges.size() == 5 && ranges[0].bytes == 12*MiB);
        unsigned long long copied = 0;
        for (int trial = 0; trial < 3; ++trial) {
            auto* host = static_cast<unsigned char*>(mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
                MAP_PRIVATE|MAP_ANONYMOUS, -1, 0));
            require(host != MAP_FAILED);
            hr::Registry registry;
            auto unregister_fn = [](int device, void* address) noexcept {
                const auto error = cudaSetDevice(device);
                if (error != cudaSuccess) return static_cast<int>(error);
                return static_cast<int>(cudaHostUnregister(address));
            };
            auto unmap_fn = [](void* address, std::size_t size) noexcept { return munmap(address,size); };
            void* device = nullptr;
            bool tracked = false;
            try {
                for (std::size_t i = 0; i < bytes; ++i) host[i] = static_cast<unsigned char>((i+trial*7)%251);
                int rollback = -1;
                const auto count = registry.add(host, bytes, bytes, chunk, true, 0,
                    [](void* address, std::size_t size) {
                        return static_cast<int>(cudaHostRegister(address,size,cudaHostRegisterDefault));
                    }, unregister_fn, rollback, &ranges);
                tracked = true;
                require(count == ranges.size() && rollback == 0);
                check(cudaMalloc(&device, chunk));
                std::vector<unsigned char> copy(chunk);
                cudaStream_t stream;
                check(cudaStreamCreate(&stream));
                for (const auto& tensor : tensors) {
                    check(cudaMemcpyAsync(device,host+tensor.offset,tensor.bytes,cudaMemcpyHostToDevice,stream));
                    check(cudaStreamSynchronize(stream));
                    check(cudaMemcpy(copy.data(),device,tensor.bytes,cudaMemcpyDeviceToHost));
                    require(std::memcmp(copy.data(),host+tensor.offset,tensor.bytes) == 0);
                    copied += tensor.bytes;
                }
                check(cudaStreamDestroy(stream));
                check(cudaFree(device));
                device = nullptr;
                const auto released = registry.release(host,unregister_fn,unmap_fn);
                tracked = false;
                require(released.found && released.unregistered_ranges == ranges.size()
                    && released.unregister_error == 0 && released.unmap_error == 0);
                require(!registry.release(host,unregister_fn,unmap_fn).found);
            } catch (...) {
                if (device) cudaFree(device);
                if (tracked) registry.release(host,unregister_fn,unmap_fn);
                else munmap(host,bytes);
                throw;
            }
        }
        std::cout << "{\"status\":\"passed_small_CUDA_registration_and_tensor_DMA_NOT_full_mapping_or_model\","
                  << "\"trials\":3,\"registration_ranges_per_trial\":5,\"tensor_bytes_checked\":" << copied
                  << ",\"host_bytes\":" << bytes << ",\"GPU_bytes\":" << chunk
                  << ",\"full_model_qualified\":false,\"formal_test_permitted\":false}\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
