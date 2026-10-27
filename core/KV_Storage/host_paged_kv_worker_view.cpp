#include "host_paged_kv_worker_view.h"
#include "host_pinned_layer_registration.h"

#include <cuda_runtime_api.h>

#include <functional>
#include <cstdlib>
#include <unistd.h>
#include <sstream>
#include <stdexcept>

namespace batchgen::kv::worker_detail {

namespace {

host_registration::Registry& PinnedKVRegistry() {
    // Track registrations only. HostPagedKVBackend remains the sole mapping owner.
    static auto* registry = new host_registration::Registry();
    return *registry;
}

int UnregisterKVRange(int device, void* address) noexcept {
    const cudaError_t selected = cudaSetDevice(device);
    if (selected != cudaSuccess) return static_cast<int>(selected);
    return static_cast<int>(cudaHostUnregister(address));
}

std::string BuildCudaErrorMessage(const char* op, cudaError_t error) {
    std::ostringstream oss;
    oss << op << " failed with error " << static_cast<int>(error) << " ("
        << cudaGetErrorString(error) << ")";
    return oss.str();
}

template <typename Callable, typename ErrorLogger>
void InvokeCudaChecked(const char* op,
                       const std::shared_ptr<spdlog::logger>& logger,
                       Callable&& callable, ErrorLogger&& error_logger) {
    const cudaError_t result = std::invoke(std::forward<Callable>(callable));
    if (result == cudaSuccess) {
        return;
    }
    const std::string message = BuildCudaErrorMessage(op, result);
    std::invoke(std::forward<ErrorLogger>(error_logger), message);
    throw std::runtime_error(message);
}

template <typename Callable>
void InvokeCudaChecked(const char* op,
                       const std::shared_ptr<spdlog::logger>& logger,
                       Callable&& callable) {
    InvokeCudaChecked(
        op, logger, std::forward<Callable>(callable),
        [&](const std::string& message) { logger->error("{}", message); });
}

}  // namespace

void RegisterPinnedRange(void* base, std::size_t bytes, int device_index,
                         const std::shared_ptr<spdlog::logger>& logger,
                         std::size_t layer_stride_bytes) {
    if (base == nullptr || bytes == 0) {
        return;
    }
    if (device_index < 0) {
        throw std::invalid_argument(
            "device_index must be greater than or equal to zero");
    }
    const std::size_t chunk_bytes = host_registration::chunk_bytes_from_env(
        std::getenv("BATCHGEN_HOST_REGISTER_CHUNK_GIB"));
    std::vector<host_registration::Range> ranges;
    if (chunk_bytes != 0) {
        const long system_page_bytes = sysconf(_SC_PAGESIZE);
        if (system_page_bytes <= 0) {
            throw std::runtime_error("sysconf(_SC_PAGESIZE) failed for Host KV registration");
        }
        if (reinterpret_cast<std::uintptr_t>(base) %
                static_cast<std::size_t>(system_page_bytes) != 0) {
            throw std::invalid_argument("Host KV data base must be page aligned");
        }
        ranges = PlanPinnedLayerRanges(bytes, layer_stride_bytes, chunk_bytes,
                                       static_cast<std::size_t>(system_page_bytes));
    }
    int device_count = 0;
    InvokeCudaChecked("cudaGetDeviceCount", logger,
                      [&]() { return cudaGetDeviceCount(&device_count); });
    if (device_index >= device_count) {
        throw std::out_of_range("device_index " + std::to_string(device_index) +
                                " exceeds available devices " +
                                std::to_string(device_count));
    }
    InvokeCudaChecked(
        "cudaSetDevice", logger, [&]() { return cudaSetDevice(device_index); },
        [&](const std::string& message) {
            logger->error("{} (device_index={})", message, device_index);
        });
    
    // Preserve the exact one-call default. Opt-in chunks use complete physical
    // layers, not blind GiB cuts which can split a page or a DMA extent.
    int rollback_error = 0;
    bool tracked = false;
    try {
        const std::size_t count = PinnedKVRegistry().add(
            base, bytes, bytes, chunk_bytes, true, device_index,
            [](void* address, std::size_t length) noexcept {
                return static_cast<int>(cudaHostRegister(
                    address, length, cudaHostRegisterDefault));
            },
            UnregisterKVRange, rollback_error,
            chunk_bytes == 0 ? nullptr : &ranges);
        tracked = true;
        logger->info(
            "Host KV CUDA registration completed: {} ranges, bytes={}, "
            "layer_stride_bytes={}, chunk_bytes={}, device={}",
            count, bytes, layer_stride_bytes, chunk_bytes, device_index);
    } catch (...) {
        if (tracked) {
            const auto released = PinnedKVRegistry().release(
                base, UnregisterKVRange,
                [](void*, std::size_t) noexcept { return 0; });
            if (rollback_error == 0) rollback_error = released.unregister_error;
        }
        if (rollback_error != 0) {
            logger->error("Host KV registration rollback error={}", rollback_error);
        }
        throw;
    }
}

void UnregisterPinnedRange(void* base, int device_index,
                           const std::shared_ptr<spdlog::logger>& logger) {
    if (base == nullptr) {
        return;
    }
    if (device_index < 0) {
        throw std::invalid_argument(
            "device_index must be greater than or equal to zero");
    }
    InvokeCudaChecked(
        "cudaSetDevice", logger, [&]() { return cudaSetDevice(device_index); },
        [&](const std::string& message) {
            logger->error("{} (device_index={})", message, device_index);
        });
    // release attempts every successful range in reverse order. The no-op
    // unmap callback is essential: the backend, not this registry, owns mmap.
    const auto released = PinnedKVRegistry().release(
        base, UnregisterKVRange,
        [](void*, std::size_t) noexcept { return 0; });
    if (!released.found) {
        throw std::logic_error("Host KV registered mapping is not tracked");
    }
    InvokeCudaChecked(
        "cudaHostUnregister", logger,
        [&]() { return static_cast<cudaError_t>(released.unregister_error); },
        [&](const std::string& message) {
            logger->error("{} (ptr={}, ranges={})", message, base,
                          released.unregistered_ranges);
        });
}

}  // namespace batchgen::kv::worker_detail
