#pragma once

// CPU-only bookkeeping for posix_shm.cpp. CUDA callbacks are deliberately
// injected so allocation failure, rollback and lifetime behavior can be tested
// without a GPU or a hundreds-of-GiB allocation.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace batchgen::host_registration {

inline std::size_t chunk_bytes_from_env(const char* value) {
    if (value == nullptr) return 0;
    const std::string text(value);
    constexpr std::size_t gib = std::size_t{1} << 30;
    const std::size_t limit = std::numeric_limits<std::size_t>::max() / gib;
    std::size_t number = 0;
    if (text.empty()) {
        throw std::invalid_argument("BATCHGEN_HOST_REGISTER_CHUNK_GIB must be an unsigned integer");
    }
    for (const char digit : text) {
        if (digit < '0' || digit > '9' ||
            number > limit / 10 ||
            (number == limit / 10 && std::size_t(digit - '0') > limit % 10)) {
            throw std::invalid_argument("BATCHGEN_HOST_REGISTER_CHUNK_GIB is invalid or overflows size_t");
        }
        number = number * 10 + std::size_t(digit - '0');
    }
    return number * gib;
}

struct Range {
    std::size_t offset;
    std::size_t bytes;
};

inline Range checked_tensor_extent(std::int64_t offset, std::int64_t bytes,
                                   const std::vector<std::int64_t>& shape,
                                   const std::string& dtype, std::int64_t allocation_bytes) {
    if (offset < 0 || bytes <= 0 || offset > allocation_bytes || bytes > allocation_bytes - offset) {
        throw std::invalid_argument("invalid tensor offset or byte size");
    }
    // Native HtoD uses byte_size, whereas torch.from_blob uses dtype/shape.
    // They must describe the same extent to protect both copy paths.
    std::int64_t expected = 0;
    if (dtype == "float8_e4m3fn" || dtype == "uint8" || dtype == "int8") expected = 1;
    else if (dtype == "bfloat16" || dtype == "float16" || dtype == "int16") expected = 2;
    else if (dtype == "float32" || dtype == "int32") expected = 4;
    else if (dtype == "float64" || dtype == "int64") expected = 8;
    else throw std::invalid_argument("unsupported registration tensor dtype: " + dtype);
    for (const std::int64_t dimension : shape) {
        if (dimension <= 0 || dimension > bytes / expected) {
            throw std::invalid_argument("invalid or overflowing registration tensor shape");
        }
        expected *= dimension;
    }
    if (expected != bytes) throw std::invalid_argument("tensor byte size disagrees with shape/dtype");
    return {static_cast<std::size_t>(offset), static_cast<std::size_t>(bytes)};
}

inline std::vector<Range> plan(std::size_t bytes, std::size_t chunk_bytes) {
    if (bytes == 0) throw std::invalid_argument("cannot register an empty mapping");
    const std::size_t step = chunk_bytes == 0 ? bytes : std::min(bytes, chunk_bytes);
    std::vector<Range> ranges;
    ranges.reserve(bytes / step + (bytes % step != 0));
    for (std::size_t offset = 0; offset < bytes;) {
        const std::size_t length = std::min(step, bytes - offset);
        ranges.push_back({offset, length});
        offset += length;
    }
    return ranges;
}

inline void validate_plan(std::size_t bytes, const std::vector<Range>& ranges) {
    std::size_t covered = 0;
    for (const Range& range : ranges) {
        if (range.offset != covered || range.bytes == 0 || range.bytes > bytes - covered) {
            throw std::invalid_argument("registration plan has a gap, overlap, or invalid extent");
        }
        covered += range.bytes;
    }
    if (covered != bytes) throw std::invalid_argument("registration plan does not cover mapping");
}

// CUDA copies cannot necessarily cross separately registered host ranges. Pick
// page-aligned cuts which never split an original tensor; then both native
// per-tensor DMA and PyTorch Tensor.to keep their existing copy semantics.
inline std::vector<Range> tensor_aware_plan(
    std::size_t bytes, std::size_t chunk_bytes, std::size_t page_bytes,
    std::vector<Range> tensors) {
    if (bytes == 0 || chunk_bytes == 0 || page_bytes == 0 ||
        bytes % page_bytes != 0 || chunk_bytes % page_bytes != 0 || tensors.empty()) {
        throw std::invalid_argument("tensor-aware registration requires page-aligned sizes and tensor metadata");
    }
    std::sort(tensors.begin(), tensors.end(), [](const Range& a, const Range& b) {
        return a.offset < b.offset;
    });
    std::size_t previous_end = 0;
    for (const Range& tensor : tensors) {
        if (tensor.bytes == 0 || tensor.offset > bytes || tensor.bytes > bytes - tensor.offset) {
            throw std::invalid_argument("registration tensor exceeds mapping or is empty");
        }
        if (tensor.offset < previous_end) throw std::invalid_argument("overlapping registration tensor metadata");
        if (tensor.bytes > chunk_bytes) throw std::invalid_argument("tensor exceeds maximum registration chunk size");
        previous_end = tensor.offset + tensor.bytes;
    }

    std::vector<Range> ranges;
    for (std::size_t begin = 0; begin < bytes;) {
        std::size_t cut = begin + std::min(chunk_bytes, bytes - begin);
        while (cut > begin) {
            auto it = std::upper_bound(tensors.begin(), tensors.end(), cut,
                [](std::size_t offset, const Range& tensor) { return offset < tensor.offset; });
            if (it == tensors.begin()) break;
            --it;
            if (!(it->offset < cut && cut < it->offset + it->bytes)) break;
            // Every page boundary between floor(tensor.start) and the current
            // interior cut is forbidden. Jumping backwards skips no safe cut;
            // repeating also handles adjacent tensors with unaligned edges.
            cut = (it->offset / page_bytes) * page_bytes;
        }
        if (cut <= begin) {
            throw std::invalid_argument("no tensor-safe page boundary within maximum registration chunk");
        }
        ranges.push_back({begin, cut - begin});
        begin = cut;
    }
    validate_plan(bytes, ranges);
    // Independent final membership assertion: all tensors must fit wholly in
    // one selected range. Padding is covered too, including the unused tail.
    std::size_t index = 0;
    for (const Range& tensor : tensors) {
        while (index < ranges.size() && tensor.offset >= ranges[index].offset + ranges[index].bytes) ++index;
        if (index == ranges.size() || tensor.offset < ranges[index].offset ||
            tensor.offset + tensor.bytes > ranges[index].offset + ranges[index].bytes) {
            throw std::logic_error("registration planner split a tensor");
        }
    }
    return ranges;
}

class RegistrationError : public std::runtime_error {
public:
    RegistrationError(std::size_t offset, std::size_t bytes, int code)
        : std::runtime_error("cudaHostRegister failed at offset " +
                             std::to_string(offset) + " for " +
                             std::to_string(bytes) + " bytes (CUDA error " +
                             std::to_string(code) + ")"),
          offset(offset), bytes(bytes), code(code) {}
    std::size_t offset;
    std::size_t bytes;
    int code;
};

struct ReleaseResult {
    bool found = false;
    std::size_t mapped_bytes = 0;
    std::size_t unregistered_ranges = 0;
    int unregister_error = 0;
    int unmap_error = 0;
};

class Registry {
    struct Mapping {
        std::size_t mapped_bytes;
        int device;
        std::vector<Range> ranges;
        std::size_t registered = 0;
    };
    std::mutex mutex_;
    std::unordered_map<void*, Mapping> mappings_;

public:
    // register_fn(address, bytes) returns 0 on success. unregister_fn(device,
    // address) and unmap_fn(address, bytes) MUST NOT throw. No container grows
    // after the first successful CUDA call. Mutex coverage intentionally spans
    // registration/release: a second instance cannot race a partially built
    // record, and an address cannot be reused before its old record is erased.
    template <class Register, class Unregister>
    std::size_t add(void* base, std::size_t mapped_bytes, std::size_t bytes,
                    std::size_t chunk_bytes, bool pin, int device,
                    Register&& register_fn, Unregister&& unregister_fn,
                    int& rollback_error, const std::vector<Range>* planned_ranges = nullptr) {
        rollback_error = 0;
        if (base == nullptr || bytes == 0 || mapped_bytes < bytes) {
            throw std::invalid_argument("invalid host registration mapping");
        }
        Mapping mapping{mapped_bytes, device,
            pin ? (planned_ranges ? *planned_ranges : plan(bytes, chunk_bytes))
                : std::vector<Range>{}, 0};
        if (pin) validate_plan(bytes, mapping.ranges);
        std::lock_guard<std::mutex> lock(mutex_);
        auto inserted = mappings_.emplace(base, std::move(mapping));
        if (!inserted.second) throw std::logic_error("host mapping already tracked");
        Mapping& entry = inserted.first->second;
        try {
            for (const Range& range : entry.ranges) {
                const int error = register_fn(static_cast<char*>(base) + range.offset,
                                              range.bytes);
                if (error != 0) throw RegistrationError(range.offset, range.bytes, error);
                ++entry.registered;
            }
        } catch (...) {
            while (entry.registered != 0) {
                const Range& range = entry.ranges[--entry.registered];
                const int error = unregister_fn(entry.device,
                    static_cast<char*>(base) + range.offset);
                if (rollback_error == 0) rollback_error = error;
            }
            mappings_.erase(inserted.first);
            throw;
        }
        return entry.registered;
    }

    template <class Unregister, class Unmap>
    ReleaseResult release(void* base, Unregister&& unregister_fn, Unmap&& unmap_fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        ReleaseResult result;
        const auto it = mappings_.find(base);
        if (it == mappings_.end()) return result;
        Mapping& entry = it->second;
        result.found = true;
        result.mapped_bytes = entry.mapped_bytes;
        while (entry.registered != 0) {
            const Range& range = entry.ranges[--entry.registered];
            const int error = unregister_fn(entry.device,
                static_cast<char*>(base) + range.offset);
            ++result.unregistered_ranges;
            if (result.unregister_error == 0) result.unregister_error = error;
        }
        result.unmap_error = unmap_fn(base, entry.mapped_bytes);
        mappings_.erase(it);
        return result;
    }
};

}  // namespace batchgen::host_registration
