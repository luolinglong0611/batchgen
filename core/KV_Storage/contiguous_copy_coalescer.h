#ifndef BATCHGEN_CONTIGUOUS_COPY_COALESCER_H_
#define BATCHGEN_CONTIGUOUS_COPY_COALESCER_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace batchgen::kv::worker_detail {

// Merge only adjacent source AND destination byte ranges. In particular,
// fragmented, reversed and page-split destinations keep their original copies.
// Call Flush explicitly before recording completion on the owning copy stream.
// No allocation, host scatter, page remapping or asynchronous ownership changes.
template <typename Enqueue>
class ContiguousCopyCoalescer {
   public:
    explicit ContiguousCopyCoalescer(Enqueue enqueue)
        : enqueue_(std::move(enqueue)) {}

    void Add(const std::byte* src, std::byte* dst, std::size_t bytes) {
        if (bytes == 0) return;
        const auto max = std::numeric_limits<std::uintptr_t>::max();
        const auto source = reinterpret_cast<std::uintptr_t>(src_);
        const auto destination = reinterpret_cast<std::uintptr_t>(dst_);
        if (bytes_ != 0 && bytes_ <= max - source &&
            bytes_ <= max - destination &&
            reinterpret_cast<std::uintptr_t>(src) == source + bytes_ &&
            reinterpret_cast<std::uintptr_t>(dst) == destination + bytes_ &&
            bytes <= std::numeric_limits<std::size_t>::max() - bytes_) {
            bytes_ += bytes;
            return;
        }
        Flush();
        src_ = src;
        dst_ = dst;
        bytes_ = bytes;
    }

    void Flush() {
        if (bytes_ != 0) {
            enqueue_(src_, dst_, bytes_);
            bytes_ = 0;
        }
    }

   private:
    Enqueue enqueue_;
    const std::byte* src_ = nullptr;
    std::byte* dst_ = nullptr;
    std::size_t bytes_ = 0;
};

}  // namespace batchgen::kv::worker_detail
#endif
