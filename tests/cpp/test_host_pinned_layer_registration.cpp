// CPU-only: no CUDA, Torch, pinned memory, or model-size allocation.
#include "core/KV_Storage/host_pinned_layer_registration.h"
#include <array>
#include <cassert>
#include <iostream>

namespace hr = batchgen::host_registration;
using batchgen::kv::worker_detail::PlanPinnedLayerRanges;
constexpr std::size_t GiB = std::size_t{1} << 30;

template <typename Fn>
void expect_invalid(Fn&& fn) {
    bool failed = false;
    try { fn(); } catch (const std::invalid_argument&) { failed = true; }
    assert(failed);
}

int main() {
    // Production layout; arithmetic only, zero large allocation.
    constexpr std::size_t page = 64 * 576 * 2;
    constexpr std::size_t stride = 198637 * page;
    constexpr std::size_t bytes = 61 * stride;
    static_assert(stride == 14645108736ULL);
    static_assert(bytes == 893351632896ULL);
    static_assert(stride % 4096 == 0 && stride % (2 * 1024 * 1024) == 696320);
    static_assert((16 * GiB) % page == 65536); // A blind16GiB cut splits a KV page.
    auto ranges = PlanPinnedLayerRanges(bytes, stride, 16 * GiB, 4096);
    assert(ranges.size() == 61);
    for (std::size_t layer = 0; layer < 61; ++layer) {
        assert(ranges[layer].offset == layer * stride);
        assert(ranges[layer].bytes == stride && ranges[layer].bytes <= 16 * GiB);
        // Every physical page/token is contained in its one layer registration.
        const auto last_page = layer * stride + (198637 - 1) * page;
        assert(last_page + page == ranges[layer].offset + ranges[layer].bytes);
        const auto last_token = last_page + 63 * 576 * 2;
        assert(last_token + 576 * 2 <= ranges[layer].offset + ranges[layer].bytes);
    }
    hr::validate_plan(bytes, ranges);

    // Current large-pool budgets: one complete 38--42 GiB physical layer
    // must remain indivisible; chunk64 admits it while chunk32 rejects it.
    const std::array<std::size_t, 3> pools_gib = {2304, 2432, 2560};
    const std::array<std::size_t, 3> expected_strides = {
        40555708416ULL, 42808836096ULL, 45061890048ULL};
    for (std::size_t index = 0; index < pools_gib.size(); ++index) {
        const auto pages = pools_gib[index] * GiB / (61 * page);
        const auto layer_stride = pages * page;
        const auto pool_bytes = 61 * layer_stride;
        assert(layer_stride == expected_strides[index]);
        const auto large = PlanPinnedLayerRanges(pool_bytes, layer_stride, 64 * GiB, 4096);
        assert(large.size() == 61);
        for (std::size_t layer = 0; layer < 61; ++layer) {
            assert(large[layer].offset == layer * layer_stride);
            assert(large[layer].bytes == layer_stride);
        }
        hr::validate_plan(pool_bytes, large);
        expect_invalid([&] { PlanPinnedLayerRanges(pool_bytes, layer_stride, 32 * GiB, 4096); });
    }

    // Unset/zero keep one complete original range; no opt-in geometry constraint.
    for (const char* value : {static_cast<const char*>(nullptr), "0"}) {
        const auto one = PlanPinnedLayerRanges(bytes, 0, hr::chunk_bytes_from_env(value), 0);
        assert(one.size() == 1 && one[0].offset == 0 && one[0].bytes == bytes);
    }
    assert(hr::chunk_bytes_from_env("16") == 16 * GiB);
    expect_invalid([&] { PlanPinnedLayerRanges(bytes, stride, GiB, 4096); });
    expect_invalid([&] { PlanPinnedLayerRanges(bytes-1, stride, 16*GiB, 4096); });
    expect_invalid([&] { PlanPinnedLayerRanges(3*4100, 4100, 16384, 4096); });
    expect_invalid([&] { PlanPinnedLayerRanges(bytes, 0, 16*GiB, 4096); });
    expect_invalid([&] { hr::chunk_bytes_from_env("-16"); });

    // Group whole adjacent layers, never split one to hit a max byte count.
    const auto grouped = PlanPinnedLayerRanges(5*4096, 4096, 2*4096, 4096);
    assert(grouped.size() == 3 && grouped[0].bytes == 8192 && grouped[1].bytes == 8192 && grouped[2].bytes == 4096);

    alignas(4096) std::array<char, 3*4096> small{};
    const auto small_plan = PlanPinnedLayerRanges(small.size(),4096,4096,4096);
    for (int fail_at : {1,2,3,0}) {
        hr::Registry registry;
        std::vector<std::size_t> registered, removed;
        int attempts=0, rollback=-1, unmap_callbacks=0;
        bool failed=false;
        try {
            registry.add(small.data(),small.size(),small.size(),4096,true,7,
                [&](void* p,std::size_t n) {
                    assert(n==4096);
                    if (++attempts==fail_at) return 2;
                    registered.push_back(static_cast<char*>(p)-small.data());
                    return 0;
                },
                [&](int device,void* p) noexcept {
                    assert(device==7);removed.push_back(static_cast<char*>(p)-small.data());return 0;
                },rollback,&small_plan);
        } catch (const hr::RegistrationError& error) {
            failed=true;assert(error.code==2 && error.offset==std::size_t(fail_at-1)*4096 && error.bytes==4096);
        }
        assert(failed==(fail_at!=0) && rollback==0);
        if (failed) {
            assert(removed.size()==registered.size());
            for (std::size_t i=0;i<removed.size();++i) assert(removed[i]==registered[registered.size()-1-i]);
            const auto after=registry.release(small.data(),[](int,void*)noexcept{return 0;},[&](void*,std::size_t)noexcept{++unmap_callbacks;return 0;});
            assert(!after.found && unmap_callbacks==0);
        } else {
            const auto result=registry.release(small.data(),
                [&](int device,void* p)noexcept{assert(device==7);removed.push_back(static_cast<char*>(p)-small.data());return 0;},
                [&](void* p,std::size_t n)noexcept{assert(p==small.data()&&n==small.size());++unmap_callbacks;return 0;});
            assert(result.found && result.unregistered_ranges==3 && unmap_callbacks==1);
            assert((removed==std::vector<std::size_t>{8192,4096,0}));
            small[0]=42; // No-op unmap leaves backend-owned memory alive.
            assert(!registry.release(small.data(),[](int,void*)noexcept{return 0;},[](void*,std::size_t)noexcept{return 0;}).found);
        }
    }
    // Unregister failures are reported after attempting all successful ranges.
    hr::Registry registry;int rollback=0;std::size_t removed=0;
    registry.add(small.data(),small.size(),small.size(),4096,true,0,[](void*,std::size_t){return 0;},[](int,void*)noexcept{return 0;},rollback,&small_plan);
    const auto bad=registry.release(small.data(),[&](int,void*)noexcept{++removed;return removed==1?9:0;},[](void*,std::size_t)noexcept{return 0;});
    assert(removed==3 && bad.unregister_error==9);
    std::cout << "PASS: production61-layer chunk64 plans, defaults, grouping/rejection, whole-copy extents, reverse rollback/release, backend lifetime, error aggregation\n";
}
