// CPU-only test; no CUDA/toolkit, PyTorch, Linux mmap, or large allocation needed.
// From repo root:
// c++ -std=c++17 -pthread -I. tests/cpp/test_host_registration.cpp -o /tmp/test_host_registration
// /tmp/test_host_registration
#include "core/Parameter_Server/host_registration.h"

#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <thread>

namespace hr = batchgen::host_registration;

struct Event {
    char kind;
    void* address;
    std::size_t bytes;
    int device;
};

struct FakeCuda {
    std::vector<Event> events;
    int attempts = 0;
    int fail_attempt = -1;
    bool throw_instead = false;
    int unregister_error = 0;

    FakeCuda() { events.reserve(4096); }
    int add(void* address, std::size_t bytes) {
        events.push_back({'r', address, bytes, -1});
        if (++attempts == fail_attempt) {
            if (throw_instead) throw std::runtime_error("injected register exception");
            return 2;
        }
        return 0;
    }
    int remove(int device, void* address) noexcept {
        events.push_back({'u', address, 0, device});
        return unregister_error;
    }
    int unmap(void* address, std::size_t bytes) noexcept {
        events.push_back({'m', address, bytes, -1});
        return 0;
    }
};

std::size_t add(hr::Registry& registry, FakeCuda& cuda, void* base,
                std::size_t mapped, std::size_t bytes, std::size_t chunk,
                bool pin, int device, int& rollback) {
    return registry.add(base, mapped, bytes, chunk, pin, device,
        [&](void* p, std::size_t n) { return cuda.add(p, n); },
        [&](int d, void* p) noexcept { return cuda.remove(d, p); }, rollback);
}

hr::ReleaseResult release(hr::Registry& registry, FakeCuda& cuda, void* base) {
    return registry.release(base,
        [&](int d, void* p) noexcept { return cuda.remove(d, p); },
        [&](void* p, std::size_t n) noexcept { return cuda.unmap(p, n); });
}

void test_env_and_plan() {
    constexpr std::size_t gib = std::size_t{1} << 30;
    assert(hr::chunk_bytes_from_env(nullptr) == 0);
    assert(hr::chunk_bytes_from_env("0") == 0);
    assert(hr::chunk_bytes_from_env("16") == 16 * gib);
    assert(hr::chunk_bytes_from_env("00016") == 16 * gib);
    for (const char* invalid : {"", "-1", "+1", " 16", "16 ", "1.5", "16GiB",
                                "999999999999999999999999999999"}) {
        bool rejected = false;
        try { hr::chunk_bytes_from_env(invalid); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    const auto maximum = std::numeric_limits<std::size_t>::max() / gib;
    assert(hr::chunk_bytes_from_env(std::to_string(maximum).c_str()) == maximum * gib);
    bool rejected = false;
    try { hr::chunk_bytes_from_env(std::to_string(maximum + 1).c_str()); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    const auto ranges = hr::plan(675 * gib, 16 * gib);
    assert(ranges.size() == 43 && ranges.back().bytes == 3 * gib);
    std::size_t covered = 0;
    for (const auto& range : ranges) {
        assert(range.offset == covered);
        assert(range.offset % 4096 == 0);
        assert(range.bytes <= 16 * gib);
        covered += range.bytes;
    }
    assert(covered == 675 * gib);
    assert(hr::plan(675 * gib, 0).size() == 1);
    assert(hr::plan(5, 10).front().bytes == 5);
    assert(hr::plan(32, 16).size() == 2);
}

void test_default_and_owner() {
    hr::Registry registry;
    FakeCuda cuda;
    std::array<char, 128> worker{}, owner{};
    int rollback = -1;
    assert(add(registry, cuda, worker.data(), 128, 100, 0, true, 3, rollback) == 1);
    assert(rollback == 0);
    assert(cuda.events.size() == 1 && cuda.events[0].bytes == 100);
    assert(add(registry, cuda, owner.data(), 128, 100, 16, false, -1, rollback) == 0);
    auto result = release(registry, cuda, owner.data());
    assert(result.found && result.unregistered_ranges == 0);
    assert(cuda.events.size() == 2 && cuda.events.back().kind == 'm');
    result = release(registry, cuda, worker.data());
    assert(result.found && result.unregistered_ranges == 1 && result.mapped_bytes == 128);
    assert(cuda.events[2].kind == 'u' && cuda.events[2].device == 3);
    assert(cuda.events[3].kind == 'm' && cuda.events[3].bytes == 128);
    const auto count = cuda.events.size();
    assert(!release(registry, cuda, worker.data()).found);
    assert(cuda.events.size() == count);
}

void test_tensor_aware_plan() {
    // Proposed cut 64 lands in [61,80); floor(61/8)*8=56 then lands
    // in adjacent [53,61). The planner must retreat again, to safe cut 48.
    auto ranges = hr::tensor_aware_plan(128, 64, 8,
                                      {{61, 19}, {53, 8}, {0, 16}, {96, 8}});
    assert(ranges.size() == 3);
    assert(ranges[0].offset == 0 && ranges[0].bytes == 48);
    assert(ranges[1].offset == 48 && ranges[1].bytes == 64);
    assert(ranges[2].offset == 112 && ranges[2].bytes == 16);

    // A genuine large tensor straddles the naive 16 GiB boundary, but a safe
    // earlier cut keeps the entire tensor in the next registration.
    constexpr std::size_t gib = std::size_t{1} << 30;
    ranges = hr::tensor_aware_plan(40 * gib, 16 * gib, 4096,
                                  {{15 * gib, 4 * gib}, {0, gib}});
    assert(ranges.size() == 3);
    assert(ranges[0].bytes == 15 * gib);
    assert(ranges[1].offset == 15 * gib && ranges[1].bytes == 16 * gib);
    assert(ranges[2].offset == 31 * gib && ranges[2].bytes == 9 * gib);

    const std::vector<std::vector<hr::Range>> invalid{
        {}, {{0, 0}}, {{120, 9}}, {{129, 1}}, {{0, 65}},
        {{0, 16}, {8, 16}}, {{0, 16}, {0, 16}},
        // Each tensor is <=64 bytes, but unaligned adjacent tensors leave no
        // page-aligned cut between 0 and 64. Must reject rather than split one.
        {{0, 61}, {61, 19}}
    };
    for (const auto& tensors : invalid) {
        bool rejected = false;
        try { hr::tensor_aware_plan(128, 64, 8, tensors); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    for (const auto& values : std::vector<std::array<std::size_t, 3>>{
             {127, 64, 8}, {128, 63, 8}, {128, 0, 8}, {128, 64, 0}}) {
        bool rejected = false;
        try { hr::tensor_aware_plan(values[0], values[1], values[2], {{0, 1}}); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
}

void test_metadata_shapes_and_bounds() {
    const auto checked = hr::checked_tensor_extent(8, 24, {3, 4}, "bfloat16", 128);
    assert(checked.offset == 8 && checked.bytes == 24);
    assert(hr::checked_tensor_extent(0, 4, {}, "float32", 128).bytes == 4);
    struct Invalid {
        std::int64_t offset, bytes;
        std::vector<std::int64_t> shape;
        std::string dtype;
        std::int64_t total;
    };
    for (const auto& item : std::vector<Invalid>{
             {-1, 4, {4}, "uint8", 128}, {0, -1, {4}, "uint8", 128},
             {126, 4, {4}, "uint8", 128}, {129, 1, {1}, "uint8", 128},
             {0, 4, {0}, "float32", 128}, {0, 4, {-1}, "float32", 128},
             {0, 4, {2, 2}, "bfloat16", 128}, {0, 4, {1}, "unknown", 128},
             {0, 4, {std::numeric_limits<std::int64_t>::max()}, "float64", 128},
             {0, 4, {}, "float64", 128}, {0, 4, {1}, "float32", -1}}) {
        bool rejected = false;
        try { hr::checked_tensor_extent(item.offset, item.bytes, item.shape, item.dtype, item.total); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
}

void test_tensor_aware_against_page_grid_oracle() {
    std::mt19937 random(20261023);
    constexpr std::size_t bytes = 256, chunk = 64, page = 8;
    for (int trial = 0; trial < 1000; ++trial) {
        std::vector<hr::Range> tensors;
        std::size_t position = random() % 9;
        while (position < bytes) {
            const auto length = std::min<std::size_t>(1 + random() % 70, bytes - position);
            tensors.push_back({position, length});
            position += length + random() % 13;
        }
        // Independent exhaustive page-grid oracle; no interval jumping.
        std::vector<hr::Range> expected;
        bool possible = true;
        for (std::size_t begin = 0; begin < bytes;) {
            std::size_t best = begin;
            for (std::size_t candidate = begin + page;
                 candidate <= std::min(bytes, begin + chunk); candidate += page) {
                bool safe = true;
                for (const auto& tensor : tensors) {
                    if (tensor.offset < candidate && candidate < tensor.offset + tensor.bytes) safe = false;
                }
                if (safe) best = candidate;
            }
            if (best == begin) { possible = false; break; }
            expected.push_back({begin, best - begin});
            begin = best;
        }
        try {
            const auto actual = hr::tensor_aware_plan(bytes, chunk, page, tensors);
            assert(possible && actual.size() == expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                assert(actual[i].offset == expected[i].offset && actual[i].bytes == expected[i].bytes);
            }
        } catch (const std::invalid_argument&) { assert(!possible); }
    }
}

void test_planned_registration_lifetime() {
    hr::Registry registry;
    FakeCuda cuda;
    std::array<char, 128> memory{};
    const auto ranges = hr::tensor_aware_plan(128, 64, 8, {{53, 8}, {61, 19}});
    int rollback;
    const auto registered = registry.add(memory.data(), 128, 128, 64, true, 2,
        [&](void* p, std::size_t n) { return cuda.add(p, n); },
        [&](int d, void* p) noexcept { return cuda.remove(d, p); }, rollback, &ranges);
    assert(registered == 3 && cuda.events[0].bytes == 48);
    assert(release(registry, cuda, memory.data()).unregistered_ranges == 3);
    assert(cuda.events[3].address == memory.data() + 112);
    assert(cuda.events[4].address == memory.data() + 48);
    assert(cuda.events[5].address == memory.data());

    const auto count = cuda.events.size();
    const std::vector<hr::Range> invalid{{0, 48}, {56, 72}};
    bool rejected = false;
    try {
        registry.add(memory.data(), 128, 128, 64, true, 2,
            [&](void* p, std::size_t n) { return cuda.add(p, n); },
            [&](int d, void* p) noexcept { return cuda.remove(d, p); }, rollback, &invalid);
    } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected && cuda.events.size() == count);
}

void test_reverse_release_and_duplicate() {
    hr::Registry registry;
    FakeCuda cuda;
    std::array<char, 128> first{}, second{};
    int rollback;
    assert(add(registry, cuda, first.data(), 128, 100, 32, true, 0, rollback) == 4);
    assert(add(registry, cuda, second.data(), 128, 64, 32, true, 7, rollback) == 2);
    const auto before = cuda.events.size();
    bool rejected = false;
    try { add(registry, cuda, first.data(), 128, 100, 32, true, 0, rollback); }
    catch (const std::logic_error&) { rejected = true; }
    assert(rejected && before == cuda.events.size());
    const auto second_result = release(registry, cuda, second.data());
    assert(second_result.unregistered_ranges == 2);
    assert(cuda.events[before].address == second.data() + 32);
    assert(cuda.events[before + 1].address == second.data());
    cuda.unregister_error = 17;  // Continue attempting all other ranges on error.
    const auto first_result = release(registry, cuda, first.data());
    assert(first_result.unregistered_ranges == 4 && first_result.unregister_error == 17);
    const auto start = before + 3;
    for (int i = 0; i < 4; ++i) {
        assert(cuda.events[start + i].address == first.data() + (3 - i) * 32);
    }
    assert(cuda.events.back().kind == 'm');
}

void test_failure_rollback() {
    for (const bool throws : {false, true}) {
        for (const int failure : {1, 2, 4}) {
            hr::Registry registry;
            FakeCuda cuda;
            cuda.fail_attempt = failure;
            cuda.throw_instead = throws;
            cuda.unregister_error = 29;
            std::array<char, 128> memory{};
            int rollback = 0;
            bool rejected = false;
            try { add(registry, cuda, memory.data(), 128, 100, 32, true, 5, rollback); }
            catch (const hr::RegistrationError& error) {
                assert(!throws && error.code == 2);
                assert(error.offset == std::size_t(failure - 1) * 32);
                rejected = true;
            } catch (const std::runtime_error&) { assert(throws); rejected = true; }
            assert(rejected);
            assert(rollback == (failure == 1 ? 0 : 29));
            assert(cuda.events.size() == std::size_t(2 * failure - 1));
            for (int i = 0; i < failure - 1; ++i) {
                const auto& event = cuda.events[failure + i];
                assert(event.kind == 'u' && event.device == 5);
                assert(event.address == memory.data() + (failure - 2 - i) * 32);
            }
            // Failure forgets the record; allocator owns mmap/fd rollback.
            assert(!release(registry, cuda, memory.data()).found);
            cuda.fail_attempt = -1;
            assert(add(registry, cuda, memory.data(), 128, 100, 32, true, 5, rollback) == 4);
            assert(release(registry, cuda, memory.data()).found);
        }
    }
}

void test_concurrent_instances() {
    hr::Registry registry;
    FakeCuda cuda;
    std::array<std::array<char, 128>, 32> memory{};
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < memory.size(); ++i) {
        threads.emplace_back([&, i] {
            int rollback;
            assert(add(registry, cuda, memory[i].data(), 128, 100, 32, true,
                       static_cast<int>(i % 8), rollback) == 4);
            const auto result = release(registry, cuda, memory[i].data());
            assert(result.found && result.unregistered_ranges == 4);
        });
    }
    for (auto& thread : threads) thread.join();
    assert(cuda.events.size() == memory.size() * 9);
    for (auto& allocation : memory) assert(!release(registry, cuda, allocation.data()).found);
}

int main() {
    test_env_and_plan();
    test_default_and_owner();
    test_tensor_aware_plan();
    test_metadata_shapes_and_bounds();
    test_tensor_aware_against_page_grid_oracle();
    test_planned_registration_lifetime();
    test_reverse_release_and_duplicate();
    test_failure_rollback();
    test_concurrent_instances();
    std::cout << "host registration CPU contract tests passed\n";
}
