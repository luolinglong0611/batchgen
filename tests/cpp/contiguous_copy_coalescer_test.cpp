#include "core/KV_Storage/contiguous_copy_coalescer.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using batchgen::kv::worker_detail::ContiguousCopyCoalescer;

static std::size_t Check(const std::vector<int>& pages, std::size_t tokens,
                         std::size_t token_bytes = 18, std::size_t gap = 0) {
    constexpr std::size_t page_tokens = 64;
    const auto physical_pages = *std::max_element(pages.begin(), pages.end()) + 1;
    const auto page_bytes = page_tokens * token_bytes;
    std::vector<std::byte> src(tokens * token_bytes);
    std::vector<std::byte> reference(physical_pages * (page_bytes + gap), std::byte{0x55});
    auto actual = reference;
    for (std::size_t i=0; i<src.size(); ++i) src[i] = std::byte((i*47 + 3) & 255);
    std::size_t copies=0;
    auto enqueue = [&](const std::byte* s, std::byte* d, std::size_t n) {
        ++copies; std::memcpy(d, s, n);
    };
    ContiguousCopyCoalescer merged(enqueue);
    std::size_t offset=0;
    for (int page : pages) {
        const auto count=std::min(page_tokens, tokens-offset);
        const auto dst=page*(page_bytes+gap);
        std::memcpy(reference.data()+dst,src.data()+offset*token_bytes,count*token_bytes);
        merged.Add(src.data()+offset*token_bytes, actual.data()+dst, count*token_bytes);
        offset += count;
        if (offset == tokens) break;
    }
    merged.Flush(); merged.Flush();
    assert(actual == reference);
    return copies;
}
int main() {
    std::vector<int> pages(128); std::iota(pages.begin(),pages.end(),0);
    assert(Check(pages,8192,1152)==1); // exact R1 8K x 576 BF16 geometry
    assert(Check(pages,8191,1152)==1); // partial last page
    assert(Check(pages,8192,576,64*4)==128); // page-split data region skips scales
    std::reverse(pages.begin(),pages.end()); assert(Check(pages,8192)==128);
    std::mt19937 rng(20261028);
    for (int i=0;i<30;++i) { std::shuffle(pages.begin(),pages.end(),rng); Check(pages,8183); }
    std::size_t count=0;
    ContiguousCopyCoalescer empty([&](const std::byte*,std::byte*,std::size_t){++count;});
    empty.Add(nullptr,nullptr,0); empty.Flush(); assert(count==0);
    std::cout << "coalescing exact bytes OK: ascending 128 pages -> 1 copy; fragmented/reversed/split layouts preserve bytes\n";
}
