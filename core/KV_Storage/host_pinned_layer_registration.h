#pragma once

// Host KV transfers address a single physical layer/page or token. Treat the
// whole physical layer, including its padding, as an indivisible registration
// extent so no existing DMA or UVA item crosses separately registered ranges.
#include "../Parameter_Server/host_registration.h"

namespace batchgen::kv::worker_detail {

inline std::vector<host_registration::Range> PlanPinnedLayerRanges(
    std::size_t bytes, std::size_t layer_stride_bytes,
    std::size_t chunk_bytes, std::size_t system_page_bytes) {
    if (chunk_bytes == 0) {
        return host_registration::plan(bytes, 0);
    }
    if (bytes == 0 || layer_stride_bytes == 0 || system_page_bytes == 0 ||
        bytes % layer_stride_bytes != 0 ||
        layer_stride_bytes % system_page_bytes != 0) {
        throw std::invalid_argument(
            "chunked Host KV registration requires complete page-aligned physical layers");
    }
    std::vector<host_registration::Range> layers;
    layers.reserve(bytes / layer_stride_bytes);
    for (std::size_t offset = 0; offset < bytes; offset += layer_stride_bytes) {
        layers.push_back({offset, layer_stride_bytes});
    }
    return host_registration::tensor_aware_plan(
        bytes, chunk_bytes, system_page_bytes, std::move(layers));
}

}  // namespace batchgen::kv::worker_detail
