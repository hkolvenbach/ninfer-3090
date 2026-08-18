#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/pinned_transfer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

struct PlannedPagedCache {
    ninfer::PagedKVPoolLayout layout;
    std::size_t bytes = 0;
};

struct PlannedCyclicCache {
    ninfer::CyclicKVCacheLayout layout;
    std::size_t bytes = 0;
};

PlannedCyclicCache plan_cyclic_cache(std::int32_t lanes) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_cyclic_kv_cache(builder, 2, 129, 2, 3, lanes);
    return PlannedCyclicCache{std::move(layout), builder.finish(256)};
}

PlannedPagedCache
plan_paged_cache(std::uint32_t pages, std::uint32_t logical_pages, std::int32_t rows,
                 std::vector<ninfer::PagedKVPlaneSpec> planes,
                 ninfer::PagedKVPlaneOrder order = ninfer::PagedKVPlaneOrder::PageMajor) {
    ninfer::LayoutBuilder builder;
    auto layout = ninfer::plan_paged_kv_pool(builder, {.page_group_count      = pages,
                                                       .logical_page_capacity = logical_pages,
                                                       .table_rows            = rows,
                                                       .plane_order           = order,
                                                       .planes                = std::move(planes)});
    return PlannedPagedCache{std::move(layout), builder.finish(256)};
}

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_size(std::size_t actual, std::size_t expected, const char* label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

int check_shape(const ninfer::Tensor& tensor, const std::int32_t (&expected)[4],
                const char* label) {
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor.ne[i] != expected[i]) {
            ++failures;
            std::cerr << label << ".ne[" << i << "] expected " << expected[i] << ", got "
                      << tensor.ne[i] << '\n';
        }
    }
    return failures;
}

int expect_page_ids(std::span<const std::int32_t> actual,
                    std::initializer_list<std::int32_t> expected, const char* label) {
    if (actual.size() == expected.size() &&
        std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) {
        return 0;
    }
    std::cerr << label << " page IDs differ\n";
    return 1;
}

int expect_device_page_ids(const ninfer::Tensor& row, std::initializer_list<std::int32_t> expected,
                           const char* label) {
    std::vector<std::int32_t> actual(expected.size());
    const cudaError_t err = cudaMemcpy(
        actual.data(), row.data, actual.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    return expect_page_ids(actual, expected, label);
}

int expect_zeroed_pages(ninfer::PagedKVPool& pool, ninfer::PagedKVPlaneOrder order,
                        std::span<const std::int32_t> pages, cudaStream_t stream,
                        const char* label) {
    const ninfer::Tensor& plane = pool.plane(0);
    cudaError_t err             = cudaMemsetAsync(plane.data, 0x5a, plane.bytes(), stream);
    if (err != cudaSuccess) {
        std::cerr << label << " setup failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    pool.zero_pages(pages, stream);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << label << " synchronization failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::vector<unsigned char> actual(plane.bytes());
    err = cudaMemcpy(actual.data(), plane.data, actual.size(), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << label << " copy failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    std::vector<unsigned char> expected(actual.size(), 0x5a);
    for (const std::int32_t page : pages) {
        if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
            const std::size_t begin = static_cast<std::size_t>(page * plane.nb[3]);
            std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                      expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[3]), 0);
        } else {
            for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
                const std::size_t begin =
                    static_cast<std::size_t>(head * plane.nb[3] + page * plane.nb[2]);
                std::fill(expected.begin() + static_cast<std::ptrdiff_t>(begin),
                          expected.begin() + static_cast<std::ptrdiff_t>(begin + plane.nb[2]), 0);
            }
        }
    }
    if (actual == expected) { return 0; }
    std::cerr << label << " cleared bytes outside the selected physical pages\n";
    return 1;
}

int expect_logical_roundtrip(ninfer::PagedKVPlaneOrder order, cudaStream_t stream,
                              const char* label) {
    ninfer::PinnedTransferBuffer transfer(14);
    constexpr std::uint32_t kValidTokens = 5U * 64U + 7U;
    auto source_plan = plan_paged_cache(8, 8, 1, {{ninfer::DType::I8, 3, 2}}, order);
    ninfer::DeviceArena source_arena(source_plan.bytes);
    ninfer::PagedKVPool source_pool({source_arena.base(), source_arena.capacity()},
                                    source_plan.layout);

    auto released = source_pool.reserve(2);
    auto occupied = source_pool.reserve(2);
    released.materialize_pages(2);
    occupied.materialize_pages(2);
    released.release();
    auto source = source_pool.reserve(6);
    source.materialize_pages(6);
    if (source.page_ids()[2] == source.page_ids()[1] + 1) {
        std::cerr << label << " fixture did not create a fragmented source mapping\n";
        return 1;
    }

    const ninfer::Tensor& source_plane = source_pool.plane(0);
    std::vector<std::uint8_t> physical(source_plane.bytes(), 0xee);
    for (std::uint32_t logical = 0; logical < source.mapped_page_count(); ++logical) {
        const std::int32_t page = source.page_ids()[logical];
        for (std::int32_t head = 0; head < 2; ++head) {
            const std::size_t base =
                order == ninfer::PagedKVPlaneOrder::PageMajor
                    ? static_cast<std::size_t>(page) * source_plane.nb[3] +
                          static_cast<std::size_t>(head) * source_plane.nb[2]
                    : static_cast<std::size_t>(head) * source_plane.nb[3] +
                          static_cast<std::size_t>(page) * source_plane.nb[2];
            for (std::size_t offset = 0; offset < static_cast<std::size_t>(source_plane.nb[2]);
                 ++offset) {
                physical[base + offset] = static_cast<std::uint8_t>(
                    (logical * 37U + static_cast<std::uint32_t>(head) * 13U + offset) % 251U);
            }
        }
    }
    cudaError_t err = cudaMemcpyAsync(source_plane.data, physical.data(), physical.size(),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << label << " source upload failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    const ninfer::PagedKVLogicalImage image =
        ninfer::export_paged_kv_logical(source, kValidTokens, transfer, stream);
    if (image.valid_tokens != kValidTokens || image.payloads.size() != 1) {
        std::cerr << label << " exported invalid image metadata\n";
        return 1;
    }
    const auto& payload = image.payloads[0];
    const std::size_t logical_head_stride = 6U * source_plane.nb[2];
    for (std::int32_t head = 0; head < 2; ++head) {
        const std::size_t page_base =
            order == ninfer::PagedKVPlaneOrder::PageMajor
                ? 5U * source_plane.nb[3] + static_cast<std::size_t>(head) * source_plane.nb[2]
                : static_cast<std::size_t>(head) * logical_head_stride + 5U * source_plane.nb[2];
        const std::size_t tail = page_base + 7U * source_plane.nb[1];
        if (!std::all_of(payload.begin() + static_cast<std::ptrdiff_t>(tail),
                         payload.begin() + static_cast<std::ptrdiff_t>(page_base +
                                                                      source_plane.nb[2]),
                         [](std::uint8_t value) { return value == 0; })) {
            std::cerr << label << " did not canonicalize the invalid final-page tail\n";
            return 1;
        }
    }

    auto destination_plan = plan_paged_cache(8, 8, 1, {{ninfer::DType::I8, 3, 2}}, order);
    ninfer::DeviceArena destination_arena(destination_plan.bytes);
    ninfer::PagedKVPool destination_pool(
        {destination_arena.base(), destination_arena.capacity()}, destination_plan.layout);
    auto prefix = destination_pool.reserve(2);
    prefix.materialize_pages(2);
    auto destination = destination_pool.reserve(6);
    err = cudaMemsetAsync(destination_pool.plane(0).data, 0x5a,
                          destination_pool.plane(0).bytes(), stream);
    if (err != cudaSuccess) {
        std::cerr << label << " destination setup failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    ninfer::import_paged_kv_logical(destination, image, transfer, stream);
    const ninfer::PagedKVLogicalImage restored =
        ninfer::export_paged_kv_logical(destination, kValidTokens, transfer, stream);
    if (restored.payloads != image.payloads || restored.planes.size() != image.planes.size() ||
        restored.plane_order != image.plane_order) {
        std::cerr << label << " logical payload changed across fresh physical allocation\n";
        return 1;
    }
    return 0;
}

int expect_cyclic_lane_roundtrip(cudaStream_t stream) {
    ninfer::PinnedTransferBuffer transfer(14);
    auto source_plan = plan_cyclic_cache(2);
    ninfer::DeviceArena source_arena(source_plan.bytes);
    ninfer::CyclicKVCache source({source_arena.base(), source_arena.capacity()},
                                 source_plan.layout);

    std::vector<std::vector<std::uint8_t>> expected_k(source.layer_count());
    std::vector<std::vector<std::uint8_t>> expected_v(source.layer_count());
    for (std::uint32_t layer = 0; layer < source.layer_count(); ++layer) {
        const ninfer::CyclicKVCacheLayerView view = source.layer_view(layer);
        for (int plane = 0; plane < 2; ++plane) {
            const ninfer::Tensor lane = (plane == 0 ? view.k : view.v).slice(3, 1, 1);
            auto& bytes               = (plane == 0 ? expected_k : expected_v)[layer];
            bytes.resize(lane.bytes());
            for (std::size_t offset = 0; offset < bytes.size(); ++offset) {
                bytes[offset] = static_cast<std::uint8_t>(
                    (offset * 29U + layer * 71U + static_cast<std::uint32_t>(plane) * 113U) % 251U);
            }
            const cudaError_t err = cudaMemcpyAsync(lane.data, bytes.data(), bytes.size(),
                                                    cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                std::cerr << "cyclic source upload failed: " << cudaGetErrorString(err) << '\n';
                return 1;
            }
        }
    }

    const ninfer::CyclicKVCacheImage image =
        ninfer::export_cyclic_kv_lane(source, 1, transfer, stream);
    if (image.layers != 2 || image.capacity != 129 || image.padded_capacity != 256 ||
        image.num_kv_heads != 2 || image.head_dim != 3 || image.k.size() != 2 ||
        image.v.size() != 2 || image.k != expected_k || image.v != expected_v) {
        std::cerr << "cyclic export metadata or layer payloads are invalid\n";
        return 1;
    }

    auto destination_plan = plan_cyclic_cache(3);
    ninfer::DeviceArena destination_arena(destination_plan.bytes);
    ninfer::CyclicKVCache destination({destination_arena.base(), destination_arena.capacity()},
                                      destination_plan.layout);
    cudaError_t err =
        cudaMemsetAsync(destination_arena.base(), 0x5a, destination_arena.capacity(), stream);
    if (err != cudaSuccess) {
        std::cerr << "cyclic destination setup failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    ninfer::import_cyclic_kv_lane(destination, 2, image, transfer, stream);
    const ninfer::CyclicKVCacheImage restored =
        ninfer::export_cyclic_kv_lane(destination, 2, transfer, stream);
    if (restored.k != image.k || restored.v != image.v) {
        std::cerr << "cyclic lane bytes changed across export/import\n";
        return 1;
    }

    const ninfer::CyclicKVCacheImage untouched =
        ninfer::export_cyclic_kv_lane(destination, 1, transfer, stream);
    for (const auto& payload : untouched.k) {
        if (!std::all_of(payload.begin(), payload.end(),
                         [](std::uint8_t value) { return value == 0x5a; })) {
            std::cerr << "cyclic import modified another lane\n";
            return 1;
        }
    }
    for (const auto& payload : untouched.v) {
        if (!std::all_of(payload.begin(), payload.end(),
                         [](std::uint8_t value) { return value == 0x5a; })) {
            std::cerr << "cyclic import modified another lane\n";
            return 1;
        }
    }

    ninfer::CyclicKVCacheImage invalid = image;
    ++invalid.head_dim;
    try {
        ninfer::import_cyclic_kv_lane(destination, 0, invalid, transfer, stream);
        std::cerr << "cyclic import accepted incompatible geometry\n";
        return 1;
    } catch (const std::invalid_argument&) {}
    invalid = image;
    invalid.v[1].pop_back();
    try {
        ninfer::import_cyclic_kv_lane(destination, 0, invalid, transfer, stream);
        std::cerr << "cyclic import accepted a truncated layer payload\n";
        return 1;
    } catch (const std::invalid_argument&) {}
    return 0;
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    int failures = 0;
    ninfer::DeviceContext ctx(0);

    auto paged_plan = plan_paged_cache(10, 10, 2,
                                       {{ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::I8, 64, 2},
                                        {ninfer::DType::FP16, 1, 2},
                                        {ninfer::DType::FP16, 1, 2}});
    ninfer::DeviceArena paged_arena(paged_plan.bytes);
    ninfer::PagedKVPool paged_pool({paged_arena.base(), paged_arena.capacity()}, paged_plan.layout);
    failures += expect_size(paged_pool.plane_count(), 4, "paged plane count");
    failures += check_shape(paged_pool.plane(0), {64, 64, 2, 10}, "paged code plane");
    failures += check_shape(paged_pool.plane(2), {1, 64, 2, 10}, "paged scale plane");
    failures += check_shape(paged_pool.block_table_row(0), {10, 1, 1, 1}, "paged block-table row");
    if (!paged_pool.plane(0).is_contiguous() || !paged_pool.plane(2).is_contiguous() ||
        !paged_pool.block_table_row(0).is_contiguous()) {
        ++failures;
        std::cerr << "Paged planes or block-table row are not contiguous\n";
    }
    const std::size_t expected_payload =
        2 * ninfer::Tensor(nullptr, ninfer::DType::I8, {64, 64, 2, 10}).bytes() +
        2 * ninfer::Tensor(nullptr, ninfer::DType::FP16, {1, 64, 2, 10}).bytes();
    failures +=
        expect_size(paged_plan.layout.payload_bytes(), expected_payload, "paged payload bytes");
    failures += expect_size(paged_plan.layout.metadata_bytes(), 10 * 2 * sizeof(std::int32_t),
                            "paged metadata bytes");
    const std::int32_t selected_pages[] = {1, 2, 6};
    failures += expect_zeroed_pages(paged_pool, ninfer::PagedKVPlaneOrder::PageMajor,
                                    selected_pages, ctx.stream, "page-major selective zero");

    auto head_major_plan = plan_paged_cache(10, 10, 1, {{ninfer::DType::BF16, 128, 8}},
                                            ninfer::PagedKVPlaneOrder::HeadMajor);
    ninfer::DeviceArena head_major_arena(head_major_plan.bytes);
    ninfer::PagedKVPool head_major_pool({head_major_arena.base(), head_major_arena.capacity()},
                                        head_major_plan.layout);
    failures += check_shape(head_major_pool.plane(0), {128, 64, 10, 8}, "head-major paged plane");
    failures += expect_zeroed_pages(head_major_pool, ninfer::PagedKVPlaneOrder::HeadMajor,
                                     selected_pages, ctx.stream, "head-major selective zero");
    failures += expect_logical_roundtrip(ninfer::PagedKVPlaneOrder::PageMajor, ctx.stream,
                                         "page-major logical roundtrip");
    failures += expect_logical_roundtrip(ninfer::PagedKVPlaneOrder::HeadMajor, ctx.stream,
                                         "head-major logical roundtrip");
    failures += expect_cyclic_lane_roundtrip(ctx.stream);

    auto allocation_a = paged_pool.reserve(3);
    auto allocation_b = paged_pool.reserve(3);
    allocation_a.materialize_pages(3);
    allocation_b.materialize_pages(3);
    allocation_a.bind_row(0);
    allocation_b.bind_row(1);
    failures += expect_device_page_ids(allocation_a.block_table(), {0, 1, 2}, "allocation A");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "allocation B");
    allocation_a.release();

    auto allocation_c = paged_pool.reserve(6);
    allocation_c.materialize_pages(6);
    allocation_c.bind_row(0);
    failures +=
        expect_page_ids(allocation_c.page_ids(), {0, 1, 2, 6, 7, 8}, "fragmented allocation C");
    failures +=
        expect_device_page_ids(allocation_c.block_table(), {0, 1, 2, 6, 7, 8}, "fragmented row C");
    failures += expect_device_page_ids(allocation_b.block_table(), {3, 4, 5}, "isolated row B");

    allocation_c.trim_pages(2);
    if (paged_pool.can_reserve(2)) {
        ++failures;
        std::cerr << "Unmapped entitlement was exposed as reservable capacity\n";
    }
    allocation_c.cancel_unmapped_entitlement();
    if (!paged_pool.can_reserve(5)) {
        ++failures;
        std::cerr << "Cancelled entitlement did not return reservable capacity\n";
    }

    allocation_c.unbind_row();
    const std::int32_t retained_prefix[] = {allocation_c.page_ids()[0], allocation_c.page_ids()[1]};
    const ninfer::PagedKVResize claim[]  = {
        {.allocation = &allocation_c, .mapped_pages = 2, .page_entitlement = 5}};
    ninfer::resize_paged_kv_bundle(claim);
    allocation_c.bind_row(0);
    allocation_c.materialize_pages(5);
    if (allocation_c.page_ids()[0] != retained_prefix[0] ||
        allocation_c.page_ids()[1] != retained_prefix[1]) {
        ++failures;
        std::cerr << "Retained allocation moved its prefix pages\n";
    }
    allocation_c.trim_tokens(65);
    failures += expect_size(allocation_c.mapped_page_count(), 2, "partial-tail mapped pages");
    allocation_c.trim_tokens(64);
    failures += expect_size(allocation_c.mapped_page_count(), 1, "page-aligned mapped pages");

    auto main_plan    = plan_paged_cache(4, 4, 1, {{ninfer::DType::BF16, 16, 1}});
    auto backend_plan = plan_paged_cache(2, 2, 1, {{ninfer::DType::BF16, 16, 1}});
    ninfer::DeviceArena main_arena(main_plan.bytes);
    ninfer::DeviceArena backend_arena(backend_plan.bytes);
    ninfer::PagedKVPool main_pool({main_arena.base(), main_arena.capacity()}, main_plan.layout);
    ninfer::PagedKVPool backend_pool({backend_arena.base(), backend_arena.capacity()},
                                     backend_plan.layout);
    const ninfer::PagedKVReservation impossible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 3},
        {.pool = &backend_pool, .page_entitlement = 3},
    };
    try {
        auto unused = ninfer::reserve_paged_kv_bundle(impossible_bundle);
        (void)unused;
        ++failures;
        std::cerr << "Impossible multi-pool reservation succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(main_pool.entitled_pages(), 0, "failed bundle main entitlement");
    failures += expect_size(backend_pool.entitled_pages(), 0, "failed bundle backend entitlement");

    const ninfer::PagedKVReservation possible_bundle[] = {
        {.pool = &main_pool, .page_entitlement = 2},
        {.pool = &backend_pool, .page_entitlement = 2},
    };
    auto bundle = ninfer::reserve_paged_kv_bundle(possible_bundle);
    bundle[0].materialize_pages(2);
    bundle[1].materialize_pages(2);
    const ninfer::PagedKVResize impossible_resize[] = {
        {.allocation = &bundle[0], .mapped_pages = 1, .page_entitlement = 1},
        {.allocation = &bundle[1], .mapped_pages = 1, .page_entitlement = 3},
    };
    try {
        ninfer::resize_paged_kv_bundle(impossible_resize);
        ++failures;
        std::cerr << "Impossible multi-pool resize succeeded\n";
    } catch (const std::bad_alloc&) {}
    failures += expect_size(bundle[0].mapped_page_count(), 2, "failed resize main mapping");
    failures += expect_size(bundle[0].page_entitlement(), 2, "failed resize main entitlement");
    failures += expect_size(bundle[1].mapped_page_count(), 2, "failed resize backend mapping");
    failures += expect_size(bundle[1].page_entitlement(), 2, "failed resize backend entitlement");

    return failures == 0 ? 0 : fail("kv cache test failed");
}
