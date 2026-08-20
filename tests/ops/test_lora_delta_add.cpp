// Qualification suite for ops::lora_delta_add.
//
// The oracle evaluates `destination + B * (A * x)` naively in FP64 from the
// represented BF16 inputs, per token and per site, honouring the same adapter
// routing the Op is given. It shares no staging, accumulation order, or
// workspace with the kernel.
//
// Three shapes of evidence:
//   * a canary whose delta is a single analytically known entry, which pins
//     factor orientation, row/column order and banked adapter indexing;
//   * dense random factors at registered site geometries, against the oracle;
//   * routing, including a negative index that must leave a column untouched.

#include "ninfer/ops/lora.h"
#include "ops/op_tester.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// A rank-r contraction of BF16 factors accumulated in FP32. The rank bound is
// small, so the criterion follows BF16 output storage rather than depth.
constexpr PointwiseCriterion lora_bf16_criterion() {
    return {/*absolute*/ 3.0e-5, /*relative*/ 6.0e-3};
}

struct SiteShape {
    const char* name;
    std::int32_t n;
    std::int32_t k;
};

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

// x is [K,T] column-major over tokens; a is [rank,K]; b is [n,rank]; dest is [N,T].
// Every plane is banked: adapter i begins i * plane elements later.
std::vector<double> lora_oracle(const std::vector<float>& dest, const std::vector<float>& x,
                                const std::vector<float>& a, const std::vector<float>& b,
                                const std::vector<std::int32_t>& adapters, std::int32_t n,
                                std::int32_t k, std::int32_t rank, std::int32_t tokens) {
    std::vector<double> expected(dest.begin(), dest.end());
    const std::size_t a_plane = static_cast<std::size_t>(rank) * k;
    const std::size_t b_plane = static_cast<std::size_t>(n) * rank;
    const std::int32_t span   = tokens / static_cast<std::int32_t>(adapters.size());

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t adapter =
            adapters[adapters.size() == 1 ? 0 : static_cast<std::size_t>(token / span)];
        if (adapter < 0) { continue; }
        const std::size_t a_base = static_cast<std::size_t>(adapter) * a_plane;
        const std::size_t b_base = static_cast<std::size_t>(adapter) * b_plane;

        std::vector<double> temp(static_cast<std::size_t>(rank), 0.0);
        for (std::int32_t j = 0; j < rank; ++j) {
            double sum = 0.0;
            for (std::int32_t column = 0; column < k; ++column) {
                sum += static_cast<double>(a[a_base + static_cast<std::size_t>(j) * k + column]) *
                       static_cast<double>(x[static_cast<std::size_t>(token) * k + column]);
            }
            temp[static_cast<std::size_t>(j)] = sum;
        }
        for (std::int32_t row = 0; row < n; ++row) {
            double sum = 0.0;
            for (std::int32_t j = 0; j < rank; ++j) {
                sum += static_cast<double>(b[b_base + static_cast<std::size_t>(row) * rank + j]) *
                       temp[static_cast<std::size_t>(j)];
            }
            expected[static_cast<std::size_t>(token) * n + row] += sum;
        }
    }
    return expected;
}

struct SiteBuffers {
    GuardedDeviceBuffer a;
    GuardedDeviceBuffer b;
    GuardedDeviceBuffer dest;
    std::vector<float> host_a;
    std::vector<float> host_b;
    std::vector<float> host_dest;
    std::int32_t n = 0;
    std::int32_t k = 0;

    SiteBuffers(std::size_t a_bytes, std::size_t b_bytes, std::size_t dest_bytes)
        : a(a_bytes), b(b_bytes), dest(dest_bytes) {}
};

int run_dense_case(const char* label, const SiteShape& shape, std::int32_t rank,
                   std::int32_t tokens, std::int32_t adapter_count,
                   const std::vector<std::int32_t>& adapters, std::uint32_t seed) {
    const std::size_t a_count    = static_cast<std::size_t>(adapter_count) * rank * shape.k;
    const std::size_t b_count    = static_cast<std::size_t>(adapter_count) * shape.n * rank;
    const std::size_t x_count    = static_cast<std::size_t>(shape.k) * tokens;
    const std::size_t dest_count = static_cast<std::size_t>(shape.n) * tokens;

    std::vector<float> host_a(a_count), host_b(b_count), host_x(x_count), host_dest(dest_count);
    fill_uniform(host_a, seed, -0.08f, 0.08f);
    fill_uniform(host_b, seed + 1, -0.08f, 0.08f);
    fill_uniform(host_x, seed + 2, -4.0f, 4.0f);
    fill_uniform(host_dest, seed + 3, -6.0f, 6.0f);
    round_to_bf16(host_a);
    round_to_bf16(host_b);
    round_to_bf16(host_x);
    round_to_bf16(host_dest);

    const auto expected =
        lora_oracle(host_dest, host_x, host_a, host_b, adapters, shape.n, shape.k, rank, tokens);

    const auto a_bits    = encode_bf16(host_a);
    const auto b_bits    = encode_bf16(host_b);
    const auto x_bits    = encode_bf16(host_x);
    const auto dest_bits = encode_bf16(host_dest);

    GuardedDeviceBuffer device_a(a_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_b(b_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(x_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dest(dest_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_index(adapters.size() * sizeof(std::int32_t));
    device_a.copy_from_host(a_bits.data(), device_a.bytes());
    device_b.copy_from_host(b_bits.data(), device_b.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_dest.copy_from_host(dest_bits.data(), device_dest.bytes());
    device_index.copy_from_host(adapters.data(), device_index.bytes());

    ops::LoraGroup group;
    group.rank          = rank;
    group.adapter_count = adapter_count;
    group.site_count    = 1;
    group.sites[0]      = ops::LoraSite{
        .a                = device_a.data(),
        .b                = device_b.data(),
        .n                = shape.n,
        .k                = shape.k,
        .a_adapter_stride = static_cast<std::size_t>(rank) * shape.k * sizeof(std::uint16_t),
        .b_adapter_stride = static_cast<std::size_t>(shape.n) * rank * sizeof(std::uint16_t),
    };

    Tensor x_tensor(device_x.data(), DType::BF16, {shape.k, tokens});
    Tensor dest_tensor(device_dest.data(), DType::BF16, {shape.n, tokens});
    Tensor index_tensor(device_index.data(), DType::I32,
                        {static_cast<std::int32_t>(adapters.size())});
    std::array<Tensor*, 1> destinations{&dest_tensor};

    DeviceArena workspace(ops::lora_delta_add_workspace_capacity_bytes(rank, 1, tokens, tokens) +
                          4096);
    ops::lora_delta_add(x_tensor, group, index_tensor, destinations, workspace, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_dest.data(), dest_count),
                                   expected, lora_bf16_criterion());
    failures += verify_exact("lora_delta_add x unchanged",
                             from_device<std::uint16_t>(device_x.data(), x_count), x_bits);
    failures += verify_exact("lora_delta_add A unchanged",
                             from_device<std::uint16_t>(device_a.data(), a_count), a_bits);
    failures += verify_exact("lora_delta_add B unchanged",
                             from_device<std::uint16_t>(device_b.data(), b_count), b_bits);
    failures += device_dest.verify_guards(label);
    failures += device_x.verify_guards("lora_delta_add x");
    return failures;
}

// A rank-one one-hot pair: A[slot,column]=1 and B[row,slot]=c, so the delta is
// exactly c*x[column,t] at output row `row` and nothing anywhere else.
int run_canary_case(const char* label, const SiteShape& shape, std::int32_t rank,
                    std::int32_t tokens, std::int32_t slot, std::int32_t row, std::int32_t column,
                    float constant) {
    const std::size_t a_count    = static_cast<std::size_t>(rank) * shape.k;
    const std::size_t b_count    = static_cast<std::size_t>(shape.n) * rank;
    const std::size_t x_count    = static_cast<std::size_t>(shape.k) * tokens;
    const std::size_t dest_count = static_cast<std::size_t>(shape.n) * tokens;

    std::vector<float> host_a(a_count, 0.0f), host_b(b_count, 0.0f);
    host_a[static_cast<std::size_t>(slot) * shape.k + column] = 1.0f;
    host_b[static_cast<std::size_t>(row) * rank + slot]       = constant;

    std::vector<float> host_x(x_count), host_dest(dest_count, 0.0f);
    fill_uniform(host_x, 4242, -4.0f, 4.0f);
    round_to_bf16(host_x);

    const auto a_bits    = encode_bf16(host_a);
    const auto b_bits    = encode_bf16(host_b);
    const auto x_bits    = encode_bf16(host_x);
    const auto dest_bits = encode_bf16(host_dest);

    GuardedDeviceBuffer device_a(a_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_b(b_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(x_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dest(dest_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_index(sizeof(std::int32_t));
    const std::int32_t uniform = 0;
    device_a.copy_from_host(a_bits.data(), device_a.bytes());
    device_b.copy_from_host(b_bits.data(), device_b.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_dest.copy_from_host(dest_bits.data(), device_dest.bytes());
    device_index.copy_from_host(&uniform, sizeof(uniform));

    ops::LoraGroup group;
    group.rank          = rank;
    group.adapter_count = 1;
    group.site_count    = 1;
    group.sites[0]      = ops::LoraSite{
        .a                = device_a.data(),
        .b                = device_b.data(),
        .n                = shape.n,
        .k                = shape.k,
        .a_adapter_stride = static_cast<std::size_t>(rank) * shape.k * sizeof(std::uint16_t),
        .b_adapter_stride = static_cast<std::size_t>(shape.n) * rank * sizeof(std::uint16_t),
    };

    Tensor x_tensor(device_x.data(), DType::BF16, {shape.k, tokens});
    Tensor dest_tensor(device_dest.data(), DType::BF16, {shape.n, tokens});
    Tensor index_tensor(device_index.data(), DType::I32, {1});
    std::array<Tensor*, 1> destinations{&dest_tensor};

    DeviceArena workspace(ops::lora_delta_add_workspace_capacity_bytes(rank, 1, tokens, tokens) +
                          4096);
    ops::lora_delta_add(x_tensor, group, index_tensor, destinations, workspace, nullptr);
    cuda_synchronize();

    const auto observed = from_device_bf16(device_dest.data(), dest_count);
    std::vector<double> expected(dest_count, 0.0);
    for (std::int32_t token = 0; token < tokens; ++token) {
        expected[static_cast<std::size_t>(token) * shape.n + row] =
            static_cast<double>(constant) *
            static_cast<double>(host_x[static_cast<std::size_t>(token) * shape.k + column]);
    }

    int failures = verify_pointwise(label, observed, expected, lora_bf16_criterion());
    // Every other output row must remain exactly the zero it started as.
    std::int64_t stray = 0;
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t output = 0; output < shape.n; ++output) {
            if (output == row) { continue; }
            if (observed[static_cast<std::size_t>(token) * shape.n + output] != 0.0) { ++stray; }
        }
    }
    if (stray != 0) {
        std::cerr << label << ": " << stray << " entries outside the canary row were disturbed\n";
        ++failures;
    }
    failures += device_dest.verify_guards(label);
    return failures;
}

// Four sites sharing one activation, mirroring the attention projection group.
int run_group_routing_case(std::int32_t rank, std::int32_t tokens, std::int32_t adapter_count,
                           const std::vector<std::int32_t>& adapters) {
    const std::array<SiteShape, 4> shapes{SiteShape{"query", 384, 512},
                                          SiteShape{"gate", 384, 512},
                                          SiteShape{"key", 128, 512},
                                          SiteShape{"value", 128, 512}};
    constexpr std::int32_t kColumns = 512;

    const std::size_t x_count = static_cast<std::size_t>(kColumns) * tokens;
    std::vector<float> host_x(x_count);
    fill_uniform(host_x, 909, -4.0f, 4.0f);
    round_to_bf16(host_x);
    const auto x_bits = encode_bf16(host_x);
    GuardedDeviceBuffer device_x(x_count * sizeof(std::uint16_t));
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    GuardedDeviceBuffer device_index(adapters.size() * sizeof(std::int32_t));
    device_index.copy_from_host(adapters.data(), device_index.bytes());

    std::vector<std::unique_ptr<SiteBuffers>> buffers;
    std::vector<Tensor> dest_tensors;
    dest_tensors.reserve(shapes.size());

    ops::LoraGroup group;
    group.rank          = rank;
    group.adapter_count = adapter_count;
    group.site_count    = static_cast<std::int32_t>(shapes.size());

    for (std::size_t index = 0; index < shapes.size(); ++index) {
        const SiteShape& shape       = shapes[index];
        const std::size_t a_count    = static_cast<std::size_t>(adapter_count) * rank * shape.k;
        const std::size_t b_count    = static_cast<std::size_t>(adapter_count) * shape.n * rank;
        const std::size_t dest_count = static_cast<std::size_t>(shape.n) * tokens;

        auto site = std::make_unique<SiteBuffers>(a_count * sizeof(std::uint16_t),
                                                  b_count * sizeof(std::uint16_t),
                                                  dest_count * sizeof(std::uint16_t));
        site->n = shape.n;
        site->k = shape.k;
        site->host_a.resize(a_count);
        site->host_b.resize(b_count);
        site->host_dest.resize(dest_count);
        fill_uniform(site->host_a, 100 + static_cast<std::uint32_t>(index) * 7, -0.08f, 0.08f);
        fill_uniform(site->host_b, 200 + static_cast<std::uint32_t>(index) * 7, -0.08f, 0.08f);
        fill_uniform(site->host_dest, 300 + static_cast<std::uint32_t>(index) * 7, -6.0f, 6.0f);
        round_to_bf16(site->host_a);
        round_to_bf16(site->host_b);
        round_to_bf16(site->host_dest);
        site->a.copy_from_host(encode_bf16(site->host_a).data(), site->a.bytes());
        site->b.copy_from_host(encode_bf16(site->host_b).data(), site->b.bytes());
        site->dest.copy_from_host(encode_bf16(site->host_dest).data(), site->dest.bytes());

        group.sites[index] = ops::LoraSite{
            .a                = site->a.data(),
            .b                = site->b.data(),
            .n                = shape.n,
            .k                = shape.k,
            .a_adapter_stride = static_cast<std::size_t>(rank) * shape.k * sizeof(std::uint16_t),
            .b_adapter_stride = static_cast<std::size_t>(shape.n) * rank * sizeof(std::uint16_t),
        };
        buffers.push_back(std::move(site));
    }
    for (auto& site : buffers) {
        dest_tensors.push_back(Tensor(site->dest.data(), DType::BF16, {site->n, tokens}));
    }

    std::vector<Tensor*> destinations;
    destinations.reserve(dest_tensors.size());
    for (auto& tensor : dest_tensors) { destinations.push_back(&tensor); }

    Tensor x_tensor(device_x.data(), DType::BF16, {kColumns, tokens});
    Tensor index_tensor(device_index.data(), DType::I32,
                        {static_cast<std::int32_t>(adapters.size())});

    DeviceArena workspace(
        ops::lora_delta_add_workspace_capacity_bytes(
            rank, static_cast<std::int32_t>(shapes.size()), tokens, tokens) +
        4096);
    ops::lora_delta_add(x_tensor, group, index_tensor, destinations, workspace, nullptr);
    cuda_synchronize();

    int failures = 0;
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        SiteBuffers& site            = *buffers[index];
        const std::size_t dest_count = static_cast<std::size_t>(site.n) * tokens;
        const auto expected = lora_oracle(site.host_dest, host_x, site.host_a, site.host_b,
                                          adapters, site.n, site.k, rank, tokens);
        const std::string label =
            std::string("lora_delta_add group site ") + shapes[index].name;
        failures += verify_pointwise(label.c_str(), from_device_bf16(site.dest.data(), dest_count),
                                     expected, lora_bf16_criterion());
        failures += site.dest.verify_guards(label.c_str());
    }
    return failures;
}

// A negative index must leave its column bit-identical, not merely close.
int run_base_column_is_untouched_case() {
    constexpr std::int32_t kRank = 16;
    constexpr std::int32_t kN = 256, kK = 512, kTokens = 8;
    const std::vector<std::int32_t> adapters{-1, 0, 1, 0, -1, 1, 0, -1};

    const std::size_t a_count    = static_cast<std::size_t>(2) * kRank * kK;
    const std::size_t b_count    = static_cast<std::size_t>(2) * kN * kRank;
    const std::size_t x_count    = static_cast<std::size_t>(kK) * kTokens;
    const std::size_t dest_count = static_cast<std::size_t>(kN) * kTokens;

    std::vector<float> host_a(a_count), host_b(b_count), host_x(x_count), host_dest(dest_count);
    fill_uniform(host_a, 11, -0.1f, 0.1f);
    fill_uniform(host_b, 12, -0.1f, 0.1f);
    fill_uniform(host_x, 13, -4.0f, 4.0f);
    fill_uniform(host_dest, 14, -6.0f, 6.0f);
    round_to_bf16(host_a);
    round_to_bf16(host_b);
    round_to_bf16(host_x);
    round_to_bf16(host_dest);
    const auto dest_bits = encode_bf16(host_dest);

    GuardedDeviceBuffer device_a(a_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_b(b_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(x_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dest(dest_count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_index(adapters.size() * sizeof(std::int32_t));
    device_a.copy_from_host(encode_bf16(host_a).data(), device_a.bytes());
    device_b.copy_from_host(encode_bf16(host_b).data(), device_b.bytes());
    device_x.copy_from_host(encode_bf16(host_x).data(), device_x.bytes());
    device_dest.copy_from_host(dest_bits.data(), device_dest.bytes());
    device_index.copy_from_host(adapters.data(), device_index.bytes());

    ops::LoraGroup group;
    group.rank          = kRank;
    group.adapter_count = 2;
    group.site_count    = 1;
    group.sites[0]      = ops::LoraSite{
        .a                = device_a.data(),
        .b                = device_b.data(),
        .n                = kN,
        .k                = kK,
        .a_adapter_stride = static_cast<std::size_t>(kRank) * kK * sizeof(std::uint16_t),
        .b_adapter_stride = static_cast<std::size_t>(kN) * kRank * sizeof(std::uint16_t),
    };

    Tensor x_tensor(device_x.data(), DType::BF16, {kK, kTokens});
    Tensor dest_tensor(device_dest.data(), DType::BF16, {kN, kTokens});
    Tensor index_tensor(device_index.data(), DType::I32,
                        {static_cast<std::int32_t>(adapters.size())});
    std::array<Tensor*, 1> destinations{&dest_tensor};

    DeviceArena workspace(
        ops::lora_delta_add_workspace_capacity_bytes(kRank, 1, kTokens, kTokens) + 4096);
    ops::lora_delta_add(x_tensor, group, index_tensor, destinations, workspace, nullptr);
    cuda_synchronize();

    const auto observed = from_device<std::uint16_t>(device_dest.data(), dest_count);
    int failures        = 0;
    for (std::int32_t token = 0; token < kTokens; ++token) {
        if (adapters[static_cast<std::size_t>(token)] >= 0) { continue; }
        for (std::int32_t row = 0; row < kN; ++row) {
            const std::size_t index = static_cast<std::size_t>(token) * kN + row;
            if (observed[index] != dest_bits[index]) {
                std::cerr << "lora_delta_add base column " << token << " row " << row
                          << " changed: " << dest_bits[index] << " -> " << observed[index] << "\n";
                ++failures;
                break;
            }
        }
    }
    const auto expected = lora_oracle(host_dest, host_x, host_a, host_b, adapters, kN, kK, kRank,
                                      kTokens);
    failures += verify_pointwise("lora_delta_add mixed routing",
                                 from_device_bf16(device_dest.data(), dest_count), expected,
                                 lora_bf16_criterion());
    failures += device_dest.verify_guards("lora_delta_add mixed routing");
    return failures;
}

int run_rejection_cases() {
    int failures = 0;
    GuardedDeviceBuffer buffer(1024);
    Tensor x(buffer.data(), DType::BF16, {64, 4});
    Tensor destination(buffer.data(), DType::BF16, {32, 4});
    Tensor index(buffer.data(), DType::I32, {4});
    std::array<Tensor*, 1> destinations{&destination};
    DeviceArena workspace(4096);

    ops::LoraGroup valid;
    valid.rank          = 16;
    valid.adapter_count = 1;
    valid.site_count    = 1;
    valid.sites[0] = ops::LoraSite{.a                = buffer.data(),
                                   .b                = buffer.data(),
                                   .n                = 32,
                                   .k                = 64,
                                   .a_adapter_stride = 0,
                                   .b_adapter_stride = 0};

    const auto expect_throw = [&](const char* label, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return; } catch (...) {
            std::cerr << label << ": wrong exception type\n";
            ++failures;
            return;
        }
        std::cerr << label << ": expected rejection\n";
        ++failures;
    };

    expect_throw("unregistered rank", [&] {
        ops::LoraGroup group = valid;
        group.rank           = 12;
        ops::lora_delta_add(x, group, index, destinations, workspace, nullptr);
    });
    expect_throw("site count above bound", [&] {
        ops::LoraGroup group = valid;
        group.site_count     = 5;
        ops::lora_delta_add(x, group, index, destinations, workspace, nullptr);
    });
    expect_throw("adapter index dtype", [&] {
        Tensor wrong(buffer.data(), DType::BF16, {4});
        ops::lora_delta_add(x, valid, wrong, destinations, workspace, nullptr);
    });
    expect_throw("adapter index extent", [&] {
        Tensor wrong(buffer.data(), DType::I32, {3});
        ops::lora_delta_add(x, valid, wrong, destinations, workspace, nullptr);
    });
    expect_throw("site K disagrees with activation", [&] {
        ops::LoraGroup group = valid;
        group.sites[0].k     = 63;
        ops::lora_delta_add(x, group, index, destinations, workspace, nullptr);
    });
    expect_throw("banked site without stride", [&] {
        ops::LoraGroup group = valid;
        group.adapter_count  = 2;
        ops::lora_delta_add(x, group, index, destinations, workspace, nullptr);
    });
    expect_throw("capacity query rejects an unordered interval",
                 [&] { (void)ops::lora_delta_add_workspace_capacity_bytes(16, 1, 8, 4); });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "lora_delta_add: no CUDA device; skipping\n";
        return 77;
    }

    int failures = 0;

    // Registered site geometries at the 27B target, decode and prefill widths.
    const std::array<SiteShape, 4> registered{SiteShape{"attention/query", 6144, 5120},
                                              SiteShape{"attention/key", 1024, 5120},
                                              SiteShape{"attention/output", 5120, 6144},
                                              SiteShape{"mlp/down", 5120, 17408}};
    for (const SiteShape& shape : registered) {
        for (const std::int32_t tokens : {1, 4, 8}) {
            const std::string label = std::string("lora_delta_add dense ") + shape.name + " T=" +
                                      std::to_string(tokens);
            failures += run_dense_case(label.c_str(), shape, 16, tokens, 1,
                                       std::vector<std::int32_t>{0}, 77 + tokens);
        }
    }
    {
        const SiteShape prefill{"attention/key", 1024, 5120};
        failures += run_dense_case("lora_delta_add dense attention/key T=1024", prefill, 16, 1024,
                                   1, std::vector<std::int32_t>{0}, 5150);
    }

    // Every registered bank rank.
    for (const std::int32_t rank : {8, 16, 32, 64}) {
        const SiteShape shape{"rank sweep", 512, 1024};
        const std::string label =
            std::string("lora_delta_add rank ") + std::to_string(rank);
        failures += run_dense_case(label.c_str(), shape, rank, 5, 1,
                                   std::vector<std::int32_t>{0}, 31 + rank);
    }

    // Canary: orientation, indexing, and isolation.
    failures += run_canary_case("lora_delta_add canary attention/query",
                                SiteShape{"attention/query", 6144, 5120}, 16, 4,
                                /*slot*/ 0, /*row*/ 4097, /*column*/ 3001, /*constant*/ 2.0f);
    failures += run_canary_case("lora_delta_add canary shared-A slot",
                                SiteShape{"attention/gate", 6144, 5120}, 16, 4,
                                /*slot*/ 1, /*row*/ 11, /*column*/ 5119, /*constant*/ -3.0f);
    failures += run_canary_case("lora_delta_add canary mlp/down",
                                SiteShape{"mlp/down", 5120, 17408}, 8, 3,
                                /*slot*/ 7, /*row*/ 5119, /*column*/ 17407, /*constant*/ 1.5f);

    // Routing: banked adapters, mixed within one batch, and the base column.
    failures += run_group_routing_case(16, 8, 1, std::vector<std::int32_t>{0});
    failures += run_group_routing_case(16, 8, 3, std::vector<std::int32_t>{0, 1, 2, 1, 0, 2, 2, 0});
    failures += run_group_routing_case(32, 6, 8,
                                       std::vector<std::int32_t>{7, 6, 5, 4, 3, 2});
    failures += run_group_routing_case(16, 5, 2, std::vector<std::int32_t>{1, -1, 0, -1, 1});
    failures += run_base_column_is_untouched_case();

    // A [width, batch] verify tile routes per sequence: one index entry covers `width` columns.
    // T=12 with 3 entries is width 4; T=12 with 6 entries is width 2.
    failures += run_group_routing_case(16, 12, 3, std::vector<std::int32_t>{2, 0, 1});
    failures += run_group_routing_case(16, 12, 2, std::vector<std::int32_t>{1, -1, 0, 1, -1, 0});
    failures += run_dense_case("lora_delta_add per-sequence routing",
                               SiteShape{"attention/key", 1024, 5120}, 16, 8, 4,
                               std::vector<std::int32_t>{3, -1, 1, 0}, 6161);

    failures += run_rejection_cases();

    if (failures != 0) {
        std::cerr << "lora_delta_add: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "lora_delta_add: all cases passed\n";
    return 0;
}
