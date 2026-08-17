#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename W8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);

    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    w8_small_t_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<W8Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, First + static_cast<int>(Offsets)>...};
}

#if NINFER_W8_SMALL_T_PROFILE == 1
constexpr auto kVocabularyLaunchers =
    make_launchers<W8VocabularyProjectionGeometry, kW8VocabularyFirstSmallT>(
        std::make_index_sequence<kW8VocabularyLastSmallT - kW8VocabularyFirstSmallT + 1>{});
#elif NINFER_W8_SMALL_T_PROFILE == 2
constexpr auto kMtpInputLaunchers =
    make_launchers<W8MtpInputProjectionGeometry, kW8MtpInputFirstSmallT>(
        std::make_index_sequence<kW8MtpInputLastSmallT - kW8MtpInputFirstSmallT + 1>{});
#elif NINFER_W8_SMALL_T_PROFILE == 3
constexpr auto kMtpAttentionLaunchers =
    make_launchers<W8MtpAttentionProjectionGeometry, kW8MtpAttentionFirstSmallT>(
        std::make_index_sequence<kW8MtpAttentionLastSmallT - kW8MtpAttentionFirstSmallT + 1>{});
#elif NINFER_W8_SMALL_T_PROFILE == 4
constexpr auto kMtpAttentionOutputLaunchers =
    make_launchers<W8MtpAttentionOutputGeometry, kW8MtpAttentionOutputFirstSmallT>(
        std::make_index_sequence<kW8MtpAttentionOutputLastSmallT -
                                 kW8MtpAttentionOutputFirstSmallT + 1>{});
#elif NINFER_W8_SMALL_T_PROFILE == 5
constexpr auto kMtpGateUpLaunchers =
    make_launchers<W8MtpGateUpProjectionGeometry, kW8MtpGateUpFirstSmallT>(
        std::make_index_sequence<kW8MtpGateUpLastSmallT - kW8MtpGateUpFirstSmallT + 1>{});
#elif NINFER_W8_SMALL_T_PROFILE == 6
constexpr auto kMtpDownLaunchers =
    make_launchers<W8MtpDownProjectionGeometry, kW8MtpDownFirstSmallT>(
        std::make_index_sequence<kW8MtpDownLastSmallT - kW8MtpDownFirstSmallT + 1>{});
#elif NINFER_W8_SMALL_T_PROFILE == 7
constexpr auto k35bMtpProjectionLaunchers = make_launchers<W835bMtpProjectionGeometry,
                                                           kW835bMtpProjectionFirstSmallT>(
    std::make_index_sequence<kW835bMtpProjectionLastSmallT - kW835bMtpProjectionFirstSmallT + 1>{});
#endif

} // namespace

#if NINFER_W8_SMALL_T_PROFILE == 1
void launch_w8_small_t_profile_1(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream) {
    kVocabularyLaunchers[static_cast<std::size_t>(x.ne[1] - kW8VocabularyFirstSmallT)](x, weight,
                                                                                       out, stream);
}
#elif NINFER_W8_SMALL_T_PROFILE == 2
void launch_w8_small_t_profile_2(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream) {
    kMtpInputLaunchers[static_cast<std::size_t>(x.ne[1] - kW8MtpInputFirstSmallT)](x, weight, out,
                                                                                   stream);
}
#elif NINFER_W8_SMALL_T_PROFILE == 3
void launch_w8_small_t_profile_3(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream) {
    kMtpAttentionLaunchers[static_cast<std::size_t>(x.ne[1] - kW8MtpAttentionFirstSmallT)](
        x, weight, out, stream);
}
#elif NINFER_W8_SMALL_T_PROFILE == 4
void launch_w8_small_t_profile_4(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream) {
    kMtpAttentionOutputLaunchers[static_cast<std::size_t>(
        x.ne[1] - kW8MtpAttentionOutputFirstSmallT)](x, weight, out, stream);
}
#elif NINFER_W8_SMALL_T_PROFILE == 5
void launch_w8_small_t_profile_5(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream) {
    kMtpGateUpLaunchers[static_cast<std::size_t>(x.ne[1] - kW8MtpGateUpFirstSmallT)](x, weight, out,
                                                                                     stream);
}
#elif NINFER_W8_SMALL_T_PROFILE == 6
void launch_w8_small_t_profile_6(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream) {
    kMtpDownLaunchers[static_cast<std::size_t>(x.ne[1] - kW8MtpDownFirstSmallT)](x, weight, out,
                                                                                 stream);
}
#elif NINFER_W8_SMALL_T_PROFILE == 7
void launch_w8_small_t_profile_7(const Tensor& x, const Weight& weight, Tensor& out,
                                 cudaStream_t stream) {
    k35bMtpProjectionLaunchers[static_cast<std::size_t>(x.ne[1] - kW835bMtpProjectionFirstSmallT)](
        x, weight, out, stream);
}
#else

void launch_w8_small_t_profile_1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t_profile_2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t_profile_3(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t_profile_4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t_profile_5(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t_profile_6(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t_profile_7(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == W8VocabularyProjectionGeometry::kOutputRows &&
        weight.k == W8VocabularyProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8VocabularyProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8VocabularyFirstSmallT && x.ne[1] <= kW8VocabularyLastSmallT) {
        launch_w8_small_t_profile_1(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpInputProjectionGeometry::kOutputRows &&
        weight.k == W8MtpInputProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpInputProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpInputFirstSmallT && x.ne[1] <= kW8MtpInputLastSmallT) {
        launch_w8_small_t_profile_2(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpAttentionProjectionGeometry::kOutputRows &&
        weight.k == W8MtpAttentionProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpAttentionProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpAttentionFirstSmallT && x.ne[1] <= kW8MtpAttentionLastSmallT) {
        launch_w8_small_t_profile_3(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpAttentionOutputGeometry::kOutputRows &&
        weight.k == W8MtpAttentionOutputGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpAttentionOutputGeometry::kInputRows &&
        x.ne[1] >= kW8MtpAttentionOutputFirstSmallT && x.ne[1] <= kW8MtpAttentionOutputLastSmallT) {
        launch_w8_small_t_profile_4(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpGateUpProjectionGeometry::kOutputRows &&
        weight.k == W8MtpGateUpProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpGateUpProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpGateUpFirstSmallT && x.ne[1] <= kW8MtpGateUpLastSmallT) {
        launch_w8_small_t_profile_5(x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpDownProjectionGeometry::kOutputRows &&
        weight.k == W8MtpDownProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpDownProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpDownFirstSmallT && x.ne[1] <= kW8MtpDownLastSmallT) {
        launch_w8_small_t_profile_6(x, weight, out, stream);
        return;
    }
    if (weight.n == W835bMtpProjectionGeometry::kOutputRows &&
        weight.k == W835bMtpProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W835bMtpProjectionGeometry::kInputRows &&
        x.ne[1] >= kW835bMtpProjectionFirstSmallT && x.ne[1] <= kW835bMtpProjectionLastSmallT) {
        launch_w8_small_t_profile_7(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("W8 Linear small-T: unsupported exact problem");
}
#endif

} // namespace ninfer::ops::detail
