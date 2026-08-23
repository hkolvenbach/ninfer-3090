#pragma once

// Live NVIDIA board telemetry for GET /telemetry.
//
// NVML ships with the same driver this engine already requires, so it is linked directly rather
// than probed. The handshake can still fail where the driver's management interface is masked
// (some containers), so `available` and `error` are reported as ordinary fields instead of being
// treated as a fatal condition: the dashboard degrades to engine-side panels and says why.
//
// Clock-throttle reasons are decoded because `docs/performance.md` names sustained-prefill
// throttling on a 24 GB RTX 4090 as a measurement hazard - a reading that is easy to misread as
// a kernel regression. Surfacing the reason bits makes that visible while it happens.

#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GpuTelemetry {
    bool available = false;
    std::string error; // set only when available is false

    // Static board identity, read once at construction.
    std::string name;
    std::string uuid;
    std::string driver_version;
    std::uint64_t memory_total_bytes = 0;
    unsigned int sm_clock_max_mhz    = 0;
    double power_limit_watts         = 0.0;

    // Live samples.
    unsigned int temperature_c              = 0;
    unsigned int fan_percent                = 0; // 0 where the board exposes no controllable fan
    double power_watts                      = 0.0;
    unsigned int utilization_gpu_percent    = 0;
    unsigned int utilization_memory_percent = 0;
    unsigned int sm_clock_mhz               = 0;
    unsigned int memory_clock_mhz           = 0;
    std::uint64_t memory_used_bytes         = 0;
    std::uint64_t pcie_rx_bytes_per_second  = 0;
    std::uint64_t pcie_tx_bytes_per_second  = 0;
    std::vector<std::string> throttle_reasons;
};

// Owns the process NVML session and the pinned device handle. Construction performs the whole
// handshake and caches the static identity; read() then costs only the live queries.
class GpuTelemetryReader {
public:
    explicit GpuTelemetryReader(int device);
    ~GpuTelemetryReader();

    GpuTelemetryReader(const GpuTelemetryReader&)            = delete;
    GpuTelemetryReader& operator=(const GpuTelemetryReader&) = delete;

    [[nodiscard]] bool available() const noexcept { return initialized_; }

    // One live sample. Individual queries that the board does not implement leave their field at
    // its default rather than failing the sample.
    [[nodiscard]] GpuTelemetry read() const;

private:
    int device_       = 0;
    bool initialized_ = false;
    std::string error_;
    void* handle_ = nullptr; // nvmlDevice_t, kept opaque so nvml.h stays out of this header
    GpuTelemetry identity_;
};

} // namespace ninfer::serve
