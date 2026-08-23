#include "serve/gpu_telemetry.h"

#include <nvml.h>

namespace ninfer::serve {
namespace {

std::string trimmed(const char* buffer, std::size_t capacity) {
    std::size_t length = 0;
    while (length < capacity && buffer[length] != '\0') { ++length; }
    return std::string(buffer, length);
}

void decode_throttle_reasons(unsigned long long bits, std::vector<std::string>& out) {
    // GpuIdle is a reason the clocks are low, not a limit on useful work, so it is reported like
    // the rest and left to the reader to interpret.
    struct Reason {
        unsigned long long bit;
        const char* name;
    };
    static constexpr Reason kReasons[] = {
        {nvmlClocksEventReasonGpuIdle, "gpu_idle"},
        {nvmlClocksEventReasonApplicationsClocksSetting, "applications_clocks_setting"},
        {nvmlClocksEventReasonSwPowerCap, "sw_power_cap"},
        {nvmlClocksThrottleReasonHwSlowdown, "hw_slowdown"},
        {nvmlClocksEventReasonSyncBoost, "sync_boost"},
        {nvmlClocksEventReasonSwThermalSlowdown, "sw_thermal_slowdown"},
        {nvmlClocksThrottleReasonHwThermalSlowdown, "hw_thermal_slowdown"},
        {nvmlClocksThrottleReasonHwPowerBrakeSlowdown, "hw_power_brake_slowdown"},
        {nvmlClocksEventReasonDisplayClockSetting, "display_clock_setting"},
    };
    for (const Reason& reason : kReasons) {
        if ((bits & reason.bit) != 0ULL) { out.emplace_back(reason.name); }
    }
}

} // namespace

GpuTelemetryReader::GpuTelemetryReader(int device) : device_(device) {
    nvmlReturn_t status = nvmlInit_v2();
    if (status != NVML_SUCCESS) {
        error_ = std::string("nvmlInit_v2: ") + nvmlErrorString(status);
        return;
    }

    nvmlDevice_t handle = nullptr;
    status = nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(device_), &handle);
    if (status != NVML_SUCCESS) {
        error_ = std::string("nvmlDeviceGetHandleByIndex_v2: ") + nvmlErrorString(status);
        nvmlShutdown();
        return;
    }

    handle_      = handle;
    initialized_ = true;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE]             = {};
    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE]             = {};
    char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = {};
    if (nvmlDeviceGetName(handle, name, sizeof(name)) == NVML_SUCCESS) {
        identity_.name = trimmed(name, sizeof(name));
    }
    if (nvmlDeviceGetUUID(handle, uuid, sizeof(uuid)) == NVML_SUCCESS) {
        identity_.uuid = trimmed(uuid, sizeof(uuid));
    }
    if (nvmlSystemGetDriverVersion(driver, sizeof(driver)) == NVML_SUCCESS) {
        identity_.driver_version = trimmed(driver, sizeof(driver));
    }
    nvmlMemory_t memory{};
    if (nvmlDeviceGetMemoryInfo(handle, &memory) == NVML_SUCCESS) {
        identity_.memory_total_bytes = memory.total;
    }
    unsigned int max_sm_clock = 0;
    if (nvmlDeviceGetMaxClockInfo(handle, NVML_CLOCK_SM, &max_sm_clock) == NVML_SUCCESS) {
        identity_.sm_clock_max_mhz = max_sm_clock;
    }
    unsigned int power_limit_mw = 0;
    if (nvmlDeviceGetEnforcedPowerLimit(handle, &power_limit_mw) == NVML_SUCCESS) {
        identity_.power_limit_watts = static_cast<double>(power_limit_mw) / 1000.0;
    }
}

GpuTelemetryReader::~GpuTelemetryReader() {
    if (initialized_) { nvmlShutdown(); }
}

GpuTelemetry GpuTelemetryReader::read() const {
    if (!initialized_) {
        GpuTelemetry telemetry;
        telemetry.available = false;
        telemetry.error     = error_;
        return telemetry;
    }

    GpuTelemetry telemetry = identity_;
    telemetry.available    = true;
    auto handle            = static_cast<nvmlDevice_t>(handle_);

    nvmlTemperature_t temperature{};
    temperature.version    = nvmlTemperature_v1;
    temperature.sensorType = NVML_TEMPERATURE_GPU;
    if (nvmlDeviceGetTemperatureV(handle, &temperature) == NVML_SUCCESS &&
        temperature.temperature > 0) {
        telemetry.temperature_c = static_cast<unsigned int>(temperature.temperature);
    }
    unsigned int fan = 0;
    if (nvmlDeviceGetFanSpeed(handle, &fan) == NVML_SUCCESS) { telemetry.fan_percent = fan; }
    unsigned int power_mw = 0;
    if (nvmlDeviceGetPowerUsage(handle, &power_mw) == NVML_SUCCESS) {
        telemetry.power_watts = static_cast<double>(power_mw) / 1000.0;
    }
    nvmlUtilization_t utilization{};
    if (nvmlDeviceGetUtilizationRates(handle, &utilization) == NVML_SUCCESS) {
        telemetry.utilization_gpu_percent    = utilization.gpu;
        telemetry.utilization_memory_percent = utilization.memory;
    }
    unsigned int sm_clock = 0;
    if (nvmlDeviceGetClockInfo(handle, NVML_CLOCK_SM, &sm_clock) == NVML_SUCCESS) {
        telemetry.sm_clock_mhz = sm_clock;
    }
    unsigned int memory_clock = 0;
    if (nvmlDeviceGetClockInfo(handle, NVML_CLOCK_MEM, &memory_clock) == NVML_SUCCESS) {
        telemetry.memory_clock_mhz = memory_clock;
    }
    nvmlMemory_t memory{};
    if (nvmlDeviceGetMemoryInfo(handle, &memory) == NVML_SUCCESS) {
        telemetry.memory_used_bytes  = memory.used;
        telemetry.memory_total_bytes = memory.total;
    }
    // NVML reports PCIe throughput in KB/s over a driver-chosen sampling window.
    unsigned int pcie_rx_kb = 0;
    if (nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_RX_BYTES, &pcie_rx_kb) == NVML_SUCCESS) {
        telemetry.pcie_rx_bytes_per_second = static_cast<std::uint64_t>(pcie_rx_kb) * 1000ULL;
    }
    unsigned int pcie_tx_kb = 0;
    if (nvmlDeviceGetPcieThroughput(handle, NVML_PCIE_UTIL_TX_BYTES, &pcie_tx_kb) == NVML_SUCCESS) {
        telemetry.pcie_tx_bytes_per_second = static_cast<std::uint64_t>(pcie_tx_kb) * 1000ULL;
    }
    unsigned long long throttle_bits = 0;
    if (nvmlDeviceGetCurrentClocksEventReasons(handle, &throttle_bits) == NVML_SUCCESS) {
        decode_throttle_reasons(throttle_bits, telemetry.throttle_reasons);
    }
    return telemetry;
}

} // namespace ninfer::serve
