#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer {

struct DeviceToHostTransfer {
    std::vector<std::uint8_t>* destination = nullptr;
    std::size_t destination_offset         = 0;
    const void* source                     = nullptr;
    std::size_t bytes                      = 0;
};

struct HostToDeviceTransfer {
    void* destination                       = nullptr;
    const std::vector<std::uint8_t>* source = nullptr;
    std::size_t source_offset               = 0;
    std::size_t bytes                       = 0;
};

// Fixed-capacity, two-slot pinned pipeline. Pageable owning vectors are touched only by the CPU;
// CUDA transfers use the bounded pinned slots and events protect every slot reuse.
class PinnedTransferBuffer {
public:
    explicit PinnedTransferBuffer(std::size_t capacity_bytes);
    ~PinnedTransferBuffer();

    PinnedTransferBuffer(const PinnedTransferBuffer&)            = delete;
    PinnedTransferBuffer& operator=(const PinnedTransferBuffer&) = delete;
    PinnedTransferBuffer(PinnedTransferBuffer&&) noexcept;
    PinnedTransferBuffer& operator=(PinnedTransferBuffer&&) noexcept;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t slot_capacity() const noexcept;

    void copy_device_to_host(std::span<const DeviceToHostTransfer> transfers,
                             cudaStream_t stream = nullptr);
    void copy_host_to_device(std::span<const HostToDeviceTransfer> transfers,
                             cudaStream_t stream = nullptr);
    void copy_device_to_host(std::vector<std::uint8_t>& destination, const void* source,
                             std::size_t bytes, cudaStream_t stream = nullptr,
                             std::size_t destination_offset = 0);
    void copy_host_to_device(void* destination, const std::vector<std::uint8_t>& source,
                             std::size_t bytes, cudaStream_t stream = nullptr,
                             std::size_t source_offset = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
