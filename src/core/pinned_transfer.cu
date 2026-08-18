#include "core/pinned_transfer.h"

#include "core/arena.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ninfer {
namespace {

std::size_t checked_slot_capacity(std::size_t capacity_bytes) {
    if (capacity_bytes < 2 || capacity_bytes % 2 != 0) {
        throw std::invalid_argument(
            "PinnedTransferBuffer capacity must divide evenly across two slots");
    }
    return capacity_bytes / 2;
}

} // namespace

struct PinnedTransferBuffer::Impl {
    explicit Impl(std::size_t capacity_bytes)
        : slot_bytes(checked_slot_capacity(capacity_bytes)),
          slots{PinnedHostBuffer(slot_bytes), PinnedHostBuffer(slot_bytes)} {
        try {
            for (cudaEvent_t& event : events) {
                CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            }
        } catch (...) {
            for (cudaEvent_t event : events) {
                if (event != nullptr) { (void)cudaEventDestroy(event); }
            }
            throw;
        }
    }

    ~Impl() {
        for (cudaEvent_t event : events) {
            if (event != nullptr) { (void)cudaEventDestroy(event); }
        }
    }

    std::size_t slot_bytes;
    std::array<PinnedHostBuffer, 2> slots;
    std::array<cudaEvent_t, 2> events{};
};

PinnedTransferBuffer::PinnedTransferBuffer(std::size_t capacity_bytes)
    : impl_(std::make_unique<Impl>(capacity_bytes)) {}

PinnedTransferBuffer::~PinnedTransferBuffer() = default;
PinnedTransferBuffer::PinnedTransferBuffer(PinnedTransferBuffer&&) noexcept = default;
PinnedTransferBuffer& PinnedTransferBuffer::operator=(PinnedTransferBuffer&&) noexcept = default;

std::size_t PinnedTransferBuffer::capacity() const noexcept {
    return impl_ == nullptr ? 0 : impl_->slot_bytes * impl_->slots.size();
}

std::size_t PinnedTransferBuffer::slot_capacity() const noexcept {
    return impl_ == nullptr ? 0 : impl_->slot_bytes;
}

void PinnedTransferBuffer::copy_device_to_host(
    std::span<const DeviceToHostTransfer> transfers, cudaStream_t stream) {
    struct Pending {
        std::vector<std::uint8_t>* destination = nullptr;
        std::size_t offset                     = 0;
        std::size_t bytes                      = 0;
        bool active                            = false;
    };
    std::array<Pending, 2> pending{};
    std::size_t next_slot = 0;

    const auto finish = [&](std::size_t slot) {
        if (!pending[slot].active) { return; }
        CUDA_CHECK(cudaEventSynchronize(impl_->events[slot]));
        std::memcpy(pending[slot].destination->data() + pending[slot].offset,
                    impl_->slots[slot].data(), pending[slot].bytes);
        pending[slot].active = false;
    };

    try {
        for (const DeviceToHostTransfer& transfer : transfers) {
            if (transfer.destination == nullptr ||
                (transfer.bytes != 0 && transfer.source == nullptr) ||
                transfer.destination_offset > transfer.destination->size() ||
                transfer.bytes > transfer.destination->size() - transfer.destination_offset) {
                throw std::invalid_argument("device-to-host pinned transfer is out of bounds");
            }
            std::size_t copied = 0;
            while (copied < transfer.bytes) {
                const std::size_t slot = next_slot++ % pending.size();
                finish(slot);
                const std::size_t chunk = std::min(impl_->slot_bytes, transfer.bytes - copied);
                CUDA_CHECK(cudaMemcpyAsync(
                    impl_->slots[slot].data(),
                    static_cast<const std::uint8_t*>(transfer.source) + copied, chunk,
                    cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaEventRecord(impl_->events[slot], stream));
                pending[slot] =
                    {transfer.destination, transfer.destination_offset + copied, chunk, true};
                copied += chunk;
            }
        }
        for (std::size_t slot = 0; slot < pending.size(); ++slot) { finish(slot); }
    } catch (...) {
        // A failed enqueue/record must not let pinned storage die while prior work still references it.
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

void PinnedTransferBuffer::copy_host_to_device(
    std::span<const HostToDeviceTransfer> transfers, cudaStream_t stream) {
    std::array<bool, 2> pending{};
    std::size_t next_slot = 0;

    const auto finish = [&](std::size_t slot) {
        if (!pending[slot]) { return; }
        CUDA_CHECK(cudaEventSynchronize(impl_->events[slot]));
        pending[slot] = false;
    };

    try {
        for (const HostToDeviceTransfer& transfer : transfers) {
            if (transfer.source == nullptr ||
                (transfer.bytes != 0 && transfer.destination == nullptr) ||
                transfer.source_offset > transfer.source->size() ||
                transfer.bytes > transfer.source->size() - transfer.source_offset) {
                throw std::invalid_argument("host-to-device pinned transfer is out of bounds");
            }
            std::size_t copied = 0;
            while (copied < transfer.bytes) {
                const std::size_t slot = next_slot++ % pending.size();
                finish(slot);
                const std::size_t chunk = std::min(impl_->slot_bytes, transfer.bytes - copied);
                std::memcpy(impl_->slots[slot].data(),
                            transfer.source->data() + transfer.source_offset + copied, chunk);
                CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint8_t*>(transfer.destination) + copied,
                                           impl_->slots[slot].data(), chunk, cudaMemcpyHostToDevice,
                                           stream));
                CUDA_CHECK(cudaEventRecord(impl_->events[slot], stream));
                pending[slot] = true;
                copied += chunk;
            }
        }
        for (std::size_t slot = 0; slot < pending.size(); ++slot) { finish(slot); }
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

void PinnedTransferBuffer::copy_device_to_host(std::vector<std::uint8_t>& destination,
                                                const void* source, std::size_t bytes,
                                                cudaStream_t stream,
                                                std::size_t destination_offset) {
    const DeviceToHostTransfer transfer{&destination, destination_offset, source, bytes};
    copy_device_to_host(std::span<const DeviceToHostTransfer>(&transfer, 1), stream);
}

void PinnedTransferBuffer::copy_host_to_device(void* destination,
                                                const std::vector<std::uint8_t>& source,
                                                std::size_t bytes, cudaStream_t stream,
                                                std::size_t source_offset) {
    const HostToDeviceTransfer transfer{destination, &source, source_offset, bytes};
    copy_host_to_device(std::span<const HostToDeviceTransfer>(&transfer, 1), stream);
}

} // namespace ninfer
