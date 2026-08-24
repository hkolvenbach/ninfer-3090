#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::artifact {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
public:
    void update(std::span<const std::byte> bytes);
    [[nodiscard]] Sha256Digest finish();

private:
    void process_block(const std::uint8_t* block);
    void process_blocks(const std::uint8_t* data, std::size_t blocks);

    std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U,
                                        0xa54ff53aU, 0x510e527fU, 0x9b05688cU,
                                        0x1f83d9abU, 0x5be0cd19U};
    std::array<std::uint8_t, 64> pending_{};
    std::size_t pending_size_ = 0;
    std::uint64_t total_bytes_ = 0;
    bool finished_ = false;
};

} // namespace ninfer::artifact
