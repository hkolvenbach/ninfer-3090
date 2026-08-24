// SHA-256 is the content address of every continuation-cache chunk and image and the identity of
// every artifact, so a wrong digest silently mis-resolves cached state rather than failing loudly.
// The accelerated path is selected at runtime and cannot be observed from the digest alone, which
// is why these cases deliberately cross block boundaries and split updates: an implementation can
// be correct on short inputs and wrong on the multi-block schedule.
#include "artifact/sha256.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::artifact::Sha256;
using ninfer::artifact::Sha256Digest;

std::string hex(const Sha256Digest& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string out(64, '0');
    for (std::size_t i = 0; i != digest.size(); ++i) {
        out[i * 2]     = digits[digest[i] >> 4];
        out[i * 2 + 1] = digits[digest[i] & 15];
    }
    return out;
}

std::string digest_of(std::span<const std::uint8_t> bytes) {
    Sha256 hash;
    hash.update(std::as_bytes(bytes));
    return hex(hash.finish());
}

void expect(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

// Published SHA-256 vectors, including the one-million-character case that exercises the
// multi-block path the short vectors never reach.
void test_published_vectors() {
    expect(digest_of({}) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
           "empty-string digest is wrong");

    const std::string abc = "abc";
    expect(digest_of({reinterpret_cast<const std::uint8_t*>(abc.data()), abc.size()}) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "\"abc\" digest is wrong");

    const std::string two_block =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    expect(digest_of({reinterpret_cast<const std::uint8_t*>(two_block.data()), two_block.size()}) ==
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
           "two-block digest is wrong");

    const std::vector<std::uint8_t> million(1000000, static_cast<std::uint8_t>('a'));
    expect(digest_of(million) ==
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
           "one-million-'a' digest is wrong");
}

// A digest must not depend on how the payload was handed to update(). The cache streams images
// chunk by chunk while the artifact reader hashes whole spans; both must agree.
void test_split_updates_agree() {
    std::mt19937 rng(20260824U);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<std::uint8_t> payload(rng() % 40000U);
        for (auto& byte : payload) { byte = static_cast<std::uint8_t>(rng()); }

        const std::string whole = digest_of(payload);

        Sha256 streamed;
        std::size_t at = 0;
        while (at < payload.size()) {
            const std::size_t take = std::min<std::size_t>(payload.size() - at, 1U + rng() % 997U);
            streamed.update(std::as_bytes(std::span(payload.data() + at, take)));
            at += take;
        }
        expect(hex(streamed.finish()) == whole, "split updates disagree with a single update");
    }
}

// Block-boundary sizes are where a multi-block implementation and its tail handling meet.
void test_block_boundaries() {
    constexpr std::size_t kSizes[] = {0,   1,   55,  56,  57,  63,   64,   65,  119, 120,
                                      127, 128, 129, 191, 192, 255,  256,  8192, 8193};
    for (const std::size_t size : kSizes) {
        std::vector<std::uint8_t> payload(size);
        for (std::size_t i = 0; i < size; ++i) { payload[i] = static_cast<std::uint8_t>(i * 31U); }

        Sha256 once;
        once.update(std::as_bytes(std::span(payload.data(), payload.size())));
        const std::string whole = hex(once.finish());

        for (std::size_t split = 0; split <= size; split += (size / 4) + 1) {
            Sha256 halves;
            halves.update(std::as_bytes(std::span(payload.data(), split)));
            halves.update(std::as_bytes(std::span(payload.data() + split, size - split)));
            expect(hex(halves.finish()) == whole, "split at a block boundary changes the digest");
        }
    }
}

void test_finalized_digest_is_single_use() {
    Sha256 hash;
    hash.update(std::as_bytes(std::span<const std::uint8_t>{}));
    (void)hash.finish();
    bool threw = false;
    try {
        (void)hash.finish();
    } catch (const std::logic_error&) { threw = true; }
    expect(threw, "finishing a finalized digest must fail");
}

} // namespace

int main() {
    try {
        test_published_vectors();
        test_split_updates_agree();
        test_block_boundaries();
        test_finalized_digest_is_single_use();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
