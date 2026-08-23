#include "runtime/cache/continuation_cache.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#ifdef _WIN32
#    include <io.h>
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::cache {
namespace {

using Clock = std::chrono::system_clock;
constexpr std::array<char, 8> kImageMagic{'N', 'I', 'C', 'I', 'M', 'G', '0', '3'};
constexpr std::array<char, 8> kManifestMagic{'N', 'I', 'C', 'M', 'A', 'N', '0', '5'};
constexpr std::array<char, 8> kAliasMagic{'N', 'I', 'C', 'A', 'L', 'I', 'A', 'S'};
constexpr std::uint32_t kAliasVersion      = 1;
constexpr std::size_t kMaxAliasBytes       = 1U << 20;
constexpr std::size_t kMaxSessionNameBytes = 64U << 10;
constexpr std::size_t kMaxField                    = 1ULL << 34;
constexpr std::size_t kMaxSelectionMetadataBytes   = 1U << 20;
constexpr std::size_t kPrefixDigestBytes            = 32;
// Chunks stop at segment boundaries, so a manifest carries at most one extra partial chunk per
// segment beyond the fixed-stride count. This bounds that surplus for manifest sizing and
// validation; the target's segment inventory (one per KV plane plus the linear/hidden state) is
// far below it.
constexpr std::size_t kMaxImageSegments             = 4096;

class Sha256 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        total_ += size;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, data, take);
            used_ += take;
            data += take;
            size -= take;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0;
            }
        }
    }

    [[nodiscard]] std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bits = static_cast<std::uint64_t>(total_) * 8;
        block_[used_++]          = 0x80;
        if (used_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_), block_.end(), 0);
            transform(block_.data());
            used_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_), block_.begin() + 56, 0);
        for (int i = 0; i != 8; ++i) block_[63 - i] = static_cast<std::uint8_t>(bits >> (i * 8));
        transform(block_.data());
        std::array<std::uint8_t, 32> out{};
        for (std::size_t i = 0; i != state_.size(); ++i) {
            for (int j = 0; j != 4; ++j)
                out[i * 4 + j] = static_cast<std::uint8_t>(state_[i] >> (24 - j * 8));
        }
        return out;
    }

private:
    static constexpr std::array<std::uint32_t, 64> k{
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};

    void transform(const std::uint8_t* p) {
        std::array<std::uint32_t, 64> w{};
        for (int i = 0; i != 16; ++i)
            w[i] = (std::uint32_t(p[i * 4]) << 24) | (std::uint32_t(p[i * 4 + 1]) << 16) |
                   (std::uint32_t(p[i * 4 + 2]) << 8) | p[i * 4 + 3];
        for (int i = 16; i != 64; ++i) {
            const auto s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const auto s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i]          = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto [a, b, c, d, e, f, g, h] = state_;
        for (int i = 0; i != 64; ++i) {
            const auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const auto t1 = h + s1 + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const auto t2 = s0 + ((a & b) ^ (a & c) ^ (b & c));
            h             = g;
            g             = f;
            f             = e;
            e             = d + t1;
            d             = c;
            c             = b;
            b             = a;
            a             = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> block_{};
    std::size_t used_  = 0;
    std::size_t total_ = 0;
};

std::string sha256(const std::uint8_t* data, std::size_t size) {
    Sha256 hash;
    hash.update(data, size);
    const auto digest       = hash.finish();
    constexpr char digits[] = "0123456789abcdef";
    std::string out(64, '0');
    for (std::size_t i = 0; i != digest.size(); ++i) {
        out[i * 2]     = digits[digest[i] >> 4];
        out[i * 2 + 1] = digits[digest[i] & 15];
    }
    return out;
}

std::string sha256(const Bytes& bytes) { return sha256(bytes.data(), bytes.size()); }

void put_u32(Bytes& out, std::uint32_t value) {
    for (int i = 0; i != 4; ++i) out.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
}

void put_u64(Bytes& out, std::uint64_t value) {
    for (int i = 0; i != 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
}

void put_bytes(Bytes& out, const std::uint8_t* data, std::size_t size) {
    put_u64(out, size);
    out.insert(out.end(), data, data + size);
}

void put_bytes(Bytes& out, const Bytes& value) { put_bytes(out, value.data(), value.size()); }

void put_string(Bytes& out, std::string_view value) {
    put_bytes(out, reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
}

template <class Sink, class RegionSink>
void emit_serialized(const ContinuationImage& image, Sink&& sink, RegionSink&& region) {
    if (image.parent_id && !image.parent_id->valid())
        throw std::invalid_argument("invalid parent ID");
    const auto scalar = [&](std::uint64_t value, std::size_t bytes) {
        std::array<std::uint8_t, 8> encoded{};
        for (std::size_t i = 0; i != bytes; ++i)
            encoded[i] = static_cast<std::uint8_t>(value >> (i * 8));
        sink(encoded.data(), bytes);
    };
    const auto bytes = [&](const std::uint8_t* data, std::size_t size) {
        scalar(size, sizeof(std::uint64_t));
        if (size != 0) sink(data, size);
    };
    const auto string = [&](std::string_view value) {
        bytes(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
    };

    sink(reinterpret_cast<const std::uint8_t*>(kImageMagic.data()), kImageMagic.size());
    scalar(image.format_version, sizeof(std::uint32_t));
    bytes(image.compatibility_key.data(), image.compatibility_key.size());
    bytes(image.prefix_identity.data(), image.prefix_identity.size());
    scalar(image.frontier_tokens, sizeof(std::uint64_t));
    scalar(image.boundary_tokens, sizeof(std::uint64_t));
    bytes(image.frontier_prefix_digest.data(), image.frontier_prefix_digest.size());
    bytes(image.boundary_prefix_digest.data(), image.boundary_prefix_digest.size());
    bytes(image.frontier_metadata.data(), image.frontier_metadata.size());
    bytes(image.boundary_metadata.data(), image.boundary_metadata.size());
    scalar(image.parent_id.has_value(), 1);
    if (image.parent_id) string(image.parent_id->hex);
    scalar(image.segments.size(), sizeof(std::uint64_t));
    for (const auto& [name, value] : image.segments) {
        if (name.empty()) throw std::invalid_argument("empty segment name");
        string(name);
        scalar(value.size(), sizeof(std::uint64_t));
        region();
        if (!value.empty()) sink(value.data(), value.size());
    }
}

template <class Sink>
void emit_serialized(const ContinuationImage& image, Sink&& sink) {
    emit_serialized(image, std::forward<Sink>(sink), [] {});
}

class Reader {
public:
    explicit Reader(const Bytes& bytes) : bytes_(bytes) {}

    std::uint8_t byte() {
        need(1);
        return bytes_[at_++];
    }

    std::uint32_t u32() {
        need(4);
        std::uint32_t v = 0;
        for (int i = 0; i != 4; ++i) v |= std::uint32_t(bytes_[at_++]) << (i * 8);
        return v;
    }

    std::uint64_t u64() {
        need(8);
        std::uint64_t v = 0;
        for (int i = 0; i != 8; ++i) v |= std::uint64_t(bytes_[at_++]) << (i * 8);
        return v;
    }

    Bytes bytes() {
        const auto n = u64();
        if (n > kMaxField || n > bytes_.size() - at_)
            throw std::runtime_error("invalid field size");
        Bytes value(bytes_.begin() + static_cast<std::ptrdiff_t>(at_),
                    bytes_.begin() + static_cast<std::ptrdiff_t>(at_ + n));
        at_ += static_cast<std::size_t>(n);
        return value;
    }

    std::string string() {
        auto value = bytes();
        return {reinterpret_cast<const char*>(value.data()), value.size()};
    }

    void magic(const std::array<char, 8>& expected) {
        need(expected.size());
        if (!std::equal(expected.begin(), expected.end(), bytes_.begin() + at_))
            throw std::runtime_error("invalid magic");
        at_ += expected.size();
    }

    void done() const {
        if (at_ != bytes_.size()) throw std::runtime_error("trailing bytes");
    }
private:
    void need(std::size_t n) const {
        if (n > bytes_.size() - at_) throw std::runtime_error("truncated data");
    }

    const Bytes& bytes_;
    std::size_t at_ = 0;
};

Bytes serialize(const ContinuationImage& image) {
    Bytes out;
    out.reserve(continuation_image_bytes(image));
    emit_serialized(image, [&](const std::uint8_t* data, std::size_t size) {
        out.insert(out.end(), data, data + size);
    });
    return out;
}

// The serialized image plus the offset of every segment payload. L3 chunks never span a segment,
// so a segment's chunks depend only on that segment's own bytes. This is what makes an append-only
// payload share every chunk below its previous length with the image published one turn earlier;
// chunking the blob as one run instead ties every boundary to the length of everything before it,
// and the growing prefix identity then moves all of them on every turn.
struct SerializedImage {
    Bytes payload;
    std::vector<std::size_t> region_starts;
};

SerializedImage serialize_with_regions(const ContinuationImage& image) {
    SerializedImage out;
    out.payload.reserve(continuation_image_bytes(image));
    out.region_starts.reserve(image.segments.size() + 1);
    emit_serialized(
        image,
        [&](const std::uint8_t* data, std::size_t size) {
            out.payload.insert(out.payload.end(), data, data + size);
        },
        [&] { out.region_starts.push_back(out.payload.size()); });
    return out;
}

ContentId hash_serialized(const ContinuationImage& image) {
    Sha256 hash;
    emit_serialized(image, [&](const std::uint8_t* data, std::size_t size) {
        hash.update(data, size);
    });
    const auto digest = hash.finish();
    constexpr char digits[] = "0123456789abcdef";
    std::string hex(64, '0');
    for (std::size_t i = 0; i != digest.size(); ++i) {
        hex[i * 2]     = digits[digest[i] >> 4];
        hex[i * 2 + 1] = digits[digest[i] & 15];
    }
    return {std::move(hex)};
}

ContinuationImage deserialize(const Bytes& bytes) {
    Reader r(bytes);
    r.magic(kImageMagic);
    ContinuationImage image;
    image.format_version    = r.u32();
    image.compatibility_key = r.bytes();
    image.prefix_identity   = r.bytes();
    image.frontier_tokens   = r.u64();
    image.boundary_tokens   = r.u64();
    image.frontier_prefix_digest = r.bytes();
    image.boundary_prefix_digest = r.bytes();
    image.frontier_metadata = r.bytes();
    image.boundary_metadata = r.bytes();
    const auto parent       = r.byte();
    if (parent > 1) throw std::runtime_error("invalid parent flag");
    if (parent != 0) image.parent_id = ContentId{r.string()};
    const auto count = r.u64();
    if (count > 1'000'000) throw std::runtime_error("too many segments");
    for (std::uint64_t i = 0; i != count; ++i) {
        auto name = r.string();
        if (name.empty() || !image.segments.emplace(std::move(name), r.bytes()).second)
            throw std::runtime_error("invalid segment name");
    }
    r.done();
    return image;
}

void ensure_private_dir(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) throw std::runtime_error("cannot create cache directory: " + ec.message());
#ifndef _WIN32
    if (::chmod(path.c_str(), 0700) != 0) throw std::runtime_error("cannot secure cache directory");
#endif
}

void sync_directory(const std::filesystem::path& path) {
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (fd >= 0) {
        (void)::fsync(fd);
        (void)::close(fd);
    }
#else
    (void)path;
#endif
}

enum class AtomicWriteMode { CreateIfAbsent, Replace };

void write_atomic_impl(const std::filesystem::path& final, const Bytes& bytes,
                       const std::filesystem::path& temp_dir, AtomicWriteMode mode,
                       const std::function<void(const std::filesystem::path&)>& before_replace) {
    static std::atomic<std::uint64_t> sequence = 0;
#ifdef _WIN32
    const auto process = static_cast<unsigned long long>(GetCurrentProcessId());
#else
    const auto process = static_cast<unsigned long long>(::getpid());
#endif
    const auto temp =
        temp_dir / (final.filename().string() + ".tmp." + std::to_string(process) + "." +
                    std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
#ifdef _WIN32
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    if (!out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("cache write failed");
    out.close();
    HANDLE h = CreateFileW(temp.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(h);
        CloseHandle(h);
    }
#else
    const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) throw std::runtime_error("cache temp create failed");
    std::size_t done = 0;
    while (done != bytes.size()) {
        const auto n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n <= 0) {
            ::close(fd);
            std::filesystem::remove(temp);
            throw std::runtime_error("cache write failed");
        }
        done += static_cast<std::size_t>(n);
    }
    if (::fsync(fd) != 0 || ::close(fd) != 0) {
        std::filesystem::remove(temp);
        throw std::runtime_error("cache fsync failed");
    }
#endif
    if (mode == AtomicWriteMode::Replace && before_replace) {
        try {
            before_replace(final);
        } catch (...) {
            std::error_code remove_ec;
            std::filesystem::remove(temp, remove_ec);
            throw;
        }
    }
    std::error_code ec;
#ifdef _WIN32
    const DWORD flags = MOVEFILE_WRITE_THROUGH |
                        (mode == AtomicWriteMode::Replace ? MOVEFILE_REPLACE_EXISTING : 0);
    if (!MoveFileExW(temp.wstring().c_str(), final.wstring().c_str(), flags)) {
        const DWORD error = GetLastError();
        if (mode != AtomicWriteMode::CreateIfAbsent ||
            (error != ERROR_ALREADY_EXISTS && error != ERROR_FILE_EXISTS)) {
            ec = std::error_code(static_cast<int>(error), std::system_category());
        }
    }
#else
    if (mode == AtomicWriteMode::CreateIfAbsent) {
        if (::link(temp.c_str(), final.c_str()) != 0 && errno != EEXIST) {
            ec = std::error_code(errno, std::generic_category());
        }
        std::error_code remove_ec;
        std::filesystem::remove(temp, remove_ec);
    } else {
        std::filesystem::rename(temp, final, ec);
    }
#endif
    if (ec) {
        std::error_code remove_ec;
        std::filesystem::remove(temp, remove_ec);
        throw std::runtime_error("cache publish failed: " + ec.message());
    }
#ifdef _WIN32
    if (mode == AtomicWriteMode::CreateIfAbsent && std::filesystem::exists(temp)) {
        std::error_code remove_ec;
        std::filesystem::remove(temp, remove_ec);
    }
#endif
    sync_directory(final.parent_path());
}

void write_atomic_create_if_absent(const std::filesystem::path& final, const Bytes& bytes,
                                   const std::filesystem::path& temp_dir) {
    write_atomic_impl(final, bytes, temp_dir, AtomicWriteMode::CreateIfAbsent, {});
}

void write_atomic_replace(
    const std::filesystem::path& final, const Bytes& bytes,
    const std::filesystem::path& temp_dir,
    const std::function<void(const std::filesystem::path&)>& before_replace = {}) {
    write_atomic_impl(final, bytes, temp_dir, AtomicWriteMode::Replace, before_replace);
}

Bytes read_file(const std::filesystem::path& path, std::size_t expected = 0,
                std::size_t maximum = kMaxField) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size > maximum || size > std::numeric_limits<std::size_t>::max() ||
        (expected != 0 && size != expected))
        throw std::runtime_error("invalid cache file size");
    Bytes bytes(static_cast<std::size_t>(size));
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size())) ||
        in.peek() != EOF)
        throw std::runtime_error("cache file read failed");
    return bytes;
}

void read_chunk(const std::filesystem::path& path, std::uint8_t* destination,
                std::size_t expected, std::size_t maximum, std::string_view expected_hash) {
    if (expected == 0 || expected > maximum) throw std::runtime_error("invalid chunk size");
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec || size != expected || size > maximum) throw std::runtime_error("invalid chunk size");
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(expected)) ||
        in.peek() != EOF || sha256(destination, expected) != expected_hash) {
        throw std::runtime_error("corrupt chunk");
    }
}

struct Chunk {
    std::string id;
    std::uint64_t size = 0;
};

struct Manifest {
    std::uint64_t image_bytes = 0;
    std::uint32_t image_format_version = 0;
    Bytes compatibility_key;
    std::uint64_t frontier_tokens = 0;
    std::uint64_t boundary_tokens = 0;
    Bytes frontier_prefix_digest;
    Bytes boundary_prefix_digest;
    std::uint64_t l2_idle_ttl_seconds = 0;
    std::uint64_t l3_idle_ttl_seconds = 0;
    std::int64_t expires_ms   = 0;
    double cost               = 1;
    std::vector<Chunk> chunks;
    std::size_t manifest_bytes = 0;
    double score               = 0;
    bool durable               = false;
};

bool metadata_matches(const Manifest& manifest, const ContinuationImage& image) {
    return manifest.image_bytes == continuation_image_bytes(image) &&
           manifest.image_format_version == image.format_version &&
           manifest.compatibility_key == image.compatibility_key &&
           manifest.frontier_tokens == image.frontier_tokens &&
           manifest.boundary_tokens == image.boundary_tokens &&
           manifest.frontier_prefix_digest == image.frontier_prefix_digest &&
           manifest.boundary_prefix_digest == image.boundary_prefix_digest;
}

bool metadata_matches(const Manifest& manifest,
                      const SessionCandidateDescriptor& descriptor) {
    return manifest.image_bytes == descriptor.image_bytes &&
           manifest.image_format_version == descriptor.image_format_version &&
           manifest.compatibility_key == descriptor.compatibility_key &&
           manifest.frontier_tokens == descriptor.frontier_tokens &&
           manifest.boundary_tokens == descriptor.boundary_tokens &&
           manifest.frontier_prefix_digest == descriptor.frontier_prefix_digest &&
           manifest.boundary_prefix_digest == descriptor.boundary_prefix_digest;
}

Bytes encode_manifest(const Manifest& manifest) {
    Bytes out(kManifestMagic.begin(), kManifestMagic.end());
    put_u64(out, manifest.image_bytes);
    put_u32(out, manifest.image_format_version);
    put_bytes(out, manifest.compatibility_key);
    put_u64(out, manifest.frontier_tokens);
    put_u64(out, manifest.boundary_tokens);
    put_bytes(out, manifest.frontier_prefix_digest);
    put_bytes(out, manifest.boundary_prefix_digest);
    put_u64(out, manifest.l2_idle_ttl_seconds);
    put_u64(out, manifest.l3_idle_ttl_seconds);
    put_u64(out, static_cast<std::uint64_t>(manifest.expires_ms));
    put_u64(out, std::bit_cast<std::uint64_t>(manifest.cost));
    put_u64(out, manifest.chunks.size());
    for (const auto& chunk : manifest.chunks) {
        put_string(out, chunk.id);
        put_u64(out, chunk.size);
    }
    return out;
}

std::size_t max_chunk_count(std::size_t image_limit, std::size_t chunk_limit) {
    const std::size_t stride = image_limit / chunk_limit + (image_limit % chunk_limit != 0);
    return stride > std::numeric_limits<std::size_t>::max() - kMaxImageSegments
               ? std::numeric_limits<std::size_t>::max()
               : stride + kMaxImageSegments;
}

Manifest decode_manifest(const Bytes& bytes, std::size_t image_limit, std::size_t chunk_limit) {
    Reader r(bytes);
    r.magic(kManifestMagic);
    Manifest m;
    m.image_bytes    = r.u64();
    m.image_format_version = r.u32();
    m.compatibility_key = r.bytes();
    m.frontier_tokens = r.u64();
    m.boundary_tokens = r.u64();
    m.frontier_prefix_digest = r.bytes();
    m.boundary_prefix_digest = r.bytes();
    m.l2_idle_ttl_seconds = r.u64();
    m.l3_idle_ttl_seconds = r.u64();
    m.expires_ms     = static_cast<std::int64_t>(r.u64());
    m.cost           = std::bit_cast<double>(r.u64());
    const auto count = r.u64();
    if (m.image_bytes == 0 || m.image_bytes > image_limit || chunk_limit == 0 ||
        m.image_format_version == 0 || m.compatibility_key.empty() ||
        m.compatibility_key.size() > kMaxSelectionMetadataBytes ||
        m.frontier_tokens == 0 || m.frontier_prefix_digest.size() != kPrefixDigestBytes ||
        ((m.boundary_tokens == 0) != m.boundary_prefix_digest.empty()) ||
        (!m.boundary_prefix_digest.empty() &&
         m.boundary_prefix_digest.size() != kPrefixDigestBytes) ||
        m.boundary_tokens > m.frontier_tokens ||
        // Chunks stop at segment boundaries so a segment's chunks depend only on its own bytes.
        // The count is therefore not the fixed-stride count; what must hold is that every chunk
        // is non-empty, none exceeds the configured chunk size, and together they cover the image
        // exactly. One extra partial chunk per segment is the most the layout can add.
        count == 0 || count > max_chunk_count(image_limit, chunk_limit) ||
        m.l2_idle_ttl_seconds >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        m.l3_idle_ttl_seconds >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        !std::isfinite(m.cost) || m.cost < 0)
        throw std::runtime_error("invalid manifest");
    std::uint64_t total = 0;
    for (std::uint64_t i = 0; i != count; ++i) {
        Chunk c{r.string(), r.u64()};
        if (!ContentId{c.id}.valid() || c.size == 0 || c.size > chunk_limit ||
            total > kMaxField - c.size || total + c.size > m.image_bytes)
            throw std::runtime_error("invalid chunk");
        total += c.size;
        m.chunks.push_back(std::move(c));
    }
    if (total != m.image_bytes || (count == 0 && m.image_bytes != 0))
        throw std::runtime_error("invalid image size");
    r.done();
    m.manifest_bytes = bytes.size();
    return m;
}

struct Alias {
    std::string session;
    std::vector<ContentId> history;
};

Bytes encode_alias(std::string_view session, const std::vector<ContentId>& history) {
    if (session.size() > kMaxSessionNameBytes) throw std::invalid_argument("session name too long");
    Bytes out(kAliasMagic.begin(), kAliasMagic.end());
    put_u32(out, kAliasVersion);
    put_string(out, session);
    put_u64(out, history.size());
    for (const auto& id : history) {
        if (!id.valid()) throw std::invalid_argument("invalid alias content ID");
        put_string(out, id.hex);
    }
    if (out.size() > kMaxAliasBytes) throw std::invalid_argument("alias too large");
    return out;
}

Alias decode_alias(const Bytes& bytes, std::size_t history_depth) {
    if (bytes.size() > kMaxAliasBytes) throw std::runtime_error("alias too large");
    Reader r(bytes);
    r.magic(kAliasMagic);
    if (r.u32() != kAliasVersion) throw std::runtime_error("unsupported alias version");
    Alias alias{r.string(), {}};
    if (alias.session.size() > kMaxSessionNameBytes)
        throw std::runtime_error("session name too long");
    const auto count = r.u64();
    if (count == 0 || count > history_depth) throw std::runtime_error("invalid alias history size");
    alias.history.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i != count; ++i) {
        ContentId id{r.string()};
        if (!id.valid()) throw std::runtime_error("invalid alias content ID");
        alias.history.push_back(std::move(id));
    }
    r.done();
    return alias;
}

std::int64_t millis(Clock::time_point point) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(point.time_since_epoch()).count();
}

std::int64_t expiration_ms(Clock::time_point now, std::chrono::seconds ttl) {
    if (ttl <= std::chrono::seconds::zero()) return 0;
    constexpr auto max = std::numeric_limits<std::int64_t>::max();
    if (ttl.count() > max / 1000) return max;
    const auto delta = ttl.count() * 1000;
    const auto base  = millis(now);
    return base > max - delta ? max : base + delta;
}

} // namespace

bool ContentId::valid() const {
    return hex.size() == 64 && std::all_of(hex.begin(), hex.end(), [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           });
}

ContentId content_id(const ContinuationImage& image) {
    return hash_serialized(image);
}

std::size_t continuation_image_bytes(const ContinuationImage& image) {
    if (image.parent_id && !image.parent_id->valid()) {
        throw std::invalid_argument("invalid parent ID");
    }
    constexpr auto add = [](std::size_t& total, std::size_t value) {
        if (value > std::numeric_limits<std::size_t>::max() - total) {
            throw std::length_error("continuation image size overflow");
        }
        total += value;
    };
    std::size_t total = kImageMagic.size() + sizeof(std::uint32_t);
    const auto add_bytes = [&](std::size_t size) {
        add(total, sizeof(std::uint64_t));
        add(total, size);
    };
    add_bytes(image.compatibility_key.size());
    add_bytes(image.prefix_identity.size());
    add(total, sizeof(std::uint64_t) * 2);
    add_bytes(image.frontier_prefix_digest.size());
    add_bytes(image.boundary_prefix_digest.size());
    add_bytes(image.frontier_metadata.size());
    add_bytes(image.boundary_metadata.size());
    add(total, 1);
    if (image.parent_id) add_bytes(image.parent_id->hex.size());
    add(total, sizeof(std::uint64_t));
    for (const auto& [name, bytes] : image.segments) {
        if (name.empty()) throw std::invalid_argument("empty segment name");
        add_bytes(name.size());
        add_bytes(bytes.size());
    }
    return total;
}

bool persistence_due(const PersistencePolicy& policy,
                     const std::optional<PersistedState>& previous,
                     std::uint64_t candidate_tokens, Clock::time_point queued_at,
                     Clock::time_point now) {
    const std::uint64_t prior_tokens = previous ? previous->tokens : 0;
    const std::uint64_t growth =
        candidate_tokens >= prior_tokens ? candidate_tokens - prior_tokens : 0;
    const bool token_due = policy.min_tokens == 0 || growth >= policy.min_tokens;
    const Clock::time_point interval_base = previous ? previous->at : queued_at;
    const bool interval_due = policy.interval > Clock::duration::zero() &&
                              now - interval_base >= policy.interval;
    return token_due || interval_due;
}

class ContinuationCache::Impl : public std::enable_shared_from_this<ContinuationCache::Impl> {
public:
    explicit Impl(CacheConfig config) : config_(std::move(config)) {
        if (config_.chunk_bytes == 0 || config_.session_history_depth == 0 || !config_.now ||
            config_.l2_idle_ttl < std::chrono::seconds::zero() ||
            config_.l3_idle_ttl < std::chrono::seconds::zero() ||
            config_.persist_interval < std::chrono::seconds::zero() ||
            (l3_enabled() && config_.root.empty()))
            throw std::invalid_argument("invalid continuation cache config");
        if (!l3_enabled()) return;
        chunks_    = config_.root / "chunks";
        manifests_ = config_.root / "manifests";
        aliases_   = config_.root / "aliases";
        temp_      = config_.root / "tmp";
        ensure_private_dir(config_.root);
        ensure_private_dir(chunks_);
        ensure_private_dir(manifests_);
        ensure_private_dir(aliases_);
        ensure_private_dir(temp_);
        discover();
        cleanup_orphans();
        discover_aliases();
        evict_l3();
        persistence_worker_ = std::thread([this] { persistence_loop(); });
    }

    ~Impl() {
        {
            std::lock_guard lock(mu_);
            persistence_stopping_ = true;
        }
        persistence_cv_.notify_all();
        if (persistence_worker_.joinable()) persistence_worker_.join();
    }

    SessionPublishResult store_impl(std::shared_ptr<const ContinuationImage> image,
                                     const StoreOptions& options,
                                     bool persist_l3,
                                     std::optional<std::string_view> cas_session   = std::nullopt,
                                    const std::optional<ContentId>& expected_head = std::nullopt,
                                    std::optional<std::uint64_t> expected_generation =
                                         std::nullopt) {
        const auto admission_started = std::chrono::steady_clock::now();
        if (cas_session && (cas_session->empty() || cas_session->starts_with(kStableAliasPrefix) ||
                            (expected_head && !expected_head->valid()))) {
            throw std::invalid_argument("invalid session publication");
        }
        const ContentId id = hash_serialized(*image);
        const std::size_t image_bytes = continuation_image_bytes(*image);
        if (image->format_version == 0 || image->compatibility_key.empty() ||
            image->compatibility_key.size() > kMaxSelectionMetadataBytes ||
            image->frontier_tokens == 0 ||
            image->frontier_prefix_digest.size() != kPrefixDigestBytes ||
            ((image->boundary_tokens == 0) != image->boundary_prefix_digest.empty()) ||
            (!image->boundary_prefix_digest.empty() &&
             image->boundary_prefix_digest.size() != kPrefixDigestBytes) ||
            image->boundary_tokens > image->frontier_tokens) {
            throw std::invalid_argument("invalid continuation selection metadata");
        }
        const auto l2_ttl = options.l2_idle_ttl.value_or(config_.l2_idle_ttl);
        const auto l3_ttl = options.l3_idle_ttl.value_or(config_.l3_idle_ttl);
        if (l2_ttl < std::chrono::seconds::zero() || l3_ttl < std::chrono::seconds::zero()) {
            throw std::invalid_argument("continuation cache TTL must be nonnegative");
        }
        const auto now = config_.now();
        std::lock_guard lock(mu_);
        auto found = catalog_.find(id.hex);
        if (found != catalog_.end()) {
            expire_l2(found->first);
            if (found->second.durable && expired_l3(found->second)) {
                expire_l3(found);
                found = catalog_.find(id.hex);
            }
            if (found != catalog_.end() && !found->second.durable && !l2_.contains(id.hex)) {
                remove_from_sessions(id.hex);
                catalog_.erase(found);
                found = catalog_.end();
            }
        }
        if (found == catalog_.end()) {
            Manifest manifest;
            manifest.image_bytes = image_bytes;
            manifest.image_format_version = image->format_version;
            manifest.compatibility_key = image->compatibility_key;
            manifest.frontier_tokens = image->frontier_tokens;
            manifest.boundary_tokens = image->boundary_tokens;
            manifest.frontier_prefix_digest = image->frontier_prefix_digest;
            manifest.boundary_prefix_digest = image->boundary_prefix_digest;
            manifest.l2_idle_ttl_seconds = static_cast<std::uint64_t>(l2_ttl.count());
            manifest.l3_idle_ttl_seconds = static_cast<std::uint64_t>(l3_ttl.count());
            manifest.expires_ms  = expiration_ms(now, l3_ttl);
            manifest.cost        = std::max(0.0, options.recompute_cost);
            add_manifest(id.hex, std::move(manifest));
            found = catalog_.find(id.hex);
        } else if (persist_l3) {
            found->second.l2_idle_ttl_seconds = static_cast<std::uint64_t>(l2_ttl.count());
            found->second.l3_idle_ttl_seconds = static_cast<std::uint64_t>(l3_ttl.count());
            found->second.expires_ms = expiration_ms(now, l3_ttl);
        }
        const bool too_large_for_l2 =
            config_.l2_byte_budget == 0 || image_bytes > config_.l2_byte_budget;
        touch_l2(id.hex, std::move(image), image_bytes, l2_ttl);
        if (found != catalog_.end()) touch(found->second);
        const bool entered_l2 = l2_.contains(id.hex);
        // A session publication is the newest state of a live conversation, so it must not be the
        // victim of the eviction pass its own admission triggers: losing it forfeits the alias
        // advance and forces the next turn to prefill from nothing.
        evict_l2(cas_session ? std::string_view(id.hex) : std::string_view());
        evict_l3();
        found = catalog_.find(id.hex);
        const bool stored =
            found != catalog_.end() &&
            (found->second.durable || l2_.contains(id.hex) || (persist_l3 && l3_enabled()));
        bool alias_advanced = false;
        SessionPublishOutcome outcome = SessionPublishOutcome::Advanced;
        if (!stored) {
            outcome = too_large_for_l2 ? SessionPublishOutcome::RejectedTooLarge
                      : entered_l2     ? SessionPublishOutcome::EvictedOnAdmission
                                       : SessionPublishOutcome::RejectedTooLarge;
        }
        if (stored) {
            if (cas_session) {
                const auto existing = sessions_.find(std::string(*cas_session));
                const std::optional<ContentId> current =
                    existing == sessions_.end() || existing->second.empty()
                        ? std::nullopt
                        : std::optional<ContentId>(existing->second.back());
                const auto generation = alias_generations_.find(std::string(*cas_session));
                const std::uint64_t current_generation =
                    generation == alias_generations_.end() ? 0 : generation->second;
                if (current != expected_head) {
                    outcome = SessionPublishOutcome::HeadMoved;
                } else if (expected_generation && *expected_generation != current_generation) {
                    outcome = SessionPublishOutcome::GenerationMoved;
                } else {
                    update_session(std::string(*cas_session), id, persist_l3);
                    alias_advanced = true;
                }
            } else if (options.session) {
                update_session(*options.session, id, persist_l3);
            }
        } else if (found != catalog_.end()) {
            catalog_.erase(found);
        }
        ++l2_admission_operations_;
        l2_admission_microseconds_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - admission_started)
                .count());
        return {.id             = id,
                .stored         = stored,
                .alias_advanced = alias_advanced,
                .outcome        = outcome};
    }

    ContentId store(const ContinuationImage& image, const StoreOptions& options) {
        auto shared       = std::make_shared<const ContinuationImage>(image);
        const auto result = store_impl(shared, options, true);
        if (!promote_image(result.id, std::move(shared))) {
            discard_unstored(result.id.hex);
        }
        persist_related_aliases(result.id.hex);
        return result.id;
    }

    ContentId admit(const ContinuationImage& image, const StoreOptions& options) {
        return store_impl(std::make_shared<const ContinuationImage>(image), options, false).id;
    }

    ContentId admit(ContinuationImage&& image, const StoreOptions& options) {
        return store_impl(std::make_shared<const ContinuationImage>(std::move(image)), options,
                          false)
            .id;
    }

    SessionPublishResult publish_session(const ContinuationImage& image, std::string_view session,
                                         const std::optional<ContentId>& expected_head,
                                         const StoreOptions& options,
                                         std::optional<std::uint64_t> expected_generation) {
        if (image.parent_id != expected_head) {
            throw std::invalid_argument("session publication parent does not match expected head");
        }
        StoreOptions without_session = options;
        without_session.session.reset();
        auto shared = std::make_shared<const ContinuationImage>(image);
        auto result = store_impl(shared, without_session, true, session, expected_head,
                                 expected_generation);
        if (!promote_image(result.id, std::move(shared))) {
            if (discard_unstored(result.id.hex)) {
                result.stored = false;
                result.alias_advanced = false;
            }
        }
        persist_related_aliases(result.id.hex);
        return result;
    }

    SessionPublishResult publish_session_l2(const ContinuationImage& image, std::string_view session,
                                             const std::optional<ContentId>& expected_head,
                                             const StoreOptions& options,
                                             std::optional<std::uint64_t> expected_generation) {
        if (image.parent_id != expected_head) {
            throw std::invalid_argument("session publication parent does not match expected head");
        }
        StoreOptions without_session = options;
        without_session.session.reset();
        return store_impl(std::make_shared<const ContinuationImage>(image), without_session, false,
                          session, expected_head, expected_generation);
    }

    SessionPublishResult publish_session_l2(
        ContinuationImage&& image, std::string_view session,
        const std::optional<ContentId>& expected_head, const StoreOptions& options,
        std::optional<std::uint64_t> expected_generation) {
        if (image.parent_id != expected_head) {
            throw std::invalid_argument("session publication parent does not match expected head");
        }
        StoreOptions without_session = options;
        without_session.session.reset();
        return store_impl(std::make_shared<const ContinuationImage>(std::move(image)),
                          without_session, false, session, expected_head, expected_generation);
    }

    CacheLookupResult lookup_shared(const ContentId& id) {
        const auto started = std::chrono::steady_clock::now();
        const auto elapsed = [&] {
            return static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started)
                    .count());
        };
        if (!id.valid()) return {.status = CacheLookupStatus::Absent};
        Manifest snapshot;
        {
            std::lock_guard lock(mu_);
            auto found = catalog_.find(id.hex);
            if (found == catalog_.end()) return {.status = CacheLookupStatus::Absent};
            expire_l2(id.hex);
            if (found->second.durable && expired_l3(found->second)) {
                expire_l3(found);
                found = catalog_.find(id.hex);
                if (found == catalog_.end()) {
                    return {.source = CacheSource::L3,
                            .status = CacheLookupStatus::UnavailableOrCorrupt,
                            .io_microseconds = elapsed()};
                }
            }
            if (auto hot = l2_.find(id.hex); hot != l2_.end()) {
                hot->second.recency = ++tick_;
                hot->second.score = std::max(hot->second.score, l2_inflation_) +
                                    found->second.cost /
                                        std::max<std::size_t>(1, hot->second.bytes) +
                                    (tick_ * 1e-15);
                hot->second.expires_ms = expiration_ms(config_.now(), hot->second.idle_ttl);
                touch(found->second);
                return {.image = lease_locked(id.hex, hot->second.image),
                        .source = CacheSource::L2,
                        .status = CacheLookupStatus::Hit,
                        .io_microseconds = elapsed()};
            }
            if (!l3_available(id.hex, found->second)) {
                return {.source = CacheSource::L3,
                        .status = CacheLookupStatus::UnavailableOrCorrupt,
                        .io_microseconds = elapsed()};
            }
            snapshot = found->second;
            ++pins_[id.hex];
        }

        std::shared_ptr<const ContinuationImage> restored;
        bool refreshed = false;
        try {
            if (config_.before_l3_restore_io) config_.before_l3_restore_io();
            Bytes payload(static_cast<std::size_t>(snapshot.image_bytes));
            std::size_t at = 0;
            Sha256 image_hash;
            for (const auto& chunk : snapshot.chunks) {
                read_chunk(chunks_ / chunk.id, payload.data() + at,
                           static_cast<std::size_t>(chunk.size), config_.chunk_bytes, chunk.id);
                image_hash.update(payload.data() + at, static_cast<std::size_t>(chunk.size));
                at += static_cast<std::size_t>(chunk.size);
            }
            const auto digest = image_hash.finish();
            constexpr char digits[] = "0123456789abcdef";
            std::string hash(64, '0');
            for (std::size_t i = 0; i != digest.size(); ++i) {
                hash[i * 2] = digits[digest[i] >> 4];
                hash[i * 2 + 1] = digits[digest[i] & 15];
            }
            if (at != payload.size() || hash != id.hex) throw std::runtime_error("corrupt image");
            auto image = deserialize(payload);
            if (!metadata_matches(snapshot, image)) {
                throw std::runtime_error("manifest selection metadata mismatch");
            }
            restored = std::make_shared<const ContinuationImage>(std::move(image));
            if (snapshot.l3_idle_ttl_seconds != 0) {
                snapshot.expires_ms = expiration_ms(
                    config_.now(), std::chrono::seconds(snapshot.l3_idle_ttl_seconds));
                write_atomic_replace(manifests_ / (id.hex + ".manifest"),
                                     encode_manifest(snapshot), temp_,
                                     config_.before_mutable_replace);
                refreshed = true;
            }
        } catch (const std::bad_alloc&) {
        } catch (const std::length_error&) {
        } catch (...) {}

        std::lock_guard lock(mu_);
        auto found = catalog_.find(id.hex);
        if (!restored || found == catalog_.end() || !found->second.durable) {
            if (!restored && found != catalog_.end() && found->second.durable) expire_l3(found, true);
            unpin_locked(id.hex);
            evict_l2();
            return {.source = CacheSource::L3,
                    .status = CacheLookupStatus::UnavailableOrCorrupt,
                    .io_microseconds = elapsed()};
        }
        if (pending_l3_drops_.contains(id.hex)) {
            return {.image = lease_locked(id.hex, std::move(restored), true),
                    .source = CacheSource::L3,
                    .status = CacheLookupStatus::Hit,
                    .io_microseconds = elapsed()};
        }
        if (refreshed) found->second.expires_ms = snapshot.expires_ms;
        touch_l2(id.hex, restored, static_cast<std::size_t>(snapshot.image_bytes),
                 std::chrono::seconds(snapshot.l2_idle_ttl_seconds));
        touch(found->second);
        evict_l2();
        return {.image = lease_locked(id.hex, std::move(restored), true),
                .source = CacheSource::L3,
                .status = CacheLookupStatus::Hit,
                .io_microseconds = elapsed()};
    }

    std::shared_ptr<const ContinuationImage> get_shared(const ContentId& id) {
        return lookup_shared(id).image;
    }

    std::optional<ContinuationImage> get(const ContentId& id) {
        auto shared = get_shared(id);
        if (!shared) return std::nullopt;
        return *shared;
    }

    std::optional<ContentId> session_current(std::string_view name) const {
        std::lock_guard lock(mu_);
        const auto it = sessions_.find(std::string(name));
        if (it == sessions_.end() || it->second.empty()) return std::nullopt;
        return it->second.back();
    }

    std::vector<ContentId> session_history(std::string_view name) const {
        std::lock_guard lock(mu_);
        const auto it = sessions_.find(std::string(name));
        return it == sessions_.end() ? std::vector<ContentId>{} : it->second;
    }

    SessionCandidates session_candidates(std::string_view name) {
        std::vector<ContentId> ids;
        SessionCandidates result;
        {
            std::lock_guard lock(mu_);
            const std::string key(name);
            const auto generation = alias_generations_.find(key);
            result.generation = generation == alias_generations_.end() ? 0 : generation->second;
            const auto history = sessions_.find(key);
            if (history != sessions_.end()) ids = history->second;
        }
        result.newest_to_oldest.reserve(ids.size());
        for (auto it = ids.rbegin(); it != ids.rend(); ++it) {
            SessionCandidateDescriptor descriptor{.id = *it};
            std::lock_guard lock(mu_);
            expire_l2(it->hex);
            auto found = catalog_.find(it->hex);
            if (found != catalog_.end() && found->second.durable && expired_l3(found->second)) {
                expire_l3(found);
                found = catalog_.find(it->hex);
            }
            if (found != catalog_.end()) {
                const Manifest& manifest = found->second;
                descriptor.image_format_version = manifest.image_format_version;
                descriptor.compatibility_key = manifest.compatibility_key;
                descriptor.image_bytes = manifest.image_bytes;
                descriptor.frontier_tokens = manifest.frontier_tokens;
                descriptor.boundary_tokens = manifest.boundary_tokens;
                descriptor.frontier_prefix_digest = manifest.frontier_prefix_digest;
                descriptor.boundary_prefix_digest = manifest.boundary_prefix_digest;
                if (l2_.contains(it->hex)) {
                    descriptor.source = CacheSource::L2;
                    descriptor.status = CacheLookupStatus::Hit;
                } else if (l3_available(it->hex, manifest)) {
                    descriptor.source = CacheSource::L3;
                    descriptor.status = CacheLookupStatus::Hit;
                } else {
                    descriptor.source = manifest.durable ? CacheSource::L3 : CacheSource::None;
                    descriptor.status = CacheLookupStatus::UnavailableOrCorrupt;
                }
            }
            result.newest_to_oldest.push_back(std::move(descriptor));
        }
        return result;
    }

    CacheLookupResult resolve_candidate(const SessionCandidateDescriptor& candidate) {
        {
            std::lock_guard lock(mu_);
            const auto found = catalog_.find(candidate.id.hex);
            if (found == catalog_.end() || !metadata_matches(found->second, candidate)) {
                return {.source = candidate.source,
                        .status = CacheLookupStatus::UnavailableOrCorrupt};
            }
        }
        return lookup_shared(candidate.id);
    }

    SessionPublishResult publish_immutable_alias(std::string_view name, const ContentId& id) {
        const auto reject = [&](SessionPublishOutcome outcome) {
            return SessionPublishResult{
                .id = id, .stored = false, .alias_advanced = false, .outcome = outcome};
        };
        if (!name.starts_with(kStableAliasPrefix) || !id.valid()) {
            return reject(SessionPublishOutcome::RejectedTooLarge);
        }
        std::lock_guard lock(mu_);
        expire_l2(id.hex);
        const auto entry = catalog_.find(id.hex);
        if (entry == catalog_.end() ||
            (entry->second.durable &&
             (expired_l3(entry->second) || pending_l3_drops_.contains(id.hex)) &&
             !l2_.contains(id.hex)) ||
            (!entry->second.durable && !l2_.contains(id.hex))) {
            return reject(SessionPublishOutcome::EvictedOnAdmission);
        }
        const std::string key(name);
        const auto existing = sessions_.find(key);
        if (existing != sessions_.end()) {
            const bool same = existing->second.size() == 1 && existing->second.front() == id;
            return SessionPublishResult{.id             = id,
                                        .stored         = true,
                                        .alias_advanced = same,
                                        .outcome = same ? SessionPublishOutcome::Advanced
                                                        : SessionPublishOutcome::AliasAlreadyOwned};
        }
        sessions_.emplace(key, std::vector<ContentId>{id});
        ++alias_generations_[key];
        if (entry->second.durable) persist_session(key, sessions_.at(key));
        return SessionPublishResult{.id             = id,
                                    .stored         = true,
                                    .alias_advanced = true,
                                    .outcome        = SessionPublishOutcome::Advanced};
    }

    bool promote(const ContentId& id) {
        return promote_image(id, {});
    }

    bool promote_image(const ContentId& id,
                       std::shared_ptr<const ContinuationImage> supplied_image) {
        if (!id.valid()) return false;
        Manifest snapshot;
        std::shared_ptr<const ContinuationImage> image;
        std::size_t reservation = 0;
        {
            std::lock_guard lock(mu_);
            auto found = catalog_.find(id.hex);
            if (found == catalog_.end()) return false;
            expire_l2(id.hex);
            if (found->second.durable && !expired_l3(found->second) &&
                !pending_l3_drops_.contains(id.hex)) return true;
            if (found->second.durable && expired_l3(found->second)) {
                expire_l3(found);
                found = catalog_.find(id.hex);
                if (found == catalog_.end() || pending_l3_drops_.contains(id.hex)) return false;
            }
            if (config_.l2_byte_budget != 0 &&
                found->second.image_bytes > config_.l2_byte_budget) return false;
            if (!promotions_.insert(id.hex).second) return false;
            const auto hot = l2_.find(id.hex);
            if ((hot == l2_.end() && !supplied_image) || !l3_enabled()) {
                promotions_.erase(id.hex);
                return false;
            }
            snapshot = found->second;
            image = hot == l2_.end() ? std::move(supplied_image) : hot->second.image;
            ++pins_[id.hex];
        }

        Bytes payload;
        try {
            auto serialized = serialize_with_regions(*image);
            payload          = std::move(serialized.payload);
            if (payload.size() != snapshot.image_bytes)
                throw std::runtime_error("continuation image changed");
            snapshot.chunks.clear();
            std::vector<std::size_t> boundaries = std::move(serialized.region_starts);
            boundaries.push_back(payload.size());
            std::size_t at = 0;
            for (const std::size_t boundary : boundaries) {
                while (at < boundary) {
                    const auto size = std::min(config_.chunk_bytes, boundary - at);
                    snapshot.chunks.push_back({sha256(payload.data() + at, size), size});
                    at += size;
                }
            }
            snapshot.expires_ms = expiration_ms(
                config_.now(), std::chrono::seconds(snapshot.l3_idle_ttl_seconds));
            snapshot.manifest_bytes = encode_manifest(snapshot).size();
        } catch (...) {
            std::lock_guard lock(mu_);
            promotions_.erase(id.hex);
            unpin_locked(id.hex);
            evict_l2();
            return false;
        }

        {
            std::lock_guard lock(mu_);
            retry_cleanup_locked();
            auto found = catalog_.find(id.hex);
            if (found == catalog_.end() || found->second.durable) {
                promotions_.erase(id.hex);
                unpin_locked(id.hex);
                evict_l2();
                return found != catalog_.end() && found->second.durable;
            }
            reservation = incremental_l3_bytes(snapshot);
            if (reservation > config_.l3_byte_budget || !make_l3_room(reservation)) {
                promotions_.erase(id.hex);
                unpin_locked(id.hex);
                evict_l2();
                return false;
            }
            l3_reserved_bytes_ += reservation;
            for (const auto& chunk : snapshot.chunks) ++promoting_chunks_[chunk.id];
        }

        bool written = false;
        try {
            if (config_.before_persistence_io) config_.before_persistence_io();
            if (!filesystem_has_room(reservation)) throw std::runtime_error("cache disk is full");
            // Chunks stop at segment boundaries, so their sizes are not a fixed stride and the
            // offset has to follow the list itself.
            std::size_t at = 0;
            for (const auto& descriptor : snapshot.chunks) {
                const auto size = static_cast<std::size_t>(descriptor.size);
                if (size == 0 || size > payload.size() - at) {
                    throw std::runtime_error("chunk layout does not cover the image");
                }
                // A chunk file is named by the hash of its contents, so one that is already
                // present is already this chunk. Writing it again would produce an identical file
                // and then lose the race to link() with EEXIST, which is what a continuation turn
                // did to every byte of its unchanged prefix: a full image write and fsync per
                // turn. Restores verify each chunk's hash, so skipping is not a weaker check.
                std::error_code present_ec;
                const auto path     = chunks_ / descriptor.id;
                const auto on_disk  = std::filesystem::file_size(path, present_ec);
                if (present_ec || on_disk != size) {
                    Bytes chunk(payload.begin() + static_cast<std::ptrdiff_t>(at),
                                payload.begin() + static_cast<std::ptrdiff_t>(at + size));
                    write_atomic_create_if_absent(path, chunk, temp_);
                }
                at += size;
            }
            if (at != payload.size()) {
                throw std::runtime_error("chunk layout does not cover the image");
            }
            auto encoded = encode_manifest(snapshot);
            write_atomic_create_if_absent(manifests_ / (id.hex + ".manifest"), encoded, temp_);
            written = true;
        } catch (...) {}

        std::lock_guard lock(mu_);
        l3_reserved_bytes_ -= reservation;
        promotions_.erase(id.hex);
        for (const auto& chunk : snapshot.chunks) {
            auto writer = promoting_chunks_.find(chunk.id);
            if (writer != promoting_chunks_.end() && --writer->second == 0) {
                promoting_chunks_.erase(writer);
            }
        }
        unpin_locked(id.hex);
        auto found = catalog_.find(id.hex);
        if (!written || found == catalog_.end() || found->second.durable) {
            if (!written) cleanup_failed_promotion_locked(id.hex, snapshot);
            evict_l2();
            return found != catalog_.end() && found->second.durable;
        }
        found->second.expires_ms = snapshot.expires_ms;
        found->second.chunks = std::move(snapshot.chunks);
        found->second.manifest_bytes = snapshot.manifest_bytes;
        found->second.durable = true;
        l3_bytes_ += found->second.manifest_bytes;
        for (const auto& chunk : found->second.chunks) {
            auto& ref = chunk_refs_[chunk.id];
            if (ref.refs++ == 0 && ref.bytes == 0) {
                ref.bytes = static_cast<std::size_t>(chunk.size);
                l3_bytes_ += ref.bytes;
            }
        }
        touch(found->second);
        evict_l2();
        evict_l3();
        return found->second.durable;
    }

    void persist_related_aliases(const std::string& id) {
        std::vector<AliasRecord> records;
        {
            std::lock_guard lock(mu_);
            for (const auto& [name, history] : sessions_) {
                if (std::find(history.begin(), history.end(), ContentId{id}) != history.end()) {
                    try {
                        records.push_back(make_alias_record(name, history));
                    } catch (...) {}
                }
            }
        }
        for (const auto& record : records) (void)write_alias_record_outside_lock(record);
    }

    bool discard_unstored(const std::string& id) {
        std::lock_guard lock(mu_);
        const auto found = catalog_.find(id);
        if (found == catalog_.end() || found->second.durable || l2_.contains(id)) return false;
        remove_from_sessions(id);
        catalog_.erase(found);
        return true;
    }

    bool queue_persistence(std::string_view name, const ContentId& id, std::uint64_t tokens,
                           bool immutable) {
        if (!l3_enabled() || name.empty() || !id.valid() ||
            (immutable != name.starts_with(kStableAliasPrefix))) {
            return false;
        }
        std::lock_guard lock(mu_);
        expire_l2(id.hex);
        const auto entry = catalog_.find(id.hex);
        const auto alias = sessions_.find(std::string(name));
        if (entry == catalog_.end() || !l2_.contains(id.hex) || alias == sessions_.end() ||
            alias->second.empty() || alias->second.back() != id) {
            return false;
        }
        ++persistence_queued_;
        const std::string key(name);
        if (!persisted_.contains(key)) {
            for (auto it = alias->second.rbegin(); it != alias->second.rend(); ++it) {
                if (*it == id) continue;
                const auto prior = catalog_.find(it->hex);
                if (prior != catalog_.end() && prior->second.durable &&
                    l3_available(it->hex, prior->second)) {
                    persisted_.emplace(
                        key, PersistedState{prior->second.frontier_tokens, config_.now()});
                    break;
                }
            }
        }
        auto pending = pending_persistence_.find(key);
        if (pending != pending_persistence_.end()) {
            ++persistence_coalesced_;
            if (pending->second.id != id) {
                ++pins_[id.hex];
                unpin_locked(pending->second.id.hex);
            }
            pending->second.id        = id;
            pending->second.tokens    = tokens;
            pending->second.retry_at  = {};
            pending->second.immutable = immutable;
        } else {
            ++pins_[id.hex];
            pending_persistence_.emplace(
                key, PendingPersistence{.id = id,
                                        .tokens = tokens,
                                        .queued_at = config_.now(),
                                        .retry_at = {},
                                        .immutable = immutable});
        }
        persistence_cv_.notify_one();
        return true;
    }

    bool rollback(std::string_view name, std::size_t depth) {
        if (name.starts_with(kStableAliasPrefix)) return false;
        std::lock_guard lock(mu_);
        auto it = sessions_.find(std::string(name));
        if (it == sessions_.end() || depth >= it->second.size())
            return depth == 0 && it != sessions_.end();
        if (depth == 0) return true;
        it->second.erase(it->second.end() - static_cast<std::ptrdiff_t>(depth), it->second.end());
        ++alias_generations_[it->first];
        persist_session(it->first, it->second);
        return true;
    }

    SessionRollbackResult rollback_to(std::string_view name, const ContentId& id,
                                      std::uint64_t expected_generation) {
        if (name.starts_with(kStableAliasPrefix) || !id.valid()) return {};
        SessionRollbackResult result;
        {
            std::lock_guard lock(mu_);
            const std::string key(name);
            const auto generation = alias_generations_.find(key);
            const std::uint64_t current_generation =
                generation == alias_generations_.end() ? 0 : generation->second;
            result.generation = current_generation;
            if (current_generation != expected_generation) return result;
            auto session = sessions_.find(key);
            if (session == sessions_.end()) return result;
            const auto selected = std::find(session->second.begin(), session->second.end(), id);
            if (selected == session->second.end()) return result;
            if (l3_enabled()) {
                const std::vector<ContentId> rolled_back(session->second.begin(), selected + 1);
                AliasRecord record = make_alias_record(key, rolled_back);
                record.generation  = current_generation + 1;
                pending_alias_writes_.insert_or_assign(key, std::move(record));
            }
            session->second.erase(selected + 1, session->second.end());
            result.generation = ++alias_generations_[key];
            result.rolled_back = true;
            if (l3_enabled()) persistence_cv_.notify_one();
        }
        return result;
    }

    bool pin(const ContentId& id) {
        if (!id.valid()) return false;
        std::lock_guard lock(mu_);
        expire_l2(id.hex);
        auto found = catalog_.find(id.hex);
        if (found == catalog_.end()) return false;
        if (found->second.durable && expired_l3(found->second) && !l2_.contains(id.hex)) {
            expire_l3(found);
            return false;
        }
        if (pending_l3_drops_.contains(id.hex) && !l2_.contains(id.hex)) return false;
        ++pins_[id.hex];
        return true;
    }

    void unpin(const ContentId& id) {
        std::lock_guard lock(mu_);
        unpin_locked(id.hex);
        expire_l2(id.hex);
        retry_cleanup_locked();
        evict_l2();
        evict_l3();
    }

    CacheStats stats() const {
        std::lock_guard lock(mu_);
        std::size_t durable = 0;
        for (const auto& [id, manifest] : catalog_) {
            (void)id;
            durable += manifest.durable;
        }
        return {.entries = catalog_.size(),
                .l2_entries = l2_.size(),
                .l2_bytes = l2_bytes_,
                .l3_entries = durable,
                .l3_bytes = l3_bytes_,
                .persistence_queued = persistence_queued_,
                .persistence_coalesced = persistence_coalesced_,
                 .persistence_successes = persistence_successes_,
                 .persistence_failures = persistence_failures_,
                 .l2_admission_microseconds = l2_admission_microseconds_,
                 .l2_admission_operations = l2_admission_operations_,
                 .l3_persistence_microseconds = l3_persistence_microseconds_,
                 .l3_persistence_operations = l3_persistence_operations_};
    }

private:
    struct Hot {
        std::shared_ptr<const ContinuationImage> image;
        std::size_t bytes;
        std::uint64_t recency;
        double score;
        std::chrono::seconds idle_ttl;
        std::int64_t expires_ms;
    };

    struct PendingPersistence {
        ContentId id;
        std::uint64_t tokens;
        Clock::time_point queued_at;
        Clock::time_point retry_at{};
        bool immutable;
    };

    struct ChunkRef {
        std::size_t refs  = 0;
        std::size_t bytes = 0;
    };

    void discover() {
        const std::size_t image_limit = configured_image_limit();
        const std::size_t chunk_count = max_chunk_count(image_limit, config_.chunk_bytes);
        constexpr std::size_t manifest_header_bytes =
            8 + sizeof(std::uint32_t) + 13 * sizeof(std::uint64_t) +
            kMaxSelectionMetadataBytes + 2 * kPrefixDigestBytes;
        constexpr std::size_t manifest_chunk_bytes =
            sizeof(std::uint64_t) + 64 + sizeof(std::uint64_t);
        const std::size_t manifest_limit =
            chunk_count > (std::numeric_limits<std::size_t>::max() - manifest_header_bytes) /
                              manifest_chunk_bytes
                ? std::numeric_limits<std::size_t>::max()
                : manifest_header_bytes + chunk_count * manifest_chunk_bytes;
        std::error_code ec;
        std::filesystem::directory_iterator it(manifests_, ec), end;
        while (!ec && it != end) {
            try {
                const auto path = it->path();
                if (!it->is_regular_file() || path.extension() != ".manifest") {
                    it.increment(ec);
                    continue;
                }
                const auto id = path.stem().string();
                if (!ContentId{id}.valid()) {
                    it.increment(ec);
                    continue;
                }
                auto manifest = decode_manifest(read_file(path, 0, manifest_limit), image_limit,
                                                config_.chunk_bytes);
                bool chunks_exist = true;
                for (const auto& c : manifest.chunks) {
                    std::error_code size_ec;
                    chunks_exist &=
                        std::filesystem::file_size(chunks_ / c.id, size_ec) == c.size && !size_ec;
                }
                if (chunks_exist) {
                    manifest.durable = true;
                    add_manifest(id, std::move(manifest));
                }
            } catch (
                ...) { /* Invalid manifests and atomic-write remnants are intentionally ignored. */
            }
            it.increment(ec);
        }
    }

    std::size_t configured_image_limit() const {
        std::size_t limit = config_.l3_byte_budget;
        if (config_.l2_byte_budget != 0) limit = std::min(limit, config_.l2_byte_budget);
        return limit;
    }

    void cleanup_orphans() {
        const auto clean_directory = [&](const std::filesystem::path& directory,
                                         const auto& keep) {
            std::error_code ec;
            std::filesystem::directory_iterator it(directory, ec), end;
            while (!ec && it != end) {
                const auto path = it->path();
                if (it->is_regular_file(ec) && !keep(path)) {
                    remove_or_account_orphan(path);
                }
                it.increment(ec);
            }
        };
        clean_directory(temp_, [](const std::filesystem::path&) { return false; });
        clean_directory(chunks_, [&](const std::filesystem::path& path) {
            return chunk_refs_.contains(path.filename().string());
        });
        clean_directory(manifests_, [&](const std::filesystem::path& path) {
            return path.extension() == ".manifest" &&
                   catalog_.contains(path.stem().string());
        });
    }

    void remove_or_account_orphan(const std::filesystem::path& path) {
        std::error_code ec;
        if (std::filesystem::remove(path, ec) || (!ec && !std::filesystem::exists(path))) {
            const auto orphan = orphan_files_.find(path.string());
            if (orphan != orphan_files_.end()) {
                l3_bytes_ -= orphan->second;
                orphan_files_.erase(orphan);
            }
            return;
        }
        const auto bytes = std::filesystem::file_size(path, ec);
        if (!ec && bytes <= std::numeric_limits<std::size_t>::max() &&
            !orphan_files_.contains(path.string())) {
            orphan_files_.emplace(path.string(), static_cast<std::size_t>(bytes));
            l3_bytes_ += static_cast<std::size_t>(bytes);
        }
    }

    void cleanup_failed_promotion_locked(const std::string& id, const Manifest& manifest) {
        remove_or_account_orphan(manifests_ / (id + ".manifest"));
        for (const auto& chunk : manifest.chunks) {
            if (!chunk_refs_.contains(chunk.id) && !promoting_chunks_.contains(chunk.id)) {
                remove_or_account_orphan(chunks_ / chunk.id);
            }
        }
    }

    void retry_cleanup_locked() {
        std::vector<std::string> orphan_paths;
        orphan_paths.reserve(orphan_files_.size());
        for (const auto& [path, bytes] : orphan_files_) {
            (void)bytes;
            orphan_paths.push_back(path);
        }
        for (const auto& path : orphan_paths) remove_or_account_orphan(path);

        for (auto it = chunk_refs_.begin(); it != chunk_refs_.end();) {
            if (it->second.refs != 0) {
                ++it;
                continue;
            }
            std::error_code ec;
            const auto path = chunks_ / it->first;
            if (std::filesystem::remove(path, ec) || (!ec && !std::filesystem::exists(path))) {
                l3_bytes_ -= it->second.bytes;
                it = chunk_refs_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void discover_aliases() {
        std::error_code ec;
        std::filesystem::directory_iterator it(aliases_, ec), end;
        while (!ec && it != end) {
            try {
                const auto path = it->path();
                if (!it->is_regular_file() || path.extension() != ".alias") {
                    it.increment(ec);
                    continue;
                }
                auto alias = decode_alias(read_file(path), config_.session_history_depth);
                if (path.stem() !=
                    sha256(reinterpret_cast<const std::uint8_t*>(alias.session.data()),
                           alias.session.size()))
                    throw std::runtime_error("alias filename mismatch");
                const auto original_size = alias.history.size();
                std::erase_if(alias.history, [&](const ContentId& id) {
                    const auto found = catalog_.find(id.hex);
                    return found == catalog_.end() || !found->second.durable ||
                            expired_l3(found->second);
                });
                if (alias.history.empty()) {
                    std::error_code remove_ec;
                    if (std::filesystem::remove(path, remove_ec)) sync_directory(aliases_);
                } else {
                    sessions_.insert_or_assign(alias.session, alias.history);
                    ++alias_generations_[alias.session];
                    if (alias.history.size() != original_size)
                        persist_session(alias.session, alias.history);
                }
            } catch (...) {
                // Alias corruption is isolated control-data loss and must never prevent startup.
            }
            it.increment(ec);
        }
    }

    void add_manifest(const std::string& id, Manifest manifest) {
        if (catalog_.contains(id)) return;
        if (manifest.durable) {
            l3_bytes_ += manifest.manifest_bytes;
            for (const auto& c : manifest.chunks) {
                auto& ref = chunk_refs_[c.id];
                if (ref.refs++ == 0 && ref.bytes == 0) {
                    ref.bytes = c.size;
                    l3_bytes_ += c.size;
                }
            }
        }
        touch(manifest);
        catalog_.emplace(id, std::move(manifest));
    }

    bool l3_enabled() const { return config_.enable_l3 && config_.l3_byte_budget != 0; }

    std::size_t incremental_l3_bytes(const Manifest& manifest) const {
        std::size_t bytes = manifest.manifest_bytes;
        for (const auto& chunk : manifest.chunks) {
            if (!chunk_refs_.contains(chunk.id)) {
                if (chunk.size > std::numeric_limits<std::size_t>::max() - bytes)
                    return std::numeric_limits<std::size_t>::max();
                bytes += static_cast<std::size_t>(chunk.size);
            }
        }
        return bytes;
    }

    bool fits_l3(const Manifest& manifest) const {
        const auto bytes = incremental_l3_bytes(manifest);
        return bytes <= config_.l3_byte_budget;
    }

    bool filesystem_has_room(std::uintmax_t bytes) const {
        std::error_code ec;
        const auto info = std::filesystem::space(config_.root, ec);
        if (ec || info.available < config_.filesystem_reserve_bytes) return false;
        return bytes <= info.available - config_.filesystem_reserve_bytes;
    }

    bool filesystem_has_room(const Manifest& manifest) const {
        return filesystem_has_room(incremental_l3_bytes(manifest));
    }

    void persistence_loop() noexcept {
        std::unique_lock lock(mu_);
        for (;;) {
            if (pending_alias_writes_.empty() && pending_persistence_.empty() &&
                !persistence_stopping_) {
                persistence_cv_.wait(lock);
                continue;
            }
            if (!pending_alias_writes_.empty()) {
                auto selected = pending_alias_writes_.begin();
                AliasRecord record = std::move(selected->second);
                pending_alias_writes_.erase(selected);
                lock.unlock();
                (void)write_alias_record_outside_lock(std::move(record));
                lock.lock();
                continue;
            }
            if (pending_persistence_.empty()) return;

            const auto now = config_.now();
            auto selected  = pending_persistence_.end();
            Clock::time_point next_due = Clock::time_point::max();
            for (auto it = pending_persistence_.begin(); it != pending_persistence_.end(); ++it) {
                if (!persistence_stopping_ && it->second.retry_at > now) {
                    next_due = std::min(next_due, it->second.retry_at);
                    continue;
                }
                const auto previous = persisted_.find(it->first);
                const std::optional<PersistedState> state =
                    previous == persisted_.end() ? std::nullopt
                                                 : std::optional<PersistedState>(previous->second);
                if (persistence_stopping_ ||
                    persistence_due({config_.persist_interval, config_.persist_min_tokens}, state,
                                    it->second.tokens, it->second.queued_at, now)) {
                    selected = it;
                    break;
                }
                if (config_.persist_interval > Clock::duration::zero()) {
                    const auto base = state ? state->at : it->second.queued_at;
                    next_due        = std::min(next_due, base + config_.persist_interval);
                }
            }
            if (selected == pending_persistence_.end()) {
                if (next_due == Clock::time_point::max()) {
                    persistence_cv_.wait(lock);
                    continue;
                }
                const auto delay = next_due > now ? next_due - now : Clock::duration::zero();
                persistence_cv_.wait_for(lock, delay);
                continue;
            }

            const std::string alias = selected->first;
            const PendingPersistence item = selected->second;
            pending_persistence_.erase(selected);
            const auto persistence_started = std::chrono::steady_clock::now();
            lock.unlock();
            const bool promoted = promote(item.id);
            lock.lock();
            bool alias_current  = false;
            const auto current  = sessions_.find(alias);
            if (current != sessions_.end() && !current->second.empty()) {
                alias_current = item.immutable ? current->second.front() == item.id
                                               : current->second.back() == item.id;
            }
            if (!alias_current) {
                unpin_locked(item.id.hex);
                evict_l2();
            } else if (promoted) {
                std::optional<AliasRecord> record;
                try {
                    record = make_alias_record(alias, current->second);
                } catch (...) {}
                lock.unlock();
                const bool persisted = record && write_alias_record_outside_lock(*record);
                lock.lock();
                const auto latest = sessions_.find(alias);
                const bool still_current = latest != sessions_.end() && !latest->second.empty() &&
                                           (item.immutable ? latest->second.front() == item.id
                                                           : latest->second.back() == item.id);
                if (!still_current) {
                    unpin_locked(item.id.hex);
                    evict_l2();
                } else if (persisted) {
                    persisted_.insert_or_assign(alias,
                                                PersistedState{item.tokens, config_.now()});
                    ++persistence_successes_;
                    unpin_locked(item.id.hex);
                    evict_l2();
                } else {
                    ++persistence_failures_;
                    retry_persistence(alias, item);
                }
            } else {
                ++persistence_failures_;
                retry_persistence(alias, item);
            }
            ++l3_persistence_operations_;
            l3_persistence_microseconds_ += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - persistence_started)
                    .count());
        }
    }

    void retry_persistence(const std::string& alias, PendingPersistence item) {
        if (persistence_stopping_) {
            unpin_locked(item.id.hex);
            evict_l2();
            return;
        }
        const auto backoff = config_.persist_interval.count() == 0
                                 ? std::chrono::seconds(1)
                                 : config_.persist_interval;
        item.retry_at = config_.now() + backoff;
        pending_persistence_.emplace(alias, std::move(item));
    }

    bool expired_at(std::int64_t expires_ms) const {
        return expires_ms != 0 && expires_ms <= millis(config_.now());
    }

    bool expired_l3(const Manifest& m) const { return expired_at(m.expires_ms); }

    bool l3_available(const std::string& id, const Manifest& m) const {
        return m.durable && !expired_l3(m) && !pending_l3_drops_.contains(id);
    }

    void expire_l2(const std::string& id) {
        const auto hot = l2_.find(id);
        if (hot == l2_.end() || pins_.contains(id) || !expired_at(hot->second.expires_ms)) return;
        l2_bytes_ -= hot->second.bytes;
        l2_.erase(hot);
    }

    void touch(Manifest& m) {
        const double size = std::max<std::uint64_t>(1, m.image_bytes);
        m.score           = std::max(m.score, inflation_) + m.cost / size + (++tick_ * 1e-15);
    }

    void touch_l2(const std::string& id, std::shared_ptr<const ContinuationImage> image,
                  std::size_t bytes,
                  std::chrono::seconds idle_ttl) {
        if (config_.l2_byte_budget == 0 || bytes > config_.l2_byte_budget) return;
        const auto manifest = catalog_.find(id);
        const double cost = manifest == catalog_.end() ? 1.0 : manifest->second.cost;
        const double score = l2_inflation_ + cost / std::max<std::size_t>(1, bytes) +
                             (++tick_ * 1e-15);
        auto existing = l2_.find(id);
        if (existing != l2_.end()) {
            existing->second.recency = tick_;
            existing->second.score = std::max(existing->second.score, score);
            existing->second.idle_ttl = idle_ttl;
            existing->second.expires_ms = expiration_ms(config_.now(), idle_ttl);
            return;
        }
        l2_.emplace(id, Hot{std::move(image), bytes, tick_, score, idle_ttl,
                            expiration_ms(config_.now(), idle_ttl)});
        l2_bytes_ += bytes;
    }

    std::shared_ptr<const ContinuationImage>
    lease_locked(const std::string& id, std::shared_ptr<const ContinuationImage> image,
                 bool already_pinned = false) {
        struct Lease {
            std::weak_ptr<Impl> cache;
            std::string id;
            std::shared_ptr<const ContinuationImage> image;
            ~Lease() {
                if (auto owner = cache.lock()) owner->unpin(ContentId{id});
            }
        };
        try {
            auto lease   = std::make_shared<Lease>();
            lease->cache = shared_from_this();
            lease->id    = id;
            lease->image = std::move(image);
            if (!already_pinned) ++pins_[id];
            const auto* pointer = lease->image.get();
            return {std::move(lease), pointer};
        } catch (...) {
            if (already_pinned) unpin_locked(id);
            return {};
        }
    }

    // `protected_id`, when set, is the entry this eviction pass must not choose. It is the image a
    // session publication has just admitted: evicting it would undo the publication that triggered
    // the pass, so the caller would be told the state was not stored while older states survive.
    void evict_l2(std::string_view protected_id = {}) {
        while (l2_bytes_ > config_.l2_byte_budget) {
            auto victim = l2_.end();
            for (auto it = l2_.begin(); it != l2_.end(); ++it)
                if (!pins_.contains(it->first) && it->first != protected_id &&
                    (victim == l2_.end() || it->second.score < victim->second.score ||
                     (it->second.score == victim->second.score &&
                      it->second.recency < victim->second.recency)))
                    victim = it;
            if (victim == l2_.end()) break;
            l2_inflation_ = std::max(l2_inflation_, victim->second.score);
            const auto id = victim->first;
            l2_bytes_ -= victim->second.bytes;
            l2_.erase(victim);
            if (auto entry = catalog_.find(id); entry != catalog_.end() && !entry->second.durable) {
                remove_from_sessions(id);
                catalog_.erase(entry);
            }
        }
    }

    void evict_l3() {
        while (l3_bytes_ > config_.l3_byte_budget) {
            auto victim = catalog_.end();
            double best = std::numeric_limits<double>::infinity();
            for (auto it = catalog_.begin(); it != catalog_.end(); ++it) {
                if (!it->second.durable || pins_.contains(it->first)) continue;
                const double score = expired_l3(it->second) ? -1 : it->second.score;
                if (score < best) {
                    best   = score;
                    victim = it;
                }
            }
            if (victim == catalog_.end()) break;
            inflation_ = std::max(inflation_, best);
            drop_l3(victim);
        }
    }

    bool make_l3_room(std::size_t bytes) {
        const std::size_t after_reservation = config_.l3_byte_budget - bytes;
        if (l3_reserved_bytes_ > after_reservation) return false;
        const std::size_t available = after_reservation - l3_reserved_bytes_;
        while (l3_bytes_ > available) {
            auto victim = catalog_.end();
            double best = std::numeric_limits<double>::infinity();
            for (auto it = catalog_.begin(); it != catalog_.end(); ++it) {
                if (!it->second.durable || pins_.contains(it->first)) continue;
                const double score = expired_l3(it->second) ? -1 : it->second.score;
                if (score < best) {
                    best = score;
                    victim = it;
                }
            }
            if (victim == catalog_.end()) return false;
            inflation_ = std::max(inflation_, best);
            drop_l3(victim);
        }
        return true;
    }

    void unpin_locked(const std::string& id) {
        auto it = pins_.find(id);
        if (it == pins_.end() || --it->second != 0) return;
        pins_.erase(it);
        if (pending_l3_drops_.erase(id) != 0) {
            const auto found = catalog_.find(id);
            if (found != catalog_.end()) drop_l3(found);
        }
    }

    void expire_l3(std::unordered_map<std::string, Manifest>::iterator it, bool force = false) {
        if (!it->second.durable || (!force && !expired_l3(it->second))) return;
        if (pins_.contains(it->first)) {
            pending_l3_drops_.insert(it->first);
            remove_from_sessions(it->first);
            return;
        }
        drop_l3(it);
    }

    void drop_l3(std::unordered_map<std::string, Manifest>::iterator it) {
        if (!it->second.durable) return;
        if (pins_.contains(it->first)) {
            pending_l3_drops_.insert(it->first);
            return;
        }
        const auto id = it->first;
        remove_l3_files(it->first, it->second);
        it->second.durable = false;
        if (!l2_.contains(id)) {
            remove_from_sessions(id, false);
            catalog_.erase(it);
            return;
        }
    }

    void remove_l3_files(const std::string& id, const Manifest& manifest) {
        std::error_code ec;
        const auto manifest_path = manifests_ / (id + ".manifest");
        const bool manifest_removed = std::filesystem::remove(manifest_path, ec);
        if (manifest_removed || (!ec && !std::filesystem::exists(manifest_path))) {
            l3_bytes_ -= manifest.manifest_bytes;
        } else {
            orphan_files_.try_emplace(manifest_path.string(), manifest.manifest_bytes);
        }
        for (const auto& c : manifest.chunks) {
            auto ref = chunk_refs_.find(c.id);
            if (ref != chunk_refs_.end() && --ref->second.refs == 0) {
                ec.clear();
                const auto path = chunks_ / c.id;
                const bool removed = std::filesystem::remove(path, ec);
                if (removed || (!ec && !std::filesystem::exists(path))) {
                    l3_bytes_ -= ref->second.bytes;
                    chunk_refs_.erase(ref);
                }
            }
        }
    }

    void update_session(const std::string& name, const ContentId& id, bool persist) {
        if (name.starts_with(kStableAliasPrefix)) return;
        auto& history = sessions_[name];
        if (!history.empty() && history.back() == id) return;
        history.push_back(id);
        if (history.size() > config_.session_history_depth)
            history.erase(history.begin(),
                          history.begin() + static_cast<std::ptrdiff_t>(
                                                history.size() - config_.session_history_depth));
        ++alias_generations_[name];
        if (persist) persist_session(name, history);
    }

    struct AliasRecord {
        std::string name;
        std::uint64_t generation;
        std::filesystem::path path;
        std::optional<Bytes> bytes;
    };

    AliasRecord make_alias_record(const std::string& name,
                                  const std::vector<ContentId>& history) const {
        std::vector<ContentId> durable;
        durable.reserve(history.size());
        for (const auto& id : history) {
            const auto found = catalog_.find(id.hex);
            if (found != catalog_.end() && l3_available(id.hex, found->second)) {
                durable.push_back(id);
            }
        }
        const auto generation = alias_generations_.find(name);
        AliasRecord record{.name = name,
                           .generation = generation == alias_generations_.end() ? 0
                                                                                : generation->second,
                           .path = aliases_ /
                                    (sha256(reinterpret_cast<const std::uint8_t*>(name.data()),
                                            name.size()) +
                                     ".alias")};
        if (!durable.empty()) record.bytes = encode_alias(name, durable);
        return record;
    }

    bool write_alias_record(const AliasRecord& record) const {
        try {
            if (!record.bytes) {
                std::error_code ec;
                if (std::filesystem::remove(record.path, ec)) sync_directory(aliases_);
                return !ec;
            }
            write_atomic_replace(record.path, *record.bytes, temp_,
                                 config_.before_mutable_replace);
            return true;
        } catch (...) { return false; }
    }

    bool write_alias_record_outside_lock(AliasRecord record) {
        if (config_.before_alias_persistence_io) {
            config_.before_alias_persistence_io(record.name);
        }
        for (;;) {
            const bool written = write_alias_record(record);
            std::lock_guard lock(mu_);
            const auto generation = alias_generations_.find(record.name);
            const std::uint64_t current_generation =
                generation == alias_generations_.end() ? 0 : generation->second;
            if (current_generation == record.generation) return written;
            const auto current = sessions_.find(record.name);
            try {
                record = make_alias_record(
                    record.name,
                    current == sessions_.end() ? std::vector<ContentId>{} : current->second);
            } catch (...) {
                return false;
            }
        }
    }

    bool persist_session(const std::string& name, const std::vector<ContentId>& history) const {
        if (!l3_enabled()) return false;
        try {
            return write_alias_record(make_alias_record(name, history));
        } catch (...) {
            // Session persistence is opportunistic; inference and the in-memory alias remain valid.
            return false;
        }
    }

    void remove_from_sessions(const std::string& id, bool persist = true) {
        for (auto it = sessions_.begin(); it != sessions_.end();) {
            const auto old_size = it->second.size();
            std::erase_if(it->second, [&](const ContentId& value) { return value.hex == id; });
            if (it->second.size() == old_size) {
                ++it;
                continue;
            }
            ++alias_generations_[it->first];
            if (persist) persist_session(it->first, it->second);
            if (it->second.empty())
                it = sessions_.erase(it);
            else
                ++it;
        }
    }

    CacheConfig config_;
    std::filesystem::path chunks_, manifests_, aliases_, temp_;
    mutable std::mutex mu_;
    std::unordered_map<std::string, Manifest> catalog_;
    std::unordered_map<std::string, ChunkRef> chunk_refs_;
    std::unordered_map<std::string, std::size_t> orphan_files_;
    std::unordered_map<std::string, Hot> l2_;
    std::unordered_map<std::string, std::vector<ContentId>> sessions_;
    std::unordered_map<std::string, std::uint64_t> alias_generations_;
    std::unordered_map<std::string, std::size_t> pins_;
    std::unordered_set<std::string> pending_l3_drops_;
    std::unordered_set<std::string> promotions_;
    std::unordered_map<std::string, std::size_t> promoting_chunks_;
    std::unordered_map<std::string, PendingPersistence> pending_persistence_;
    std::unordered_map<std::string, AliasRecord> pending_alias_writes_;
    std::unordered_map<std::string, PersistedState> persisted_;
    std::size_t l2_bytes_ = 0, l3_bytes_ = 0, l3_reserved_bytes_ = 0;
    std::uint64_t tick_ = 0;
    double inflation_ = 0, l2_inflation_ = 0;
    std::uint64_t persistence_queued_ = 0, persistence_coalesced_ = 0;
    std::uint64_t persistence_successes_ = 0, persistence_failures_ = 0;
    std::uint64_t l2_admission_microseconds_ = 0, l2_admission_operations_ = 0;
    std::uint64_t l3_persistence_microseconds_ = 0, l3_persistence_operations_ = 0;
    std::condition_variable persistence_cv_;
    bool persistence_stopping_ = false;
    std::thread persistence_worker_;
};

ContinuationCache::ContinuationCache(CacheConfig config)
    : impl_(std::make_shared<Impl>(std::move(config))) {}

ContinuationCache::~ContinuationCache()                                       = default;
ContinuationCache::ContinuationCache(ContinuationCache&&) noexcept            = default;
ContinuationCache& ContinuationCache::operator=(ContinuationCache&&) noexcept = default;

ContentId ContinuationCache::store(const ContinuationImage& image, const StoreOptions& options) {
    return impl_->store(image, options);
}

ContentId ContinuationCache::admit(const ContinuationImage& image, const StoreOptions& options) {
    return impl_->admit(image, options);
}

ContentId ContinuationCache::admit(ContinuationImage&& image, const StoreOptions& options) {
    return impl_->admit(std::move(image), options);
}

bool ContinuationCache::promote(const ContentId& id) { return impl_->promote(id); }

SessionPublishResult
ContinuationCache::publish_session(const ContinuationImage& image, std::string_view session,
                                   const std::optional<ContentId>& expected_head,
                                   const StoreOptions& options,
                                   std::optional<std::uint64_t> expected_generation) {
    return impl_->publish_session(image, session, expected_head, options, expected_generation);
}

SessionPublishResult
ContinuationCache::publish_session_l2(const ContinuationImage& image, std::string_view session,
                                      const std::optional<ContentId>& expected_head,
                                      const StoreOptions& options,
                                      std::optional<std::uint64_t> expected_generation) {
    return impl_->publish_session_l2(image, session, expected_head, options,
                                     expected_generation);
}

SessionPublishResult
ContinuationCache::publish_session_l2(ContinuationImage&& image, std::string_view session,
                                       const std::optional<ContentId>& expected_head,
                                       const StoreOptions& options,
                                       std::optional<std::uint64_t> expected_generation) {
    return impl_->publish_session_l2(std::move(image), session, expected_head, options,
                                     expected_generation);
}

std::optional<ContinuationImage> ContinuationCache::get(const ContentId& id) {
    return impl_->get(id);
}

std::shared_ptr<const ContinuationImage> ContinuationCache::get_shared(const ContentId& id) {
    return impl_->get_shared(id);
}

CacheLookupResult ContinuationCache::lookup_shared(const ContentId& id) {
    return impl_->lookup_shared(id);
}

std::optional<ContentId> ContinuationCache::session_current(std::string_view s) const {
    return impl_->session_current(s);
}

std::vector<ContentId> ContinuationCache::session_history(std::string_view s) const {
    return impl_->session_history(s);
}

SessionCandidates ContinuationCache::session_candidates(std::string_view s) {
    return impl_->session_candidates(s);
}

CacheLookupResult
ContinuationCache::resolve_candidate(const SessionCandidateDescriptor& candidate) {
    return impl_->resolve_candidate(candidate);
}

bool ContinuationCache::rollback_session(std::string_view s, std::size_t d) {
    return impl_->rollback(s, d);
}

SessionRollbackResult ContinuationCache::rollback_session_to(
    std::string_view s, const ContentId& id, std::uint64_t expected_generation) {
    return impl_->rollback_to(s, id, expected_generation);
}

SessionPublishResult ContinuationCache::publish_immutable_alias(std::string_view s,
                                                                const ContentId& id) {
    return impl_->publish_immutable_alias(s, id);
}

bool ContinuationCache::queue_persistence(std::string_view alias, const ContentId& id,
                                          std::uint64_t tokens, bool immutable) {
    return impl_->queue_persistence(alias, id, tokens, immutable);
}

bool ContinuationCache::pin(const ContentId& id) { return impl_->pin(id); }

void ContinuationCache::unpin(const ContentId& id) { impl_->unpin(id); }

CacheStats ContinuationCache::stats() const { return impl_->stats(); }

} // namespace ninfer::cache
