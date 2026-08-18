#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kV1Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{1},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kPayloadAlignment = 4096;

#ifndef _WIN32
constexpr std::array<std::byte, 8> kFingerprintRecordMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'F'}, std::byte{'S'},
    std::byte{'H'}, std::byte{'A'}, std::byte{0},   std::byte{1},
};
constexpr std::uint32_t kFingerprintRecordVersion = 1;
constexpr std::size_t kMaxFingerprintRecordBytes  = 64U << 10;
constexpr std::size_t kMaxCanonicalPathBytes      = 32U << 10;

struct FingerprintFileIdentity {
    std::uint64_t size;
    std::uint64_t device;
    std::uint64_t inode;
    std::int64_t mtime_seconds;
    std::int64_t mtime_nanoseconds;
    std::int64_t ctime_seconds;
    std::int64_t ctime_nanoseconds;

    bool operator==(const FingerprintFileIdentity&) const = default;
};

FingerprintFileIdentity file_identity(const struct stat& status) {
    return {
        .size              = static_cast<std::uint64_t>(status.st_size),
        .device            = static_cast<std::uint64_t>(status.st_dev),
        .inode             = static_cast<std::uint64_t>(status.st_ino),
        .mtime_seconds     = static_cast<std::int64_t>(status.st_mtim.tv_sec),
        .mtime_nanoseconds = static_cast<std::int64_t>(status.st_mtim.tv_nsec),
        .ctime_seconds     = static_cast<std::int64_t>(status.st_ctim.tv_sec),
        .ctime_nanoseconds = static_cast<std::int64_t>(status.st_ctim.tv_nsec),
    };
}

struct FingerprintFileContext {
    std::string canonical_path;
    FingerprintFileIdentity identity;

    bool operator==(const FingerprintFileContext&) const = default;
};

bool safe_namespace(std::string_view value) {
    return !value.empty() && value != "." && value != ".." &&
           std::all_of(value.begin(), value.end(), [](unsigned char c) {
               return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                      (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
           });
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) {
        output.push_back(static_cast<std::byte>((value >> (8U * i)) & 0xffU));
    }
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        output.push_back(static_cast<std::byte>((value >> (8U * i)) & 0xffU));
    }
}

bool consume_u32(std::span<const std::byte>& input, std::uint32_t& value) {
    if (input.size() < 4) { return false; }
    value = 0;
    for (unsigned i = 0; i < 4; ++i) {
        value |= static_cast<std::uint32_t>(std::to_integer<unsigned char>(input[i])) << (8U * i);
    }
    input = input.subspan(4);
    return true;
}

bool consume_u64(std::span<const std::byte>& input, std::uint64_t& value) {
    if (input.size() < 8) { return false; }
    value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(input[i])) << (8U * i);
    }
    input = input.subspan(8);
    return true;
}

std::string digest_hex(const Sha256Digest& digest) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        result[2 * i]     = digits[digest[i] >> 4U];
        result[2 * i + 1] = digits[digest[i] & 0x0fU];
    }
    return result;
}

std::string path_cache_key(std::string_view canonical_path) {
    Sha256 sha;
    sha.update(std::as_bytes(std::span(canonical_path.data(), canonical_path.size())));
    return digest_hex(sha.finish());
}

std::filesystem::path fingerprint_record_path(const FingerprintCacheOptions& options,
                                              std::string_view canonical_path) {
    return options.directory / options.cache_namespace / "artifacts" /
           (path_cache_key(canonical_path) + ".sha256");
}

bool private_owned_directory(const std::filesystem::path& path) {
    struct stat status {};
    return ::lstat(path.c_str(), &status) == 0 && S_ISDIR(status.st_mode) &&
           status.st_uid == ::geteuid() && (status.st_mode & 0777) == 0700;
}

std::optional<Sha256Digest> load_fingerprint_record(const std::filesystem::path& path,
                                                    const FingerprintFileContext& context) {
    if (!private_owned_directory(path.parent_path().parent_path()) ||
        !private_owned_directory(path.parent_path())) {
        return std::nullopt;
    }
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) { return std::nullopt; }

    struct stat status {};
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_uid != ::geteuid() ||
        (status.st_mode & 0777) != 0600 || status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) > kMaxFingerprintRecordBytes) {
        ::close(fd);
        return std::nullopt;
    }

    std::vector<std::byte> bytes(static_cast<std::size_t>(status.st_size));
    std::size_t completed = 0;
    while (completed < bytes.size()) {
        const ssize_t amount = ::read(fd, bytes.data() + completed, bytes.size() - completed);
        if (amount < 0 && errno == EINTR) { continue; }
        if (amount <= 0) {
            ::close(fd);
            return std::nullopt;
        }
        completed += static_cast<std::size_t>(amount);
    }
    if (::close(fd) != 0) { return std::nullopt; }

    std::span<const std::byte> input(bytes);
    if (input.size() < kFingerprintRecordMagic.size() ||
        !std::equal(kFingerprintRecordMagic.begin(), kFingerprintRecordMagic.end(), input.begin())) {
        return std::nullopt;
    }
    input = input.subspan(kFingerprintRecordMagic.size());
    std::uint32_t version = 0;
    std::uint32_t path_bytes = 0;
    std::array<std::uint64_t, 7> fields{};
    if (!consume_u32(input, version) || version != kFingerprintRecordVersion ||
        !consume_u32(input, path_bytes) || path_bytes == 0 ||
        path_bytes > kMaxCanonicalPathBytes) {
        return std::nullopt;
    }
    for (auto& field : fields) {
        if (!consume_u64(input, field)) { return std::nullopt; }
    }
    if (input.size() != Sha256Digest{}.size() + path_bytes) { return std::nullopt; }

    const FingerprintFileIdentity recorded{
        .size              = fields[0],
        .device            = fields[1],
        .inode             = fields[2],
        .mtime_seconds     = static_cast<std::int64_t>(fields[3]),
        .mtime_nanoseconds = static_cast<std::int64_t>(fields[4]),
        .ctime_seconds     = static_cast<std::int64_t>(fields[5]),
        .ctime_nanoseconds = static_cast<std::int64_t>(fields[6]),
    };
    if (recorded != context.identity) { return std::nullopt; }

    Sha256Digest digest{};
    std::copy_n(reinterpret_cast<const std::uint8_t*>(input.data()), digest.size(), digest.begin());
    input = input.subspan(digest.size());
    const std::string_view recorded_path(reinterpret_cast<const char*>(input.data()), input.size());
    if (recorded_path != context.canonical_path) { return std::nullopt; }
    return digest;
}

void ensure_private_directory(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec || ::chmod(path.c_str(), 0700) != 0 || !private_owned_directory(path)) {
        throw std::runtime_error("cannot secure artifact fingerprint cache directory");
    }
}

void write_fingerprint_record(const std::filesystem::path& path,
                              const FingerprintFileContext& context,
                              const Sha256Digest& digest) {
    if (context.canonical_path.empty() || context.canonical_path.size() > kMaxCanonicalPathBytes) {
        return;
    }
    ensure_private_directory(path.parent_path().parent_path());
    ensure_private_directory(path.parent_path());

    std::vector<std::byte> bytes;
    bytes.reserve(104 + context.canonical_path.size());
    bytes.insert(bytes.end(), kFingerprintRecordMagic.begin(), kFingerprintRecordMagic.end());
    append_u32(bytes, kFingerprintRecordVersion);
    append_u32(bytes, static_cast<std::uint32_t>(context.canonical_path.size()));
    append_u64(bytes, context.identity.size);
    append_u64(bytes, context.identity.device);
    append_u64(bytes, context.identity.inode);
    append_u64(bytes, static_cast<std::uint64_t>(context.identity.mtime_seconds));
    append_u64(bytes, static_cast<std::uint64_t>(context.identity.mtime_nanoseconds));
    append_u64(bytes, static_cast<std::uint64_t>(context.identity.ctime_seconds));
    append_u64(bytes, static_cast<std::uint64_t>(context.identity.ctime_nanoseconds));
    bytes.insert(bytes.end(), reinterpret_cast<const std::byte*>(digest.data()),
                 reinterpret_cast<const std::byte*>(digest.data() + digest.size()));
    bytes.insert(bytes.end(), reinterpret_cast<const std::byte*>(context.canonical_path.data()),
                 reinterpret_cast<const std::byte*>(context.canonical_path.data() +
                                                    context.canonical_path.size()));

    static std::atomic<std::uint64_t> sequence = 0;
    const auto temp = path.parent_path() /
                      (path.filename().string() + ".tmp." + std::to_string(::getpid()) + "." +
                       std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
    const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) { throw std::runtime_error("cannot create artifact fingerprint cache record"); }
    std::size_t completed = 0;
    while (completed < bytes.size()) {
        const ssize_t amount = ::write(fd, bytes.data() + completed, bytes.size() - completed);
        if (amount < 0 && errno == EINTR) { continue; }
        if (amount <= 0) {
            const int ignored = ::close(fd);
            (void)ignored;
            std::error_code ec;
            std::filesystem::remove(temp, ec);
            throw std::runtime_error("cannot write artifact fingerprint cache record");
        }
        completed += static_cast<std::size_t>(amount);
    }
    const bool file_synced = ::fsync(fd) == 0;
    const bool file_closed = ::close(fd) == 0;
    if (!file_synced || !file_closed || ::rename(temp.c_str(), path.c_str()) != 0) {
        std::error_code ec;
        std::filesystem::remove(temp, ec);
        throw std::runtime_error("cannot publish artifact fingerprint cache record");
    }
    const int directory_fd = ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory_fd < 0) {
        throw std::runtime_error("cannot sync artifact fingerprint cache directory");
    }
    const bool directory_synced = ::fsync(directory_fd) == 0;
    const bool directory_closed = ::close(directory_fd) == 0;
    if (!directory_synced || !directory_closed) {
        throw std::runtime_error("cannot sync artifact fingerprint cache directory");
    }
}
#endif

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
        mapping_file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (mapping_file_ == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFileW " + path.string());
        }

        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(mapping_file_, &file_size)) {
            const auto error = ::GetLastError();
            ::CloseHandle(mapping_file_);
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "GetFileSizeEx " + path.string());
        }
        if (file_size.QuadPart < 0 ||
            static_cast<std::uint64_t>(file_size.QuadPart) >
                static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            ::CloseHandle(mapping_file_);
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw ArtifactError("artifact size does not fit the process address space");
        }

        size_ = static_cast<std::size_t>(file_size.QuadPart);
        if (size_ != 0) {
            mapping_ = ::CreateFileMappingW(mapping_file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_ == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_file_);
                mapping_file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "CreateFileMappingW " + path.string());
            }
            const void* view = ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0);
            if (view == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_);
                ::CloseHandle(mapping_file_);
                mapping_      = nullptr;
                mapping_file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "MapViewOfFile " + path.string());
            }
            data_ = static_cast<const std::byte*>(view);
        }

        direct_file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                     OPEN_EXISTING,
                                     FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                         FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN,
                                     nullptr);
        if (direct_file_ == INVALID_HANDLE_VALUE) {
            const auto error = ::GetLastError();
            if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
            if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
            ::CloseHandle(mapping_file_);
            data_         = nullptr;
            mapping_      = nullptr;
            mapping_file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "CreateFileW direct " + path.string());
        }
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
#endif
    }

    ~MappedFile() {
#ifdef _WIN32
        if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
        if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
        if (direct_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(direct_file_); }
        if (mapping_file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(mapping_file_); }
#else
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

#ifndef _WIN32
    std::optional<FingerprintFileContext>
    fingerprint_context(const std::filesystem::path& path) const {
        struct stat opened_status {};
        if (::fstat(fd_, &opened_status) != 0 || !S_ISREG(opened_status.st_mode) ||
            opened_status.st_size < 0) {
            return std::nullopt;
        }
        std::error_code ec;
        const auto canonical = std::filesystem::canonical(path, ec);
        if (ec) { return std::nullopt; }
        const std::string canonical_path = canonical.string();
        if (canonical_path.empty() || canonical_path.size() > kMaxCanonicalPathBytes) {
            return std::nullopt;
        }
        struct stat path_status {};
        if (::stat(canonical.c_str(), &path_status) != 0 || !S_ISREG(path_status.st_mode)) {
            return std::nullopt;
        }
        const auto opened_identity = file_identity(opened_status);
        if (file_identity(path_status) != opened_identity) { return std::nullopt; }
        return FingerprintFileContext{canonical_path, opened_identity};
    }
#endif

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
#ifdef _WIN32
        std::size_t total = 0;
        while (total < destination.size()) {
            constexpr std::size_t max_read = 1ULL << 30;
            const auto amount = static_cast<DWORD>(std::min(max_read, destination.size() - total));
            const std::uint64_t offset = absolute_offset + total;
            OVERLAPPED operation{};
            operation.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
            operation.OffsetHigh = static_cast<DWORD>(offset >> 32U);

            DWORD bytes = 0;
            const BOOL started = ::ReadFile(direct_file_, destination.data() + total, amount,
                                            &bytes, &operation);
            if (!started) {
                const auto error = ::GetLastError();
                if (error == ERROR_HANDLE_EOF) { break; }
                if (error != ERROR_IO_PENDING ||
                    !::GetOverlappedResult(direct_file_, &operation, &bytes, TRUE)) {
                    const auto final_error = error == ERROR_IO_PENDING ? ::GetLastError() : error;
                    throw std::system_error(static_cast<int>(final_error), std::system_category(),
                                            "direct artifact read");
                }
            }
            total += bytes;
            if (bytes != amount) { break; }
        }
        return total;
#else
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
#endif
    }

private:
#ifdef _WIN32
    HANDLE mapping_file_       = INVALID_HANDLE_VALUE;
    HANDLE direct_file_        = INVALID_HANDLE_VALUE;
    HANDLE mapping_            = nullptr;
#else
    int fd_                = -1;
#endif
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path,
                  const FingerprintCacheOptions& fingerprint_cache,
                  const FingerprintProgress& fingerprint_progress)
        : file(path) {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the v2 prefix");
        }
        const bool legacy_v1 = std::equal(kV1Magic.begin(), kV1Magic.end(), file.data());
        if (!legacy_v1 && !std::equal(kMagic.begin(), kMagic.end(), file.data())) {
            throw ArtifactError("artifact magic is not NInfer v1 or v2");
        }

        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kPrefixBytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kPrefixBytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        if (legacy_v1) {
            static constexpr std::array root_members = {"model_id", "objects"};
            require_members(directory, root_members, "legacy directory root");
            identity.model_id = require_string(directory.at("model_id"), "model_id");
            if (identity.model_id != "qwen3.8-27b" && identity.model_id != "qwen3.6-35b-a3b") {
                throw ArtifactError("legacy artifact model_id is not a registered groupwise target");
            }
            identity.weights_id = "groupwise-int";
        } else {
            static constexpr std::array root_members = {"identity", "objects"};
            require_members(directory, root_members, "directory root");
            const auto& raw_identity = directory.at("identity");
            static constexpr std::array identity_members = {"model_id", "weights_id"};
            require_members(raw_identity, identity_members, "artifact identity");
            identity.model_id = require_string(raw_identity.at("model_id"), "model_id");
            identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");
        }

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            cursor = end;
        }

        bool cache_hit = false;
#ifndef _WIN32
        std::optional<FingerprintFileContext> fingerprint_context;
        std::filesystem::path fingerprint_path;
        if (!fingerprint_cache.directory.empty() &&
            safe_namespace(fingerprint_cache.cache_namespace)) {
            try {
                fingerprint_context = file.fingerprint_context(path);
                if (fingerprint_context) {
                    fingerprint_path = fingerprint_record_path(fingerprint_cache,
                                                               fingerprint_context->canonical_path);
                    if (const auto cached =
                            load_fingerprint_record(fingerprint_path, *fingerprint_context)) {
                        const auto current = file.fingerprint_context(path);
                        if (current && *current == *fingerprint_context) {
                            fingerprint = *cached;
                            cache_hit   = true;
                        }
                    }
                }
            } catch (...) {
                fingerprint_context.reset();
            }
        }
#endif
        if (cache_hit) {
            if (fingerprint_progress) {
                fingerprint_progress(FingerprintProgressPhase::CacheHit, file.size(), file.size());
            }
        } else {
            Sha256 sha;
            constexpr std::size_t kFingerprintChunkBytes = 1U << 20;
            if (fingerprint_progress) {
                fingerprint_progress(FingerprintProgressPhase::Scan, 0, file.size());
            }
            for (std::size_t offset = 0; offset < file.size();) {
                const std::size_t amount = std::min(kFingerprintChunkBytes, file.size() - offset);
                sha.update(std::span<const std::byte>(file.data() + offset, amount));
                offset += amount;
                if (fingerprint_progress) {
                    fingerprint_progress(FingerprintProgressPhase::Scan, offset, file.size());
                }
            }
            fingerprint = sha.finish();
#ifndef _WIN32
            if (fingerprint_context) {
                try {
                    const auto current = file.fingerprint_context(path);
                    if (current && *current == *fingerprint_context) {
                        write_fingerprint_record(fingerprint_path, *fingerprint_context,
                                                 fingerprint);
                    }
                } catch (...) {}
            }
#endif
        }
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
    Sha256Digest fingerprint{};
};

Reader::Reader(const std::filesystem::path& path, FingerprintCacheOptions fingerprint_cache,
               FingerprintProgress fingerprint_progress)
    : impl_(std::make_unique<Impl>(path, fingerprint_cache, fingerprint_progress)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

const Sha256Digest& Reader::content_fingerprint() const noexcept { return impl_->fingerprint; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
