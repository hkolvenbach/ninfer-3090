#include "artifact/reader.h"
#include "artifact/sha256.h"
#include "artifact_fixture.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
using Json = nlohmann::json;
using ninfer::test::artifact_fixture::write_fixture;

Json normative_directory() {
    return {
        {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
        {"objects", Json::array({
                        {{"name", "resource"},
                         {"kind", "resource"},
                         {"encoding", "raw-bytes-v1"},
                         {"offset", 0},
                         {"bytes", 3}},
                        {{"name", "bf16"},
                         {"kind", "tensor"},
                         {"shape", {2, 3}},
                         {"format", "BF16"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 256},
                         {"bytes", 12}},
                        {{"name", "fp32_scalar"},
                         {"kind", "tensor"},
                         {"shape", Json::array()},
                         {"format", "FP32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 512},
                         {"bytes", 4}},
                        {{"name", "i32"},
                         {"kind", "tensor"},
                         {"shape", {2}},
                         {"format", "I32"},
                         {"layout", "contiguous-le-v1"},
                         {"offset", 768},
                         {"bytes", 8}},
                        {{"name", "q4"},
                         {"kind", "tensor"},
                         {"shape", {1, 1}},
                         {"format", "Q4G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1024},
                         {"bytes", 260}},
                        {{"name", "q5"},
                         {"kind", "tensor"},
                         {"shape", {2, 130}},
                         {"format", "Q5G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 1536},
                         {"bytes", 528}},
                        {{"name", "q6"},
                         {"kind", "tensor"},
                         {"shape", {1, 64}},
                         {"format", "Q6G64_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 2304},
                         {"bytes", 516}},
                        {{"name", "w8"},
                         {"kind", "tensor"},
                         {"shape", {1, 33}},
                         {"format", "W8G32_F16S"},
                         {"layout", "row-split-k128-v1"},
                         {"offset", 3072},
                         {"bytes", 264}},
                    })},
    };
}

template <typename Function>
void expect_artifact_error(Function&& function, std::string_view label) {
    try {
        function();
    } catch (const ninfer::artifact::ArtifactError&) { return; }
    throw std::runtime_error(std::string(label) + " was accepted");
}

void test_registered_sizes() {
    using ninfer::artifact::tensor_encoded_size;
    constexpr StorageLayout direct = StorageLayout::ContiguousLeV1;
    constexpr StorageLayout rows   = StorageLayout::RowSplitK128V1;

    const std::array<std::uint64_t, 2> shape_2x3 = {2, 3};
    const std::array<std::uint64_t, 1> shape_2   = {2};
    const std::array<std::uint64_t, 2> q4_shape  = {1, 1};
    const std::array<std::uint64_t, 2> q5_shape  = {2, 130};
    const std::array<std::uint64_t, 2> q6_shape  = {1, 64};
    const std::array<std::uint64_t, 2> w8_shape  = {1, 33};

    if (tensor_encoded_size(direct, NumericFormat::BF16, shape_2x3) != 12 ||
        tensor_encoded_size(direct, NumericFormat::FP32, {}) != 4 ||
        tensor_encoded_size(direct, NumericFormat::I32, shape_2) != 8 ||
        tensor_encoded_size(rows, NumericFormat::Q4G64_F16S, q4_shape) != 260 ||
        tensor_encoded_size(rows, NumericFormat::Q5G64_F16S, q5_shape) != 528 ||
        tensor_encoded_size(rows, NumericFormat::Q6G64_F16S, q6_shape) != 516 ||
        tensor_encoded_size(rows, NumericFormat::W8G32_F16S, w8_shape) != 264) {
        throw std::runtime_error("registered encoded-size calculation is wrong");
    }
}

void test_normative_fixture() {
    auto fixture = write_fixture(normative_directory(), "valid");
    Reader reader(fixture.path);
    if (reader.identity().model_id != "fixture-model" ||
        reader.identity().weights_id != "fixture-weights" || reader.objects().size() != 8 ||
        reader.payload_offset() != 4096) {
        throw std::runtime_error("fixture root descriptor mismatch");
    }

    const std::array<std::string_view, 8> expected_names = {
        "resource", "bf16", "fp32_scalar", "i32", "q4", "q5", "q6", "w8",
    };
    for (std::size_t i = 0; i < expected_names.size(); ++i) {
        const auto& object = reader.objects()[i];
        if (ninfer::artifact::object_name(object) != expected_names[i] ||
            reader.find(expected_names[i]) != &object) {
            throw std::runtime_error("fixture name index mismatch");
        }
        const auto payload = reader.payload(object);
        if (payload.absolute_offset !=
                reader.payload_offset() + ninfer::artifact::object_offset(object) ||
            payload.data.size() != ninfer::artifact::object_bytes(object) ||
            payload.data.front() != std::byte(i + 1) || payload.data.back() != std::byte(i + 1)) {
            throw std::runtime_error("fixture payload span mismatch");
        }
    }
    if (reader.find("missing") != nullptr) {
        throw std::runtime_error("missing object unexpectedly resolved");
    }

    const auto* resource = std::get_if<ResourceDescriptor>(&reader.objects().front());
    const auto* q5       = std::get_if<TensorDescriptor>(reader.find("q5"));
    if (resource == nullptr || q5 == nullptr || q5->shape != std::vector<std::uint64_t>({2, 130}) ||
        q5->format != NumericFormat::Q5G64_F16S || q5->layout != StorageLayout::RowSplitK128V1) {
        throw std::runtime_error("fixture object signature mismatch");
    }
}

void test_common_validation() {
    {
        auto directory                   = normative_directory();
        directory["objects"][5]["bytes"] = 527;
        auto fixture                     = write_fixture(directory, "wrong_encoded_size");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "wrong encoded size");
    }
    {
        auto directory                    = normative_directory();
        directory["objects"][1]["offset"] = 257;
        auto fixture                      = write_fixture(directory, "misaligned_offset");
        expect_artifact_error([&] { Reader reader(fixture.path); }, "misaligned offset");
    }
    {
        auto normative = normative_directory();
        nlohmann::json directory{
            {"model_id", "qwen3.8-27b"},
            {"objects", normative.at("objects")},
        };
        auto fixture =
            write_fixture(directory, "legacy_v1", ninfer::test::artifact_fixture::kV1Magic);
        Reader reader(fixture.path);
        if (reader.identity().model_id != "qwen3.8-27b" ||
            reader.identity().weights_id != "groupwise-int") {
            throw std::runtime_error("legacy v1 identity mapping mismatch");
        }
    }
}

void test_content_fingerprint() {
    ninfer::artifact::Sha256 known_hash;
    const std::array<std::byte, 3> abc = {std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    known_hash.update(std::span<const std::byte>(abc.data(), 1));
    known_hash.update(std::span<const std::byte>(abc.data() + 1, 2));
    const auto known_digest = known_hash.finish();
    constexpr std::array<std::uint8_t, 32> kAbcSha256 = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
        0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
        0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    if (known_digest != kAbcSha256) {
        throw std::runtime_error("streaming artifact SHA-256 failed the known-answer test");
    }

    auto original = write_fixture(normative_directory(), "fingerprint_original");
    auto changed  = write_fixture(normative_directory(), "fingerprint_changed");
    {
        std::fstream file(changed.path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekg(-1, std::ios::end);
        char byte = 0;
        file.read(&byte, 1);
        byte ^= 0x5a;
        file.seekp(-1, std::ios::end);
        file.write(&byte, 1);
        if (!file) { throw std::runtime_error("failed to mutate fingerprint fixture"); }
    }

    const Reader first(original.path);
    const Reader reopened(original.path);
    const Reader different(changed.path);
    if (first.identity() != different.identity()) {
        throw std::runtime_error("fingerprint fixtures do not have identical labels");
    }
    if (first.content_fingerprint() != reopened.content_fingerprint()) {
        throw std::runtime_error("artifact content fingerprint is unstable for the same file");
    }
    if (first.content_fingerprint() == different.content_fingerprint()) {
        throw std::runtime_error("artifact content fingerprint ignored a payload change");
    }
}

#ifndef _WIN32
struct FingerprintEvent {
    ninfer::artifact::FingerprintProgressPhase phase;
    std::uint64_t done;
    std::uint64_t total;
};

std::vector<FingerprintEvent> open_with_fingerprint_cache(
    const std::filesystem::path& artifact, const std::filesystem::path& cache_root,
    ninfer::artifact::Sha256Digest* digest = nullptr) {
    std::vector<FingerprintEvent> events;
    Reader reader(artifact,
                  {.directory = cache_root, .cache_namespace = "reader-test"},
                  [&](auto phase, std::uint64_t done, std::uint64_t total) {
                      events.push_back({phase, done, total});
                  });
    if (digest != nullptr) { *digest = reader.content_fingerprint(); }
    return events;
}

bool has_phase(const std::vector<FingerprintEvent>& events,
               ninfer::artifact::FingerprintProgressPhase phase) {
    return std::any_of(events.begin(), events.end(),
                       [phase](const auto& event) { return event.phase == phase; });
}

void require_scan(const std::vector<FingerprintEvent>& events, std::string_view label) {
    using Phase = ninfer::artifact::FingerprintProgressPhase;
    if (!has_phase(events, Phase::Scan) || has_phase(events, Phase::CacheHit) || events.empty() ||
        events.back().done != events.back().total) {
        throw std::runtime_error(std::string(label) + " did not perform a complete fingerprint scan");
    }
}

void test_fingerprint_sidecar_cache() {
    using Phase = ninfer::artifact::FingerprintProgressPhase;
    const auto cache_root = std::filesystem::temp_directory_path() /
                            ("ninfer_fingerprint_cache_" + std::to_string(::getpid()));
    std::error_code ignored;
    std::filesystem::remove_all(cache_root, ignored);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    } cleanup{cache_root};

    auto fixture = write_fixture(normative_directory(), "fingerprint_sidecar");
    ninfer::artifact::Sha256Digest original_digest{};
    require_scan(open_with_fingerprint_cache(fixture.path, cache_root, &original_digest),
                 "first cached open");

    const auto artifacts = cache_root / "reader-test" / "artifacts";
    if (!std::filesystem::is_directory(artifacts)) {
        throw std::runtime_error("first fingerprint scan did not create the artifacts cache");
    }
    std::vector<std::filesystem::path> records;
    for (const auto& entry : std::filesystem::directory_iterator(artifacts)) {
        if (entry.is_regular_file()) { records.push_back(entry.path()); }
    }
    if (records.size() != 1) {
        throw std::runtime_error("first fingerprint scan did not create exactly one record");
    }
    struct stat directory_status {};
    struct stat namespace_status {};
    struct stat record_status {};
    if (::stat((cache_root / "reader-test").c_str(), &namespace_status) != 0 ||
        ::stat(artifacts.c_str(), &directory_status) != 0 ||
        ::stat(records.front().c_str(), &record_status) != 0 ||
        (namespace_status.st_mode & 0777) != 0700 || (directory_status.st_mode & 0777) != 0700 ||
        (record_status.st_mode & 0777) != 0600) {
        throw std::runtime_error("fingerprint cache permissions are not owner-only");
    }

    ninfer::artifact::Sha256Digest cached_digest{};
    const auto hit = open_with_fingerprint_cache(fixture.path, cache_root, &cached_digest);
    if (hit.size() != 1 || hit.front().phase != Phase::CacheHit ||
        hit.front().done != hit.front().total || cached_digest != original_digest) {
        throw std::runtime_error("unchanged artifact did not produce an immediate cache hit");
    }

    {
        std::fstream file(fixture.path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekg(-1, std::ios::end);
        char byte = 0;
        file.read(&byte, 1);
        byte ^= 0x33;
        file.seekp(-1, std::ios::end);
        file.write(&byte, 1);
        if (!file) { throw std::runtime_error("failed to mutate cached fixture content"); }
    }
    require_scan(open_with_fingerprint_cache(fixture.path, cache_root), "content change");

    {
        std::ofstream file(fixture.path, std::ios::binary | std::ios::app);
        file.put('\0');
        if (!file) { throw std::runtime_error("failed to grow cached fixture"); }
    }
    require_scan(open_with_fingerprint_cache(fixture.path, cache_root), "size change");

    const auto old_time = std::filesystem::last_write_time(fixture.path);
    std::filesystem::last_write_time(fixture.path, old_time - std::chrono::seconds(5));
    require_scan(open_with_fingerprint_cache(fixture.path, cache_root), "mtime change");

    const auto replacement = fixture.path.string() + ".replacement";
    std::filesystem::copy_file(fixture.path, replacement,
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::rename(replacement, fixture.path);
    require_scan(open_with_fingerprint_cache(fixture.path, cache_root), "inode replacement");

    std::filesystem::resize_file(records.front(), 5);
    require_scan(open_with_fingerprint_cache(fixture.path, cache_root), "truncated record");

    auto uncached = write_fixture(normative_directory(), "fingerprint_no_sidecar");
    Reader no_cache(uncached.path);
    std::size_t record_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(artifacts)) {
        if (entry.is_regular_file()) { ++record_count; }
    }
    if (record_count != 1 || std::filesystem::exists(uncached.path.string() + ".sha256")) {
        throw std::runtime_error("reader without a cache directory wrote cache output");
    }
}
#endif

} // namespace

int main() {
    try {
        test_registered_sizes();
        test_normative_fixture();
        test_common_validation();
        test_content_fingerprint();
#ifndef _WIN32
        test_fingerprint_sidecar_cache();
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
