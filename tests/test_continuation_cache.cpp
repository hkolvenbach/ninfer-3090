#include "runtime/cache/continuation_cache.h"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace {
using namespace ninfer::cache;

struct TempDir {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("ninfer-continuation-cache-" +
                                  std::to_string(
#ifdef _WIN32
                                      static_cast<unsigned long long>(GetCurrentProcessId())
#else
                                      static_cast<unsigned long long>(::getpid())
#endif
                                          ) +
                                  "-" + std::to_string(++sequence));
    inline static unsigned sequence = 0;

    TempDir() { std::filesystem::create_directories(path); }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

ContinuationImage image(std::uint8_t value, std::optional<ContentId> parent = std::nullopt) {
    ContinuationImage out;
    out.compatibility_key = {1, 2, 0, 4};
    out.prefix_identity   = {0, value, 0xff, 0};
    out.frontier_tokens   = 17;
    out.boundary_tokens   = 16;
    out.frontier_prefix_digest = Bytes(32, value);
    out.boundary_prefix_digest = Bytes(32, static_cast<std::uint8_t>(value + 1));
    out.frontier_metadata = {value, 7};
    out.boundary_metadata = {8, value};
    out.parent_id         = std::move(parent);
    out.segments          = {{"dflash.hidden", Bytes(90, value)},
                             {"gdn.recurrent", {3, 2, 1}},
                             {"kv.layers", {0, 1, 0, value}},
                             {"mtp.state", {value, 9}}};
    return out;
}

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::filesystem::path only_file(const std::filesystem::path& directory) {
    for (const auto& item : std::filesystem::directory_iterator(directory))
        if (item.is_regular_file()) return item.path();
    return {};
}

void overwrite_short(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.put('x');
}

void overwrite_u64(const std::filesystem::path& path, std::streamoff offset,
                   std::uint64_t value) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.seekp(offset);
    for (int i = 0; i != 8; ++i) file.put(static_cast<char>(value >> (i * 8)));
}

void overwrite_magic(const std::filesystem::path& path, std::string_view magic) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    file.write(magic.data(), static_cast<std::streamsize>(magic.size()));
}

void test_roundtrip_dedup_restart_and_permissions() {
    TempDir dir;
    CacheConfig config{.enable_l3      = true,
                       .root           = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20,
                       .chunk_bytes    = 32};
    const auto expected = image(42);
    ContentId id;
    std::size_t bytes = 0;
    {
        ContinuationCache cache(config);
        id = cache.store(expected);
        check(id == content_id(expected), "store uses canonical content ID");
        check(cache.get(id) == expected, "full pointer-free image roundtrips");
        bytes = cache.stats().l3_bytes;
        check(cache.store(expected) == id, "identical images deduplicate");
        check(cache.stats().entries == 1 && cache.stats().l3_bytes == bytes,
              "dedup does not consume disk budget");
    }
    ContinuationCache reopened(config);
    check(reopened.stats().entries == 1, "startup discovers committed manifests");
    check(reopened.get(id) == expected, "discovered image loads after restart");
#ifndef _WIN32
    struct stat st {};

    check(::stat(dir.path.c_str(), &st) == 0 && (st.st_mode & 077) == 0,
          "cache root is owner-only");
#endif
}

void test_image_size_matches_canonical_serialization() {
    TempDir dir;
    CacheConfig config{.enable_l3      = true,
                       .root           = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20,
                       .chunk_bytes    = 32};
    ContinuationCache cache(config);
    const auto expected = image(43, ContentId{std::string(64, 'a')});
    (void)cache.store(expected);
    check(continuation_image_bytes(expected) == cache.stats().l2_bytes,
          "allocation-free image size matches canonical persisted bytes");
}

void test_previous_development_manifest_is_ignored() {
    TempDir dir;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20};
    ContentId id;
    {
        ContinuationCache cache(config);
        id = cache.store(image(44));
    }
    overwrite_magic(dir.path / "manifests" / (id.hex + ".manifest"), "NICMAN03");
    ContinuationCache reopened(config);
    check(reopened.stats().entries == 0 && !reopened.get(id),
          "the previous development manifest format is ignored without migration");
}

void test_l2_eviction() {
    TempDir dir;
    CacheConfig config{.enable_l3      = true,
                       .root           = dir.path,
                       .l2_byte_budget = 400,
                       .l3_byte_budget = 1 << 20,
                       .chunk_bytes    = 4096};
    ContinuationCache cache(config);
    const auto first  = cache.store(image(1));
    const auto second = cache.store(image(2));
    check(cache.stats().l2_bytes <= config.l2_byte_budget, "L2 obeys byte budget");
    check(cache.stats().l2_bytes != 0, "L2 budget admits exactly one test image");
    overwrite_short(dir.path / "chunks" / first.hex); // Whole-image chunks use the image ID.
    check(!cache.get(first), "evicted L2 entry observes corrupt L3 as a miss");
    check(cache.get(second).has_value(), "most-recent L2 entry survives eviction");
}

void test_shared_hits_pin_one_charged_image() {
    CacheConfig config{.l2_byte_budget = 400};
    ContinuationCache cache(config);
    const auto id = cache.admit(image(80));
    const auto before = cache.stats();
    auto first = cache.get_shared(id);
    auto second = cache.get_shared(id);
    check(first && second && first.get() == second.get(),
          "shared L2 hits return the same immutable allocation");
    check(cache.stats().l2_bytes == before.l2_bytes && cache.stats().l2_entries == 1,
          "shared hits do not duplicate L2 budget accounting");
    (void)cache.admit(image(81));
    check(cache.get_shared(id).get() == first.get(),
          "a shared hit pins its charged L2 allocation under pressure");
}

void test_persistence_policy_helpers() {
    using Clock = std::chrono::system_clock;
    const auto start = Clock::time_point{} + std::chrono::hours(1);
    const PersistencePolicy policy{.interval = std::chrono::seconds(60), .min_tokens = 100};
    check(!persistence_due(policy, std::nullopt, 99, start, start + std::chrono::seconds(59)),
          "first persistence waits for both interval and token growth");
    check(persistence_due(policy, std::nullopt, 100, start, start),
          "first persistence is due at the token threshold");
    const PersistedState previous{.tokens = 1000, .at = start};
    check(!persistence_due(policy, previous, 1099, start, start + std::chrono::seconds(59)),
          "subsequent persistence measures growth from the durable frontier");
    check(persistence_due(policy, previous, 1100, start, start),
          "token growth triggers persistence before the interval");
    check(persistence_due(policy, previous, 1001, start, start + std::chrono::seconds(60)),
          "elapsed interval triggers persistence with small growth");
    const PersistencePolicy threshold_only{.interval = std::chrono::seconds(0),
                                           .min_tokens = 100};
    check(!persistence_due(threshold_only, previous, 1099, start,
                           start + std::chrono::hours(24)),
          "zero interval disables only the timer trigger");
    check(persistence_due(threshold_only, previous, 1100, start, start),
          "token growth still triggers with a zero interval");
    check(persistence_due(PersistencePolicy{.interval = std::chrono::seconds(60), .min_tokens = 0},
                           previous, 1001, start, start),
          "zero token threshold makes every queued state due");
    check(persistence_due(PersistencePolicy{.interval = std::chrono::seconds(0), .min_tokens = 0},
                           previous, 1001, start, start),
          "zero token threshold remains due when the timer is disabled");
}

void test_l2_cost_aware_admission_retains_value() {
    CacheConfig config{.l2_byte_budget = 500};
    ContinuationCache cache(config);
    const auto valuable = cache.admit(image(60), StoreOptions{.recompute_cost = 100000.0});
    for (std::uint8_t value = 61; value != 68; ++value) {
        (void)cache.admit(image(value), StoreOptions{.recompute_cost = 1.0});
    }
    check(cache.get(valuable).has_value(),
          "large recompute value survives a scan of low-value L2 admissions");
    check(cache.stats().l2_bytes <= config.l2_byte_budget,
          "cost-aware L2 admission remains within its byte budget");
}

void test_l2_admission_promotion_and_restart() {
    TempDir dir;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20};
    const auto expected = image(68);
    ContentId id;
    {
        ContinuationCache cache(config);
        id = cache.admit(expected);
        check(cache.get(id) == expected && cache.stats().l3_entries == 0,
              "L2 admission is visible without a durable manifest");
        check(cache.promote(id) && cache.stats().l3_entries == 1,
              "an existing nondurable content ID can be promoted");
    }
    ContinuationCache reopened(config);
    check(reopened.get(id) == expected, "explicitly promoted content survives restart");
}

void test_l2_session_publication_is_decoupled_from_filesystem() {
    TempDir dir;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20};
    ContinuationCache cache(config);
    const auto expected = image(85);
    const auto result = cache.publish_session_l2(expected, "memory-first", std::nullopt);
    check(result.stored && result.alias_advanced && cache.get(result.id) == expected,
          "L2 session publication remains immediately visible");
    check(only_file(dir.path / "chunks").empty() &&
              only_file(dir.path / "manifests").empty() &&
              only_file(dir.path / "aliases").empty(),
          "L2 session publication performs no persistence I/O");
}

void test_l2_access_does_not_wait_for_persistence_io() {
    TempDir dir;
    std::atomic<bool> entered = false;
    std::atomic<bool> release = false;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20,
                       .before_persistence_io = [&] {
                           entered.store(true, std::memory_order_release);
                           while (!release.load(std::memory_order_acquire)) {
                               std::this_thread::yield();
                           }
                       }};
    ContinuationCache cache(config);
    const auto id = cache.admit(image(84));
    std::thread promotion([&] { (void)cache.promote(id); });
    while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
    auto hit = cache.get_shared(id);
    check(hit && hit->prefix_identity == image(84).prefix_identity,
          "L2 access proceeds while persistence I/O is blocked");
    release.store(true, std::memory_order_release);
    promotion.join();
}

void test_async_coalescing_shutdown_and_restart() {
    TempDir dir;
    auto now = std::chrono::system_clock::time_point{} + std::chrono::hours(2);
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20,
                       .persist_interval = std::chrono::seconds(0),
                       .persist_min_tokens = 1000,
                       .now = [&] { return now; }};
    ContentId latest;
    {
        ContinuationCache cache(config);
        auto first_image = image(69);
        first_image.frontier_tokens = 100;
        const auto first = cache.publish_session_l2(first_image, "coalesced", std::nullopt);
        check(first.alias_advanced && cache.queue_persistence("coalesced", first.id, 100),
              "first L2 session state queues for persistence");
        auto latest_image = image(70, first.id);
        latest_image.frontier_tokens = 150;
        const auto second = cache.publish_session_l2(latest_image, "coalesced", first.id);
        latest = second.id;
        check(second.alias_advanced && cache.queue_persistence("coalesced", second.id, 150),
              "newer same-session state replaces the pending persistence candidate");
        const auto stats = cache.stats();
        check(cache.session_current("coalesced") == latest && stats.l3_entries == 0,
              "L2 session visibility does not wait for a persistence threshold");
        check(stats.persistence_queued == 2 && stats.persistence_coalesced == 1,
              "pending persistence is counted and coalesced per alias");
    }
    ContinuationCache reopened(config);
    check(reopened.session_current("coalesced") == latest && reopened.get(latest).has_value(),
          "orderly shutdown promotes the latest coalesced state and durable alias");
    check(!reopened.get(content_id([&] {
              auto first = image(69);
              first.frontier_tokens = 100;
              return first;
          }())),
          "coalescing avoids the redundant intermediate manifest");
}

void test_async_stable_alias_first_writer_and_restart() {
    TempDir dir;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20,
                       .persist_interval = std::chrono::hours(1),
                       .persist_min_tokens = 100000};
    const std::string alias = std::string(kStableAliasPrefix) + std::string(64, 'b');
    const auto first_image = image(71);
    const auto second_image = image(72);
    ContentId first;
    {
        ContinuationCache cache(config);
        first = cache.admit(first_image);
        const auto second = cache.admit(second_image);
        check(cache.publish_immutable_alias(alias, first) &&
                  cache.session_current(alias) == first,
              "stable alias is promptly visible from L2");
        check(!cache.publish_immutable_alias(alias, second),
              "L2 stable alias remains first-writer immutable");
        check(cache.queue_persistence(alias, first, first_image.frontier_tokens, true),
              "stable L2 alias queues its durable promotion");
    }
    ContinuationCache reopened(config);
    check(reopened.session_current(alias) == first && reopened.get(first) == first_image,
          "stable alias is written only after its manifest is durable");
}

void test_pin_protects_from_l3_eviction() {
    TempDir dir;
    CacheConfig config{.enable_l3      = true,
                       .root           = dir.path,
                       .l2_byte_budget = 0,
                       .l3_byte_budget = 700,
                       .chunk_bytes    = 4096};
    ContinuationCache cache(config);
    const auto pinned = cache.store(image(6));
    check(cache.pin(pinned), "existing image can be pinned");
    const auto rejected = cache.store(image(7));
    check(cache.get(pinned).has_value(), "pin protects image under L3 pressure");
    check(!cache.get(rejected), "unpinned image is evicted before pinned image");
    cache.unpin(pinned);
    const auto replacement = cache.store(image(8));
    check(cache.get(replacement).has_value() && !cache.get(pinned) &&
              cache.stats().l3_bytes <= config.l3_byte_budget,
          "promotion evicts an unpinned durable image to remain within the L3 budget");
}

void test_l3_expiry_defers_cleanup_until_restore_and_explicit_pins_release() {
    TempDir dir;
    auto now = std::chrono::system_clock::time_point{} + std::chrono::hours(1);
    std::atomic<bool> block_restore = true;
    std::atomic<bool> restore_entered = false;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 0,
                       .l3_byte_budget = 1 << 20,
                       .l3_idle_ttl = std::chrono::seconds(10),
                       .now = [&] { return now; },
                       .before_l3_restore_io = [&] {
                           restore_entered.store(true, std::memory_order_release);
                           while (block_restore.load(std::memory_order_acquire)) {
                               std::this_thread::yield();
                           }
                       }};
    ContinuationCache cache(config);
    const auto restoring = cache.store(image(90));
    std::shared_ptr<const ContinuationImage> restored;
    std::thread restore([&] { restored = cache.get_shared(restoring); });
    while (!restore_entered.load(std::memory_order_acquire)) std::this_thread::yield();
    now += std::chrono::seconds(11);
    check(!cache.get(restoring), "expired L3 is unavailable while an older restore pins it");
    check(std::filesystem::exists(dir.path / "manifests" / (restoring.hex + ".manifest")) &&
              std::filesystem::exists(dir.path / "chunks" / restoring.hex),
          "restore pin defers expired L3 file deletion");
    block_restore.store(false, std::memory_order_release);
    restore.join();
    check(static_cast<bool>(restored), "restore already in flight may finish after expiry");
    restored.reset();
    check(!std::filesystem::exists(dir.path / "manifests" / (restoring.hex + ".manifest")) &&
              !std::filesystem::exists(dir.path / "chunks" / restoring.hex),
          "final restore unpin performs deferred L3 cleanup");

    block_restore.store(false, std::memory_order_release);
    restore_entered.store(false, std::memory_order_release);
    const auto explicitly_pinned = cache.store(image(91));
    check(cache.pin(explicitly_pinned), "explicit pin acquires a live L3 entry");
    now += std::chrono::seconds(11);
    check(!cache.get(explicitly_pinned), "expired explicitly pinned L3 rejects new lookup");
    check(std::filesystem::exists(dir.path / "manifests" /
                                  (explicitly_pinned.hex + ".manifest")),
          "explicit pin also defers expired L3 deletion");
    cache.unpin(explicitly_pinned);
    check(!std::filesystem::exists(dir.path / "manifests" /
                                   (explicitly_pinned.hex + ".manifest")),
          "explicit final unpin performs deferred cleanup");
}

void test_independent_idle_ttls_and_refresh() {
    TempDir dir;
    auto now = std::chrono::system_clock::time_point{} + std::chrono::hours(1);
    CacheConfig config{.enable_l3   = true,
                       .root        = dir.path,
                       .l2_idle_ttl = std::chrono::seconds(10),
                       .l3_idle_ttl = std::chrono::seconds(30),
                       .now         = [&] { return now; }};
    const auto expected = image(3);
    ContentId id;
    {
        ContinuationCache cache(config);
        id = cache.store(expected);
        now += std::chrono::seconds(11);
        check(cache.get(id) == expected, "expired L2 restores from still-valid L3");
    }
    now += std::chrono::seconds(20);
    ContinuationCache reopened(config);
    check(reopened.get(id) == expected,
          "L3 restore refresh is persisted independently of the L2 expiry");

    TempDir hot_dir;
    now = std::chrono::system_clock::time_point{} + std::chrono::hours(2);
    config.root = hot_dir.path;
    config.l2_idle_ttl = std::chrono::seconds(30);
    config.l3_idle_ttl = std::chrono::seconds(10);
    ContinuationCache hot(config);
    const auto hot_id = hot.store(image(4));
    now += std::chrono::seconds(11);
    check(hot.get(hot_id) == image(4) && hot.stats().l3_entries == 0,
          "expired L3 is removed without invalidating hot L2");
    now += std::chrono::seconds(31);
    check(!hot.get(hot_id), "hot L2 remains independent only until its own idle TTL");

    CacheConfig l2_only{.l2_idle_ttl = std::chrono::seconds(10), .now = [&] { return now; }};
    ContinuationCache refreshed(l2_only);
    const auto refreshed_id = refreshed.store(image(5));
    now += std::chrono::seconds(9);
    check(refreshed.get(refreshed_id).has_value(), "L2 access hits before idle expiry");
    now += std::chrono::seconds(9);
    check(refreshed.get(refreshed_id).has_value(), "L2 access refreshes its idle expiry");

    CacheConfig no_expiry{.enable_l3 = true,
                          .root = dir.path / "no-expiry",
                          .l2_idle_ttl = std::chrono::seconds(0),
                          .l3_idle_ttl = std::chrono::seconds(0),
                          .now = [&] { return now; }};
    ContinuationCache permanent(no_expiry);
    const auto permanent_id = permanent.store(image(6));
    now += std::chrono::hours(24 * 365 * 100);
    check(permanent.get(permanent_id).has_value(), "zero L2/L3 TTL means no expiry");
}

void test_corrupt_truncated_and_remnants() {
    TempDir corrupt_dir;
    CacheConfig config{
        .enable_l3 = true, .root = corrupt_dir.path, .l2_byte_budget = 0, .chunk_bytes = 4096};
    ContinuationCache corrupt(config);
    const auto corrupt_id = corrupt.store(image(4));
    overwrite_short(corrupt_dir.path / "chunks" / corrupt_id.hex);
    check(!corrupt.get(corrupt_id), "corrupt chunk is a cache miss");

    TempDir truncated_dir;
    config.root = truncated_dir.path;
    ContentId truncated_id;
    {
        ContinuationCache cache(config);
        truncated_id = cache.store(image(5));
    }
    overwrite_short(only_file(truncated_dir.path / "manifests"));
    ContinuationCache reopened(config);
    check(reopened.stats().entries == 0 && !reopened.get(truncated_id),
          "truncated manifest is ignored during discovery");

    std::ofstream(truncated_dir.path / "tmp" / "half.tmp", std::ios::binary).put('x');
    std::ofstream(truncated_dir.path / "manifests" / "publish.tmp", std::ios::binary).put('x');
    ContinuationCache with_remnants(config);
    check(with_remnants.stats().entries == 0 &&
              only_file(truncated_dir.path / "tmp").empty() &&
              only_file(truncated_dir.path / "manifests").empty(),
          "atomic publication remnants are removed");
}

void test_oversized_manifest_chunk_and_orphan_cleanup() {
    TempDir manifest_dir;
    CacheConfig config{.enable_l3 = true,
                       .root = manifest_dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20,
                       .chunk_bytes = 4096};
    ContentId manifest_id;
    {
        ContinuationCache cache(config);
        manifest_id = cache.store(image(82));
    }
    overwrite_u64(manifest_dir.path / "manifests" / (manifest_id.hex + ".manifest"), 8,
                  config.l2_byte_budget + 1);
    {
        ContinuationCache reopened(config);
        check(!reopened.get(manifest_id) && reopened.stats().entries == 0,
              "an image declaration exceeding the restore budget is a safe miss");
    }
    check(only_file(manifest_dir.path / "chunks").empty(),
          "chunks belonging only to an invalid manifest are cleaned");

    TempDir chunk_dir;
    config.root = chunk_dir.path;
    ContentId chunk_id;
    {
        ContinuationCache cache(config);
        chunk_id = cache.store(image(83));
    }
    {
        std::ofstream out(chunk_dir.path / "chunks" / chunk_id.hex,
                          std::ios::binary | std::ios::app);
        out.put('x');
    }
    std::ofstream(chunk_dir.path / "tmp" / "abandoned.tmp", std::ios::binary).put('x');
    {
        ContinuationCache reopened(config);
        check(!reopened.get(chunk_id) && reopened.stats().entries == 0,
              "a chunk exceeding its declared/configured size is a safe miss");
    }
    check(only_file(chunk_dir.path / "chunks").empty() &&
              only_file(chunk_dir.path / "tmp").empty() &&
              only_file(chunk_dir.path / "manifests").empty(),
          "orphan chunks, manifests, and temp files are cleaned at startup");
}

void test_sessions_history_and_rollback() {
    CacheConfig config{.l2_byte_budget = 4096, .session_history_depth = 3};
    ContinuationCache cache(config);
    StoreOptions in_chat;
    in_chat.session    = "chat";
    const auto a       = cache.store(image(10), in_chat);
    const auto b       = cache.store(image(11, a), in_chat);
    const auto c       = cache.store(image(12, b), in_chat);
    const auto d       = cache.store(image(13, c), in_chat);
    const auto history = cache.session_history("chat");
    check(history == std::vector<ContentId>({b, c, d}), "session history is depth bounded");
    check(cache.rollback_session("chat", 2), "rollback to retained ancestor succeeds");
    check(cache.session_current("chat") == b, "rollback moves current alias");
    check(cache.session_history("chat") == std::vector<ContentId>({b}),
          "rollback destructively drops newer aliases");
    check(!cache.rollback_session("chat", 1), "rollback beyond retained history fails");
    check(cache.get(d).has_value(), "rollback does not mutate immutable CAS images");
}

void test_session_compare_and_swap_branches_and_rollback() {
    CacheConfig config{.l2_byte_budget = 4096, .session_history_depth = 4};
    ContinuationCache cache(config);
    const auto root = cache.publish_session(image(50), "chat", std::nullopt);
    check(root.alias_advanced, "CAS can explicitly publish an empty session");

    SessionPublishResult first;
    SessionPublishResult second;
    std::thread first_branch([&] {
        first = cache.publish_session(image(51, root.id), "chat", root.id);
    });
    std::thread second_branch([&] {
        second = cache.publish_session(image(52, root.id), "chat", root.id);
    });
    first_branch.join();
    second_branch.join();
    const SessionPublishResult& winner = first.alias_advanced ? first : second;
    const SessionPublishResult& stale = first.alias_advanced ? second : first;
    check(first.alias_advanced != second.alias_advanced,
          "only the first concurrent branch advances the session");
    check(cache.session_current("chat") == winner.id,
          "stale branch cannot replace the winning head");
    check(cache.get(stale.id).has_value(), "stale branch remains content-addressable");

    const auto newer = cache.publish_session(image(53, winner.id), "chat", winner.id);
    check(newer.alias_advanced && cache.rollback_session("chat", 1),
          "newer branch can be rolled back");
    const auto discarded = cache.publish_session(image(54, newer.id), "chat", newer.id);
    check(!discarded.alias_advanced && cache.session_current("chat") == winner.id,
          "publication based on a discarded head cannot resurrect it after rollback");
    const auto continued = cache.publish_session(image(55, winner.id), "chat", winner.id);
    check(continued.alias_advanced && cache.session_current("chat") == continued.id,
          "continuation from the rolled-back head advances normally");
}

void test_session_compare_and_swap_restart_persistence() {
    TempDir dir;
    CacheConfig config{.enable_l3             = true,
                       .root                  = dir.path,
                       .l2_byte_budget        = 0,
                       .l3_byte_budget        = 1 << 20,
                       .session_history_depth = 4};
    ContentId winner;
    ContentId stale;
    {
        ContinuationCache cache(config);
        const auto root   = cache.publish_session(image(56), "durable", std::nullopt);
        const auto branch = cache.publish_session(image(57, root.id), "durable", root.id);
        const auto loser  = cache.publish_session(image(58, root.id), "durable", root.id);
        check(root.alias_advanced && branch.alias_advanced && !loser.alias_advanced,
              "durable CAS chooses one branch");
        winner = branch.id;
        stale  = loser.id;
    }
    ContinuationCache reopened(config);
    check(reopened.session_current("durable") == winner,
          "CAS-selected session head survives restart");
    check(reopened.get(stale).has_value(), "durable stale image survives without becoming head");
}

void test_exact_id_rollback_generation_and_shared_candidates() {
    TempDir dir;
    CacheConfig config{.enable_l3             = true,
                       .root                  = dir.path,
                       .l2_byte_budget        = 4096,
                       .l3_byte_budget        = 1 << 20,
                       .session_history_depth = 3};
    ContentId selected;
    {
        ContinuationCache cache(config);
        const auto first = cache.publish_session(image(100), "exact", std::nullopt);
        const auto second = cache.publish_session(image(101, first.id), "exact", first.id);
        const auto third = cache.publish_session(image(102, second.id), "exact", second.id);
        selected = second.id;

        const auto before = cache.stats();
        auto candidates = cache.session_candidates("exact");
        check(candidates.newest_to_oldest.size() == 3 &&
                  candidates.newest_to_oldest[0].id == third.id &&
                  candidates.newest_to_oldest[1].id == second.id &&
                  candidates.newest_to_oldest[2].id == first.id,
              "candidate snapshot is bounded and newest-to-oldest");
        check(cache.stats().l2_bytes == before.l2_bytes && cache.stats().l2_entries == 3 &&
                  !candidates.newest_to_oldest[0].compatibility_key.empty(),
              "candidate snapshot exposes metadata without duplicating L2 accounting");
        const auto resolved = cache.resolve_candidate(candidates.newest_to_oldest[1]);
        check(resolved.image && resolved.image->prefix_identity == image(101).prefix_identity,
              "a selected descriptor resolves its complete immutable image");

        const auto stale_generation = candidates.generation;
        const auto advanced = cache.publish_session(image(103, third.id), "exact", third.id);
        check(advanced.alias_advanced &&
                  !cache.rollback_session_to("exact", second.id, stale_generation).rolled_back,
              "concurrent head advancement defeats stale exact-ID rollback");

        candidates = cache.session_candidates("exact");
        const auto rollback =
            cache.rollback_session_to("exact", selected, candidates.generation);
        check(rollback.rolled_back &&
                  cache.session_history("exact") == std::vector<ContentId>({selected}),
              "exact-ID rollback atomically prunes every newer alias");
        const auto stale = cache.publish_session(image(104, selected), "exact", selected, {},
                                                 candidates.generation);
        check(!stale.alias_advanced,
              "a pre-rollback generation cannot publish from the selected parent");
        const auto continued = cache.publish_session(image(105, selected), "exact", selected, {},
                                                     rollback.generation);
        check(continued.alias_advanced && cache.session_current("exact") == continued.id,
              "completion fenced by the rollback generation advances from its selected parent");
    }
    ContinuationCache reopened(config);
    const auto history = reopened.session_history("exact");
    check(history == std::vector<ContentId>({selected, content_id(image(105, selected))}),
          "rollback pruning and its continuation survive restart");
}

void test_corrupt_historical_candidate_keeps_other_shared_leases() {
    TempDir dir;
    std::atomic<unsigned> payload_reads = 0;
    CacheConfig config{.enable_l3             = true,
                       .root                  = dir.path,
                       .l2_byte_budget        = 0,
                       .l3_byte_budget        = 1 << 20,
                       .session_history_depth = 3,
                       .before_l3_restore_io = [&] { ++payload_reads; }};
    ContinuationCache cache(config);
    const auto first = cache.publish_session(image(106), "corrupt-history", std::nullopt);
    const auto second =
        cache.publish_session(image(107, first.id), "corrupt-history", first.id);
    const auto third =
        cache.publish_session(image(108, second.id), "corrupt-history", second.id);
    overwrite_short(dir.path / "chunks" / second.id.hex);
    const auto candidates = cache.session_candidates("corrupt-history");
    check(candidates.newest_to_oldest.size() == 3 &&
              candidates.newest_to_oldest[0].status == CacheLookupStatus::Hit &&
              candidates.newest_to_oldest[1].status == CacheLookupStatus::Hit &&
              candidates.newest_to_oldest[2].status == CacheLookupStatus::Hit &&
              payload_reads.load() == 0,
          "history discovery remains metadata-only even when an unselected chunk is corrupt");
    auto mismatched = candidates.newest_to_oldest[0];
    mismatched.compatibility_key.push_back(0xff);
    check(!cache.resolve_candidate(mismatched).image && payload_reads.load() == 0,
          "descriptor mismatch is rejected before payload I/O");
    check(!cache.resolve_candidate(candidates.newest_to_oldest[1]).image &&
              cache.resolve_candidate(candidates.newest_to_oldest[0]).image &&
              payload_reads.load() == 2,
          "corruption is authoritative when a descriptor is resolved and does not lose peers");
    check(candidates.newest_to_oldest[0].id == third.id,
          "corrupt history does not disturb newest-to-oldest identity slots");
}

void test_session_alias_restart_and_rollback_restart() {
    TempDir dir;
    CacheConfig config{.enable_l3             = true,
                       .root                  = dir.path,
                       .l2_byte_budget        = 0,
                       .l3_byte_budget        = 1 << 20,
                       .session_history_depth = 3};
    StoreOptions options;
    options.session = "chat/with unsafe path";
    ContentId b;
    {
        ContinuationCache cache(config);
        const auto a = cache.store(image(30), options);
        b            = cache.store(image(31, a), options);
        const auto c = cache.store(image(32, b), options);
        check(cache.session_history(*options.session) == std::vector<ContentId>({a, b, c}),
              "durable session history is retained in memory");
    }
    {
        ContinuationCache reopened(config);
        check(reopened.session_current(*options.session) == content_id(image(32, b)),
              "session current alias survives restart");
        check(reopened.rollback_session(*options.session, 1), "durable rollback succeeds");
        check(reopened.session_current(*options.session) == b,
              "durable rollback selects the prior image");
    }
    ContinuationCache rolled_back(config);
    check(rolled_back.session_history(*options.session) ==
              std::vector<ContentId>({content_id(image(30)), b}),
          "destructive rollback survives restart");
#ifndef _WIN32
    struct stat st {};

    const auto alias = only_file(dir.path / "aliases");
    check(!alias.empty() && ::stat(alias.c_str(), &st) == 0 && (st.st_mode & 077) == 0,
          "published alias is owner-only");
#endif
}

void test_stale_alias_write_cannot_resurrect_rolled_back_state() {
    TempDir dir;
    std::atomic<bool> block = false;
    std::atomic<bool> entered = false;
    std::atomic<bool> release = false;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20,
                       .session_history_depth = 4,
                       .before_alias_persistence_io = [&](std::string_view alias) {
                           if (alias != "generation" ||
                               !block.exchange(false, std::memory_order_acq_rel)) return;
                           entered.store(true, std::memory_order_release);
                           while (!release.load(std::memory_order_acquire)) {
                               std::this_thread::yield();
                           }
                       }};
    ContentId current;
    {
        ContinuationCache cache(config);
        const auto root = cache.publish_session(image(92), "generation", std::nullopt);
        block.store(true, std::memory_order_release);
        SessionPublishResult stale;
        std::thread publication([&] {
            stale = cache.publish_session(image(93, root.id), "generation", root.id);
        });
        while (!entered.load(std::memory_order_acquire)) std::this_thread::yield();
        check(cache.rollback_session("generation", 1),
              "rollback proceeds while an older alias write is blocked");
        const auto newer = cache.publish_session(image(94, root.id), "generation", root.id);
        current = newer.id;
        check(newer.alias_advanced,
              "new publication completes independently of stale alias I/O");
        release.store(true, std::memory_order_release);
        publication.join();
        check(stale.alias_advanced && newer.alias_advanced &&
                  cache.session_current("generation") == current,
              "in-memory alias remains at the post-rollback publication");
    }
    ContinuationCache reopened(config);
    check(reopened.session_current("generation") == current,
          "stale alias completion rewrites the current generation before restart");
}

void test_mutable_alias_replace_failure_is_reported_and_retried() {
    TempDir dir;
    std::atomic<unsigned> alias_replacements = 0;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 4096,
                       .l3_byte_budget = 1 << 20,
                       .persist_interval = std::chrono::seconds(0),
                       .persist_min_tokens = 0,
                       .before_mutable_replace = [&](const std::filesystem::path& path) {
                            if (path.extension() == ".alias" && alias_replacements++ == 1) {
                               throw std::runtime_error("injected alias replacement failure");
                           }
                       }};
    ContentId latest;
    {
        ContinuationCache cache(config);
        const auto first = cache.publish_session(image(95), "replace", std::nullopt);
        auto next_image = image(96, first.id);
        const auto next = cache.publish_session_l2(next_image, "replace", first.id);
        latest = next.id;
        check(next.alias_advanced && cache.queue_persistence("replace", latest,
                                                             next_image.frontier_tokens),
              "replacement candidate is queued");
        for (int i = 0; i != 400 && cache.stats().persistence_failures == 0; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        check(cache.stats().persistence_failures == 1,
              "existing old alias is not accepted when mutable replacement fails");
    }
    ContinuationCache reopened(config);
    check(reopened.session_current("replace") == latest,
          "failed mutable alias replacement is retried to the current state");
}

void test_immutable_alias_restart_and_classification_isolation() {
    TempDir dir;
    CacheConfig config{
        .enable_l3 = true, .root = dir.path, .l2_byte_budget = 0, .l3_byte_budget = 1 << 20};
    const std::string alias = std::string(kStableAliasPrefix) + std::string(64, 'a');
    const auto first_image  = image(40);
    const auto second_image = image(41);
    const ContentId first   = content_id(first_image);
    {
        ContinuationCache cache(config);
        check(cache.store(first_image) == first, "stable image is stored before alias publication");
        check(cache.publish_immutable_alias(alias, first), "stable alias publishes once");
        const ContentId second = cache.store(second_image, StoreOptions{.session = alias});
        check(second != first && cache.session_current(alias) == first,
              "session classification cannot replace a reserved stable alias");
        check(!cache.publish_immutable_alias(alias, second),
              "stable alias cannot be rebound to another image");
        check(!cache.rollback_session(alias, 0), "stable alias cannot enter rollback machinery");
        check(cache.session_history(alias) == std::vector<ContentId>({first}),
              "stable alias has no history");
    }
    ContinuationCache reopened(config);
    check(reopened.session_current(alias) == first && reopened.get(first) == first_image,
          "immutable stable alias survives L3 restart");
}

void test_corrupt_and_torn_aliases_are_ignored() {
    TempDir corrupt_dir;
    CacheConfig config{.enable_l3      = true,
                       .root           = corrupt_dir.path,
                       .l2_byte_budget = 0,
                       .l3_byte_budget = 1 << 20};
    StoreOptions options;
    options.session = "corrupt";
    {
        ContinuationCache cache(config);
        (void)cache.store(image(33), options);
    }
    overwrite_short(only_file(corrupt_dir.path / "aliases"));
    std::ofstream(corrupt_dir.path / "aliases" / "publish.tmp", std::ios::binary).put('x');
    std::ofstream(corrupt_dir.path / "tmp" / "alias.tmp", std::ios::binary).put('x');
    bool threw = false;
    try {
        ContinuationCache reopened(config);
        check(!reopened.session_current(*options.session), "corrupt alias is ignored at startup");
        check(reopened.stats().entries == 1, "corrupt alias does not damage immutable entries");
    } catch (...) { threw = true; }
    check(!threw, "corrupt and torn aliases never fail cache startup");
}

void test_eviction_and_stale_alias_pruning() {
    TempDir dir;
    auto now = std::chrono::system_clock::time_point{} + std::chrono::hours(1);
    CacheConfig config{.enable_l3      = true,
                       .root           = dir.path,
                       .l2_byte_budget = 0,
                       .l3_byte_budget = 1 << 20,
                       .l3_idle_ttl    = std::chrono::seconds(10),
                       .now            = [&] { return now; }};
    StoreOptions options;
    options.session = "stale";
    ContentId expired;
    {
        ContinuationCache cache(config);
        expired = cache.store(image(34), options);
    }
    now += std::chrono::seconds(11);
    {
        ContinuationCache reopened(config);
        check(!reopened.session_current(*options.session), "expired alias ID is pruned on restart");
        check(!reopened.get(expired), "expired aliased image remains a safe miss");
    }
    check(only_file(dir.path / "aliases").empty(), "empty stale alias is removed durably");

    TempDir missing_dir;
    config.root        = missing_dir.path;
    config.l3_idle_ttl = std::chrono::hours(1);
    now -= std::chrono::seconds(11);
    ContentId missing;
    {
        ContinuationCache cache(config);
        missing = cache.store(image(36), options);
    }
    std::filesystem::remove(missing_dir.path / "manifests" / (missing.hex + ".manifest"));
    {
        ContinuationCache reopened(config);
        check(!reopened.session_current(*options.session), "missing alias ID is pruned on restart");
    }
    check(only_file(missing_dir.path / "aliases").empty(), "missing alias is removed durably");

    TempDir evict_dir;
    config.root        = evict_dir.path;
    config.l3_idle_ttl = std::chrono::hours(1);
    now -= std::chrono::seconds(11);
    ContentId evicted;
    {
        ContinuationCache cache(config);
        evicted = cache.store(image(35), options);
        check(cache.session_current(*options.session) == evicted,
              "entry has an alias before startup eviction");
    }
    config.l3_byte_budget = 1;
    ContinuationCache constrained(config);
    check(!constrained.get(evicted), "entry is evicted when the reopened L3 budget shrinks");
    check(!constrained.session_current(*options.session) &&
              only_file(evict_dir.path / "aliases").empty(),
          "eviction removes the in-memory and durable alias");
}

void test_l2_only_never_touches_filesystem() {
    TempDir parent;
    const auto unused_root = parent.path / "must-not-exist";
    CacheConfig config{.root = unused_root, .l2_byte_budget = 4096};
    ContinuationCache cache(config);
    const auto expected = image(20);
    const auto id       = cache.store(expected);
    check(cache.get(id) == expected, "L2-only image roundtrips without L3");
    check(cache.stats().entries == 1 && cache.stats().l3_bytes == 0,
          "L2-only entry has no durable accounting");
    check(!std::filesystem::exists(unused_root), "disabled L3 does not create its configured root");

    bool rejected = false;
    try {
        ContinuationCache invalid(CacheConfig{.enable_l3 = true});
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "enabled L3 requires an explicit root");
}

void test_zero_budgets_disable_storage() {
    TempDir parent;
    const auto unused_root = parent.path / "zero-budget";
    CacheConfig config{
        .enable_l3 = true, .root = unused_root, .l2_byte_budget = 0, .l3_byte_budget = 0};
    ContinuationCache cache(config);
    const auto id = cache.store(image(21));
    check(!cache.get(id) && cache.stats().entries == 0, "zero budgets produce a safe miss");
    check(!std::filesystem::exists(unused_root), "zero L3 budget does not create an L3 root");
}

void test_filesystem_reserve_falls_back_to_l2() {
    TempDir dir;
    CacheConfig config{.enable_l3                = true,
                       .root                     = dir.path,
                       .l2_byte_budget           = 4096,
                       .l3_byte_budget           = 1 << 20,
                       .filesystem_reserve_bytes = std::numeric_limits<std::size_t>::max()};
    ContinuationCache cache(config);
    const auto expected = image(22);
    const auto id       = cache.store(expected);
    check(cache.get(id) == expected, "filesystem reserve rejection retains the L2 image");
    check(cache.stats().l3_bytes == 0 && only_file(dir.path / "manifests").empty(),
          "filesystem reserve prevents durable publication");
}

void test_concurrent_corruption_is_a_safe_miss() {
    TempDir dir;
    CacheConfig config{.enable_l3      = true,
                       .root           = dir.path,
                       .l2_byte_budget = 0,
                       .l3_byte_budget = 1 << 20,
                       .chunk_bytes    = 4096};
    ContinuationCache cache(config);
    const auto id = cache.store(image(23));
    overwrite_short(dir.path / "chunks" / id.hex);
    std::atomic<int> hits       = 0;
    std::atomic<int> exceptions = 0;
    std::vector<std::thread> readers;
    for (int i = 0; i != 8; ++i) {
        readers.emplace_back([&] {
            try {
                if (cache.get(id)) ++hits;
            } catch (...) { ++exceptions; }
        });
    }
    for (auto& reader : readers) reader.join();
    check(hits == 0 && exceptions == 0, "concurrent corrupt reads are thread-safe misses");
    check(cache.stats().entries == 0, "concurrent miss removes unusable catalog state once");
}

void test_lookup_diagnostics_classify_tier_and_failure() {
    ContinuationCache l2(CacheConfig{.l2_byte_budget = 4096});
    const auto l2_id = l2.admit(image(24));
    const auto l2_hit = l2.lookup_shared(l2_id);
    check(l2_hit.image && l2_hit.source == CacheSource::L2 &&
              l2_hit.status == CacheLookupStatus::Hit,
          "lookup reports an authoritative L2 hit");
    check(l2.lookup_shared(ContentId{std::string(64, 'f')}).status == CacheLookupStatus::Absent,
          "unknown content is an absent lookup");

    TempDir dir;
    CacheConfig config{.enable_l3 = true,
                       .root = dir.path,
                       .l2_byte_budget = 0,
                       .l3_byte_budget = 1 << 20,
                       .chunk_bytes = 4096,
                       .before_l3_restore_io = [] {
                           std::this_thread::sleep_for(std::chrono::milliseconds(2));
                       }};
    ContinuationCache l3(config);
    const auto l3_id = l3.store(image(25));
    const auto l3_hit = l3.lookup_shared(l3_id);
    check(l3_hit.image && l3_hit.source == CacheSource::L3 &&
              l3_hit.status == CacheLookupStatus::Hit,
          "lookup reports an authoritative L3 restore");

    const auto corrupt_id = l3.store(image(26));
    overwrite_short(dir.path / "chunks" / corrupt_id.hex);
    const auto corrupt = l3.lookup_shared(corrupt_id);
    check(!corrupt.image && corrupt.source == CacheSource::L3 &&
              corrupt.status == CacheLookupStatus::UnavailableOrCorrupt,
          "known corrupt L3 content differs from an absent ID");
    check(l3_hit.io_microseconds >= 1000, "L3 lookup reports measured I/O latency");
}

} // namespace

int main() {
    test_roundtrip_dedup_restart_and_permissions();
    test_image_size_matches_canonical_serialization();
    test_previous_development_manifest_is_ignored();
    test_l2_eviction();
    test_shared_hits_pin_one_charged_image();
    test_persistence_policy_helpers();
    test_l2_cost_aware_admission_retains_value();
    test_l2_admission_promotion_and_restart();
    test_l2_session_publication_is_decoupled_from_filesystem();
    test_l2_access_does_not_wait_for_persistence_io();
    test_async_coalescing_shutdown_and_restart();
    test_async_stable_alias_first_writer_and_restart();
    test_pin_protects_from_l3_eviction();
    test_l3_expiry_defers_cleanup_until_restore_and_explicit_pins_release();
    test_independent_idle_ttls_and_refresh();
    test_corrupt_truncated_and_remnants();
    test_oversized_manifest_chunk_and_orphan_cleanup();
    test_sessions_history_and_rollback();
    test_session_compare_and_swap_branches_and_rollback();
    test_session_compare_and_swap_restart_persistence();
    test_exact_id_rollback_generation_and_shared_candidates();
    test_corrupt_historical_candidate_keeps_other_shared_leases();
    test_session_alias_restart_and_rollback_restart();
    test_stale_alias_write_cannot_resurrect_rolled_back_state();
    test_mutable_alias_replace_failure_is_reported_and_retried();
    test_immutable_alias_restart_and_classification_isolation();
    test_corrupt_and_torn_aliases_are_ignored();
    test_eviction_and_stale_alias_pruning();
    test_l2_only_never_touches_filesystem();
    test_zero_budgets_disable_storage();
    test_filesystem_reserve_falls_back_to_l2();
    test_concurrent_corruption_is_a_safe_miss();
    test_lookup_diagnostics_classify_tier_and_failure();
    return failures == 0 ? 0 : 1;
}
