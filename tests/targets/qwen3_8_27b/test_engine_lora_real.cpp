// Runtime LoRA against the real 27B artifact.
//
// The gate that matters here is that prefill applies the selected adapter. A regression that
// bound the adapter only for the batched decode paths still produced adapted-looking output
// after the first token, so an "output differs from base" check over a long completion does not
// catch it. Asking for exactly one greedy token does: that token is the argmax of the logits the
// prefill chunk produced, so it can only move if prefill ran with the adapter.
#include "ninfer/engine.h"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

// Every adapter in one bank shares a rank and a site inventory, so the two fixtures must come
// from the same converter inventory. The trained arm carries a behavioural signature; a synthetic
// random delta is unstructured noise that does not reliably move a confident argmax, and its
// numerics are already covered by tests/ops/test_lora_delta_add.cpp.
constexpr const char* kZeroName    = "zero";
constexpr const char* kTrainedName = "trained";

// A real chat turn, not a synthetic token run. The first assistant token has to be genuinely
// contested for an argmax comparison to have any sensitivity, and the public API exposes no
// logprobs to compare instead.
ninfer::PromptInput chat_prompt() {
    ninfer::ChatMessage message;
    message.role = "user";
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text,
        .text = "Let a and b be real numbers with a + b = 5 and a * b = 3. Find a^5 - b^5.",
        .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

ninfer::RequestOptions greedy(std::uint32_t output_tokens, std::optional<std::string> adapter) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = output_tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.adapter                 = std::move(adapter);
    // The arms share one prompt. Reuse would let a later arm restore a prefix another arm
    // prefilled instead of prefilling under its own adapter.
    options.execution.allow_prefix_reuse = false;
    options.stop.include_model_defaults  = false;
    return options;
}

std::vector<ninfer::TokenId> run(ninfer::Engine& engine, std::uint32_t output_tokens,
                                 std::optional<std::string> adapter) {
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(chat_prompt()), greedy(output_tokens, adapter));
    return result.generated_token_ids;
}

int verify_registration(const ninfer::Engine& engine) {
    const std::vector<std::string>& names = engine.load_summary().lora_adapter_names;
    if (names.size() != 2 || names[0] != kZeroName || names[1] != kTrainedName) {
        std::cerr << "registered adapter names are wrong: " << names.size() << " entries\n";
        return 1;
    }
    return 0;
}

// The regression gate: one greedy token, produced entirely by prefill.
int verify_prefill_applies_adapter(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> base    = run(engine, 1, std::nullopt);
    const std::vector<ninfer::TokenId> adapted = run(engine, 1, kTrainedName);
    if (base.size() != 1 || adapted.size() != 1) {
        std::cerr << "single-token requests did not generate exactly one token\n";
        return 1;
    }
    if (base[0] == adapted[0]) {
        std::cerr << "prefill ignored the selected LoRA adapter: the first greedy token is "
                  << base[0] << " both with and without the adapter\n";
        return 1;
    }
    return 0;
}

// A zero-delta adapter must leave greedy decoding byte-identical to the base weights, and must
// not leave the bank selected for the request that follows it.
int verify_zero_adapter_is_neutral(ninfer::Engine& engine) {
    constexpr std::uint32_t kTokens = 8;
    const std::vector<ninfer::TokenId> base = run(engine, kTokens, std::nullopt);
    const std::vector<ninfer::TokenId> zero = run(engine, kTokens, kZeroName);
    const std::vector<ninfer::TokenId> after = run(engine, kTokens, std::nullopt);
    if (base.size() != kTokens || zero.size() != kTokens || after.size() != kTokens) {
        std::cerr << "greedy requests did not generate the requested token count\n";
        return 1;
    }
    if (zero != base) {
        std::cerr << "a zero-delta adapter changed greedy output\n";
        return 1;
    }
    if (after != base) {
        std::cerr << "an adapter selection leaked into the following base request\n";
        return 1;
    }
    return 0;
}

int verify_unknown_adapter_is_rejected(ninfer::Engine& engine) {
    try {
        run(engine, 1, "not-registered");
    } catch (const ninfer::RequestError& error) {
        if (error.kind() != ninfer::RequestErrorKind::UnknownAdapter) {
            std::cerr << "unknown adapter raised the wrong request error kind\n";
            return 1;
        }
        return 0;
    }
    std::cerr << "an unregistered adapter name was accepted\n";
    return 1;
}

int exercise(const char* artifact, const char* zero_path, const char* trained_path) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 2048;
    options.kv_capacity   = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk = 1024;
    options.lora_adapters = {ninfer::LoraAdapterSpec{.name = kZeroName, .path = zero_path},
                             ninfer::LoraAdapterSpec{.name = kTrainedName, .path = trained_path}};

    ninfer::Engine engine(options);
    if (const int result = verify_registration(engine); result != 0) { return result; }
    if (const int result = verify_prefill_applies_adapter(engine); result != 0) { return result; }
    if (const int result = verify_zero_adapter_is_neutral(engine); result != 0) { return result; }
    if (const int result = verify_unknown_adapter_is_rejected(engine); result != 0) {
        return result;
    }
    return 0;
}

const char* env_or_null(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr && *value != '\0') ? value : nullptr;
}

} // namespace

int main() {
    const char* artifact = env_or_null("NINFER_QWEN3_8_27B_WEIGHTS");
    const char* zero     = env_or_null("NINFER_QWEN3_8_27B_LORA_ZERO");
    const char* trained  = env_or_null("NINFER_QWEN3_8_27B_LORA_TRAINED");
    if (artifact == nullptr || zero == nullptr || trained == nullptr) {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS, NINFER_QWEN3_8_27B_LORA_ZERO and "
                     "NINFER_QWEN3_8_27B_LORA_TRAINED are all required\n";
        return 77;
    }
    if (const int result = exercise(artifact, zero, trained); result != 0) { return result; }
    std::cout << "ok\n";
    return 0;
}
