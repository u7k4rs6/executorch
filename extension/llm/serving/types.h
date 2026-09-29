/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {
namespace serving {

struct ET_EXPERIMENTAL TextSegment {
  std::string text;
};

struct ET_EXPERIMENTAL TokenIdsSegment {
  std::vector<batching::Token> ids;
};

// Extend this variant with typed segments when another modality is supported.
using PromptSegment ET_EXPERIMENTAL = std::variant<TextSegment, TokenIdsSegment>;

// The full prompt, not a session delta. Preparation encodes each text segment
// separately and appends ID segments verbatim, in order, without adding BOS/EOS.
// Segment boundaries therefore matter even between adjacent text segments.
struct ET_EXPERIMENTAL PromptInput {
  std::vector<PromptSegment> segments;
};

struct ET_EXPERIMENTAL GenerationOptions {
  // Unset uses the available context, or the service default if unknown.
  std::optional<std::int32_t> max_new_tokens;
  batching::SamplingParams sampling;
  std::vector<batching::Token> stop_tokens;
  std::vector<std::string> stop_strings;
  std::optional<std::uint64_t> seed;
};

enum class ET_EXPERIMENTAL FinishReason {
  Stop, // EOS, a stop token, or a stop string
  Length,
  Cancelled,
  Failed,
};

enum class ET_EXPERIMENTAL ErrorCode {
  InvalidArgument,
  NotReady,
  SessionNotFound,
  SessionBusy,
  CapacityExceeded,
  Internal,
};

struct ET_EXPERIMENTAL ServingError {
  ErrorCode code = ErrorCode::Internal;
  std::string message;
};

struct ET_EXPERIMENTAL GenerationStats {
  std::size_t prompt_tokens = 0;
  // Non-terminal generated tokens, including tokens hidden by a string stop.
  // Neither this count nor visible text describes the session's logical history.
  std::size_t completion_tokens = 0;
  std::size_t reused_prompt_tokens = 0;
  // Actually consumed prompt tokens; may be partial on cancellation or failure.
  std::size_t prefilled_prompt_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double total_ms = 0.0;
  std::string session_reset_reason;
  // Absent when output trimming or cancellation makes exact replay unsafe.
  std::optional<std::vector<batching::Token>> generated_token_ids;
};

struct ET_EXPERIMENTAL TextEvent {
  std::string text;
};

struct ET_EXPERIMENTAL TerminalEvent {
  FinishReason finish_reason = FinishReason::Stop;
  GenerationStats stats;
  // Present exactly when finish_reason is Failed.
  std::optional<ServingError> error;
};

// Per-request order: zero or more text events, then exactly one terminal event.
using GenerationEvent ET_EXPERIMENTAL = std::variant<TextEvent, TerminalEvent>;

struct ET_EXPERIMENTAL ServingInfo {
  bool ready = false;
  std::size_t max_context_length = 0;
  std::size_t max_sessions = 0;
  // Logical serving slots, including slots reserved during reset/reopen.
  // This is not a count of physically live executor sessions.
  std::size_t active_sessions = 0;
};

} // namespace serving
} // namespace llm
} // namespace extension
} // namespace executorch
