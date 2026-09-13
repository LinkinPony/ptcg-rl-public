// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®,
// and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only

#ifndef SRC_NATIVE_CG_TRAIN_ROLLOUT_INTERNAL_H_
#define SRC_NATIVE_CG_TRAIN_ROLLOUT_INTERNAL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cg_rollout_encoder.h"

namespace cg_train_rollout {

inline constexpr std::uint32_t kRolloutMagic = UINT32_C(0x43475245);
inline constexpr std::uint32_t kRolloutAbiVersion = 1;
inline constexpr std::size_t kDeckSize = 60;
inline constexpr std::size_t kTokenScalarSize = 59;
inline constexpr std::size_t kOptionScalarSize = 9;
inline constexpr std::size_t kDynamicEffectSize = 33;
inline constexpr std::size_t kMaximumEntitySlots = 2;
inline constexpr std::size_t kBeliefScalarSize = 4;
inline constexpr std::size_t kHistorySize = 8;
inline constexpr std::size_t kDeckFlowSize = 14;
inline constexpr std::size_t kLogParamWidth = 7;
inline constexpr std::int32_t kReadyStatus = 1;
inline constexpr std::int32_t kFinishedStatus = 2;
inline constexpr std::int32_t kNoError = 0;
inline constexpr std::uint32_t kMissingRow = UINT32_MAX;

using CountRow = std::vector<std::pair<std::int32_t, std::int32_t>>;

struct PublicCard {
  std::int32_t owner = 0;
  std::int32_t area = 0;
  std::int32_t area_index = 0;
  std::int32_t card_id = 0;
  std::int32_t serial = 0;
  std::int32_t hp = 0;
  std::int32_t max_hp = 0;
  std::int32_t appear_this_turn = 0;
};

struct Attachment {
  std::uint32_t parent = 0;
  std::int32_t kind = 0;
  std::int32_t card_id = 0;
  std::int32_t serial = 0;
  std::int32_t energy_type = 0;
  std::int32_t energy_units = 0;
};

struct Option {
  std::int32_t type = 0;
  std::array<std::int32_t, 5> params{};
};

struct RawRow {
  bool initialized = false;
  std::int32_t status = 0;
  std::int32_t error = 0;
  std::int32_t perspective = 0;
  std::int32_t select_type = 0;
  std::int32_t select_context = 0;
  std::int32_t select_min = 0;
  std::int32_t select_max = 0;
  std::int32_t result = 0;
  std::int32_t turn = 0;
  std::int32_t turn_action_count = 0;
  std::int32_t first_player = 0;
  std::uint32_t turn_flags = 0;
  std::int32_t remain_damage_counter = 0;
  std::int32_t remain_energy_cost = 0;
  std::array<std::int32_t, 2> deck_counts{};
  std::array<std::int32_t, 2> hand_counts{};
  std::array<std::int32_t, 2> prize_counts{};
  std::array<std::int32_t, 2> bench_max{};
  std::array<std::uint32_t, 2> status_flags{};
  std::uint32_t context_card_row = kMissingRow;
  std::uint32_t effect_card_row = kMissingRow;
  std::vector<Option> options;
  std::vector<PublicCard> visible_cards;
  std::vector<Attachment> attachments;
};

struct PerspectiveHistory {
  bool initialized = false;
  std::int32_t perspective = 0;
  CountRow own_deck_counts;
  std::unordered_map<std::int32_t, std::int32_t> revealed_by_serial;
  std::map<std::int32_t, std::int32_t> revealed_no_serial_counts;
  std::map<std::int32_t, std::int32_t> revealed_counts;
  std::array<std::array<std::int32_t, 4>, 2> history{};
  std::array<std::int32_t, 2> draw_counts{};
  std::array<std::int32_t, 2> return_counts{};
  std::array<std::int32_t, 2> recent_draw_counts{};
  std::array<std::int32_t, 2> recent_return_counts{};
  std::array<std::int32_t, 2> recent_deck_deltas{};
  std::array<std::int32_t, 2> rebound_counts{};
  std::array<std::int32_t, 2> no_attack_turns{};
  std::array<std::int32_t, 2> current_draw_counts{};
  std::array<std::int32_t, 2> current_return_counts{};
  std::array<std::int32_t, 2> current_deck_deltas{};
  std::array<bool, 2> turn_open{};
  std::array<bool, 2> turn_attacked{};
  std::array<bool, 2> turn_rebounded{};
  std::array<std::int32_t, 2> observed_deck_counts{};
  std::array<bool, 2> has_observed_deck_counts{};
  std::map<std::int32_t, std::int32_t> last_attack_by_serial;
};

struct SlotState {
  std::array<PerspectiveHistory, 2> histories;
  RawRow current;
};

struct VisibleEvidence {
  std::map<std::int32_t, std::int32_t> own_counts;
  std::map<std::int32_t, std::int32_t> opponent_counts;
  std::unordered_map<std::int32_t, std::int32_t> opponent_by_serial;
};

struct Posterior {
  std::vector<float> expected_counts;
  // Python selects sparse belief rows from the float64 posterior before
  // converting values to float32. Keep that validity decision separately so
  // a positive subnormal double that rounds to 0.0F remains a valid entry.
  std::vector<std::uint8_t> expected_valid;
  float unknown_probability = 0.0F;
  float entropy = 0.0F;
  std::int32_t compatible_count = 0;
  std::int32_t known_total = 0;
};

struct PreparedRow {
  const RawRow* raw = nullptr;
  const PerspectiveHistory* history = nullptr;
  CountRow own_unseen;
  CountRow opponent_revealed;
  CountRow known;
  std::shared_ptr<const Posterior> posterior;
  std::uint32_t state_token_count = 0;
  std::uint32_t attachment_count = 0;
  std::uint32_t option_count = 0;
  std::uint32_t deck_count = 0;
};

struct PreparedPlan {
  std::vector<std::uint32_t> slots;
  std::vector<std::int32_t> perspectives;
  std::vector<PreparedRow> rows;
  std::vector<std::uint32_t> row_beliefs;
  std::vector<std::size_t> unique_belief_rows;
  CgTrainRolloutShape shape{};
};

struct CatalogData {
  std::uint32_t deck_count = 0;
  std::uint32_t card_vocab_size = 0;
  std::vector<std::int16_t> entry_counts;
  std::vector<double> exact_log_priors;
  std::array<double, 61 * 61> log_combinations{};
  std::array<double, 61> log_factorials{};
  std::vector<double> unknown_card_probabilities;
  std::vector<double> unknown_log_card_probabilities;
  double unknown_log_prior = 0.0;
  std::set<std::int32_t> supporter_card_ids;
  std::size_t posterior_cache_capacity = 0;
};

class RolloutEncoderCore {
 public:
  RolloutEncoderCore(std::uint32_t slot_capacity,
                     const CgTrainRolloutCatalog& catalog);

  RolloutEncoderCore(const RolloutEncoderCore&) = delete;
  RolloutEncoderCore& operator=(const RolloutEncoderCore&) = delete;

  void ConsumeReset(std::uint32_t batch_count, const std::uint32_t* slots,
                    const std::int32_t* decks,
                    const CgTrainOutput& source);
  void ConsumeStep(std::uint32_t batch_count, const std::uint32_t* slots,
                   const CgTrainOutput& source);
  void ClearSlots(std::span<const std::uint32_t> slots);

  CgTrainRolloutShape PlanRows(
      std::span<const std::uint32_t> slots,
      std::span<const std::int32_t> perspectives);
  void EncodeRows(std::span<const std::uint32_t> slots,
                  std::span<const std::int32_t> perspectives,
                  std::uint32_t row_offset,
                  std::uint32_t belief_row_offset,
                  CgTrainRolloutOutput* output);

  std::uint32_t PlanKnown(
      std::span<const std::uint32_t> slots,
      std::span<const std::int32_t> perspectives);
  void WriteKnown(std::span<const std::uint32_t> slots,
                  std::span<const std::int32_t> perspectives,
                  std::uint32_t value_capacity, std::uint32_t* offsets,
                  std::int32_t* card_ids, std::int32_t* counts);

 private:
  using PosteriorCacheEntry = std::pair<
      std::shared_ptr<const Posterior>,
      std::list<CountRow>::iterator>;

  void Consume(bool reset, std::uint32_t batch_count,
               const std::uint32_t* slots, const std::int32_t* decks,
               const CgTrainOutput& source);
  RawRow CopyRow(const CgTrainOutput& source, std::uint32_t row) const;
  VisibleEvidence InspectVisible(const RawRow& row) const;
  void ApplyVisible(const VisibleEvidence& visible,
                    PerspectiveHistory* history) const;
  void ApplyLogs(const CgTrainOutput& source, std::uint32_t row,
                 PerspectiveHistory* history) const;

  std::vector<PreparedRow> PrepareRows(
      std::span<const std::uint32_t> slots,
      std::span<const std::int32_t> perspectives);
  PreparedPlan PrepareModelPlan(
      std::span<const std::uint32_t> slots,
      std::span<const std::int32_t> perspectives);
  const PreparedPlan& ModelPlan(
      std::span<const std::uint32_t> slots,
      std::span<const std::int32_t> perspectives);
  void InvalidateModelPlan();
  PreparedRow PrepareRow(const SlotState& slot,
                         std::int32_t perspective);
  void ValidatePreparedRowForModel(const PreparedRow& prepared) const;
  std::shared_ptr<const Posterior> PosteriorFor(const CountRow& known);
  Posterior ComputePosterior(const CountRow& known) const;

  void ValidateSelection(std::span<const std::uint32_t> slots,
                         std::span<const std::int32_t> perspectives) const;

  std::uint32_t slot_capacity_;
  CatalogData catalog_;
  std::vector<SlotState> slots_;
  std::mutex posterior_mutex_;
  std::map<CountRow, PosteriorCacheEntry> posterior_cache_;
  std::list<CountRow> posterior_lru_;
  std::optional<PreparedPlan> prepared_plan_;
};

std::string& LastError();
int32_t Fail(std::int32_t code, std::string message);

}  // namespace cg_train_rollout

struct CgTrainRolloutEncoder {
  explicit CgTrainRolloutEncoder(
      std::unique_ptr<cg_train_rollout::RolloutEncoderCore> value)
      : core(std::move(value)) {}

  std::unique_ptr<cg_train_rollout::RolloutEncoderCore> core;
  std::mutex mutex;
};

#endif  // SRC_NATIVE_CG_TRAIN_ROLLOUT_INTERNAL_H_
