// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and character names are trademarks of Nintendo.
// SPDX-License-Identifier: LicenseRef-PTCG-ABC-Competition-Use-Only
// Part of the Pokémon TCG AI Battle Challenge. Provided for Competition use only.

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "All.h"
#include "state_token_validation.h"

namespace {

constexpr int kFuzzIterations = 2048;
constexpr std::size_t kMaximumDecodedBytes = 1U << 25;

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

State SyntheticRoot() {
  State state = {};
  state.clear();
  state.players[0].playerIndex = 0;
  state.players[1].playerIndex = 1;
  state.selectType = SelectType::Main;
  state.selectContext = SelectContext::Main;
  state.selectPlayer = 0;
  state.selectMin = 1;
  state.selectMax = 1;

  SelectOption option = {};
  option.type = SelectOptionType::End;
  state.options.push_back(option);

  Log log(LogType::Shuffle);
  log.param.push_back(0);
  state.logs.push_back(log);
  return state;
}

std::string EncodeState(const State& state) {
  BinaryWriter writer;
  state.serialize(writer);
  writer.toBase64();
  return std::string(writer.base64.begin(), writer.base64.end());
}

bool DecodeAndValidate(std::string_view token) {
  std::vector<std::uint8_t> decoded;
  std::string error;
  if (!DecodeValidatedStateToken(
          token.data(), static_cast<int>(token.size()), &decoded, &error)) {
    return false;
  }
  Check(
      decoded.size() <= kMaximumDecodedBytes,
      "decoder exceeded its documented output bound");

  BinaryReader reader;
  reader.buf = std::move(decoded);
  State state = {};
  state.clear();
  state.deserialize(reader);
  return ValidateDeserializedPlannerRoot(state, 0);
}

template <typename FixedList>
void CorruptFixedListCount(FixedList* values) {
  const int invalid_count = 1 << 20;
  const auto* source = reinterpret_cast<const unsigned char*>(&invalid_count);
  auto* destination = reinterpret_cast<unsigned char*>(values);
  std::copy(source, source + sizeof(invalid_count), destination);
}

std::string CorruptPlayerListToken() {
  State state = SyntheticRoot();
  CorruptFixedListCount(&state.players[0].hand);
  return EncodeState(state);
}

std::string CorruptCardListToken() {
  State state = SyntheticRoot();
  CorruptFixedListCount(&state.allCard[0].abilityUsed);
  return EncodeState(state);
}

std::string CorruptLogListToken() {
  State state = SyntheticRoot();
  CorruptFixedListCount(&state.logs[0].param);
  return EncodeState(state);
}

std::string CorruptVectorCardRefToken() {
  State state = SyntheticRoot();
  state.turnPlay.push_back(CardRef(255));
  return EncodeState(state);
}

std::string CorruptFunctionIndexToken() {
  State state = SyntheticRoot();
  GameFunction function = {};
  function.functionIndex = std::numeric_limits<int>::max();
  function.argType = ArgType::None;
  function.callCount = 1;
  function.calledCount = 0;
  state.functionStack.push_back(function);
  return EncodeState(state);
}

void ExerciseMalformedCorpus() {
  const std::vector<std::string> malformed = {
      "", "A", "-", "*", "!!!!", "====", "AAAA", "A===", "*///"};
  for (const std::string& token : malformed) {
    static_cast<void>(DecodeAndValidate(token));
  }

  // Each group expands to 262143 zero sextets. The bounded decoder must reject
  // this compressed bomb before allocating beyond its expansion cap.
  std::string compressed_bomb;
  for (int index = 0; index < 300; ++index) {
    compressed_bomb += "*///";
  }
  static_cast<void>(DecodeAndValidate(compressed_bomb));
}

void ExerciseDeterministicMutations(const std::string& valid_token) {
  std::mt19937 random(0x5a17c0deU);
  std::uniform_int_distribution<int> byte(0, 127);
  for (int iteration = 0; iteration < kFuzzIterations; ++iteration) {
    std::string token = valid_token;
    switch (iteration % 4) {
      case 0: {
        const std::size_t size =
            static_cast<std::size_t>(random() % (token.size() + 1));
        token.resize(size);
        break;
      }
      case 1: {
        const std::size_t offset =
            static_cast<std::size_t>(random() % token.size());
        token[offset] = static_cast<char>(byte(random));
        break;
      }
      case 2:
        token.append(static_cast<std::size_t>(random() % 8 + 1), '*');
        break;
      case 3:
        token.resize(static_cast<std::size_t>(random() % 128 + 1));
        std::generate(token.begin(), token.end(), [&]() {
          return static_cast<char>(byte(random));
        });
        break;
    }
    static_cast<void>(DecodeAndValidate(token));
  }
}

}  // namespace

int main() {
  const std::string valid_token = EncodeState(SyntheticRoot());
  Check(DecodeAndValidate(valid_token), "synthetic root token was rejected");
  Check(
      !DecodeAndValidate(CorruptPlayerListToken()),
      "corrupt PlayerState FixedList count was accepted");
  Check(
      !DecodeAndValidate(CorruptCardListToken()),
      "corrupt Card::abilityUsed count was accepted");
  Check(
      !DecodeAndValidate(CorruptLogListToken()),
      "corrupt Log::param count was accepted");
  Check(
      !DecodeAndValidate(CorruptVectorCardRefToken()),
      "out-of-range vector CardRef was accepted");
  Check(
      !DecodeAndValidate(CorruptFunctionIndexToken()),
      "out-of-range GameFunction index was accepted");
  ExerciseMalformedCorpus();
  ExerciseDeterministicMutations(valid_token);
  std::cout << "native state-token safety harness passed\n";
  return 0;
}
