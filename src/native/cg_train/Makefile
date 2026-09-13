CXX ?= g++
ENGINE_DIR ?= ../../../data/ptcg_engine/ptcgProgram 22
TARGET ?= libcg_train.so
PARITY_TARGET ?= ../../../tmp/cg_train_parity_probe
LOG_PARITY_TARGET ?= ../../../tmp/cg_train_log_parity_probe
PARALLEL_TEST_TARGET ?= ../../../tmp/cg_train_parallel_executor_test

CPPFLAGS ?=
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -fPIC -fvisibility=hidden -pthread \
	-fno-gnu-unique -Wall -Wextra
LDFLAGS ?= -shared
DEPFLAGS ?= -MMD -MP

.PHONY: all clean log-parity-probe parallel-executor-test parity-probe

all: $(TARGET)

parity-probe: $(PARITY_TARGET)

$(PARITY_TARGET): parity_probe.cpp Makefile
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I"$(ENGINE_DIR)" $< -o "$@"

log-parity-probe: $(LOG_PARITY_TARGET)

$(LOG_PARITY_TARGET): log_parity_probe.cpp public_log.h Makefile
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I"$(ENGINE_DIR)" $< -o "$@"

parallel-executor-test: $(PARALLEL_TEST_TARGET)

$(PARALLEL_TEST_TARGET): parallel_executor_test.cpp parallel_executor.cpp \
		parallel_executor.h Makefile
	mkdir -p "$(@D)"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) parallel_executor_test.cpp \
		parallel_executor.cpp -o "$@"

SOURCES := cg_train.cpp parallel_executor.cpp quota_scheduler.cpp \
	rollout_export.cpp rollout_model.cpp rollout_tracker.cpp
OBJECTS := $(SOURCES:.cpp=.o)
DEPENDENCIES := $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) Makefile
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

%.o: %.cpp cg_train.h cg_rollout_encoder.h parallel_executor.h \
	quota_scheduler.h rollout_internal.h public_log.h Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -I"$(ENGINE_DIR)" -c $< -o $@

clean:
	rm -f "$(TARGET)" $(OBJECTS) $(DEPENDENCIES) \
		"$(PARITY_TARGET)" "$(LOG_PARITY_TARGET)" \
		"$(PARALLEL_TEST_TARGET)"

-include $(DEPENDENCIES)
