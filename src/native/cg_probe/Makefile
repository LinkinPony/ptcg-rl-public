CXX ?= g++
ENGINE_DIR ?= ../../../data/ptcg_engine/ptcgProgram 22
TARGET ?= libcg_probe.so

CPPFLAGS ?=
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -fPIC -fvisibility=hidden -Wall -Wextra
LDFLAGS ?= -shared
DEPFLAGS ?= -MMD -MP

.PHONY: all clean

all: $(TARGET)

SOURCES := probe_export.cpp probe_features.cpp planner_export.cpp planner_session_export.cpp \
	planner_session_internal.cpp request_fingerprint.cpp sha256.cpp \
	state_token_validation.cpp
OBJECTS := $(SOURCES:.cpp=.o)
DEPENDENCIES := $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) Makefile
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

%.o: %.cpp Makefile
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -I"$(ENGINE_DIR)" -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS) $(DEPENDENCIES)

-include $(DEPENDENCIES)
