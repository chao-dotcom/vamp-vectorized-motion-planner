# Makefile for SIMD-Accelerated RRT-Connect Planner

CXX = g++
# For correctness testing (recommended first):
CXXFLAGS_CORRECT = -std=c++17 -O2 -mavx2 -march=native -Wall -Wextra -Wconversion -Wsign-conversion
# For performance benchmarking:
CXXFLAGS_PERF = -std=c++17 -O3 -mavx2 -march=native -Wall -Wextra -ffast-math -flto
# SLEEF flags for vectorized trig (add to any build):
SLEEF_FLAGS = -DUSE_SLEEF -I/usr/local/include -L/usr/local/lib -lsleef -Wl,-rpath,/usr/local/lib
# Default: use correctness flags
CXXFLAGS = $(CXXFLAGS_CORRECT)

SRC_DIR = src
BUILD_DIR = build

# Source files
PLANNER_V13_SRC = $(SRC_DIR)/Planner_v13.cpp

# Executables
PLANNER_V13 = $(BUILD_DIR)/planner_v13
PLANNER_V13_SLEEF = $(BUILD_DIR)/planner_v13_sleef
PLANNER_V13_PERF = $(BUILD_DIR)/planner_v13_perf
PLANNER_V13_PERF_SLEEF = $(BUILD_DIR)/planner_v13_perf_sleef

.PHONY: all clean v13 v13-sleef run-v13 run-v13-sleef v13-perf v13-perf-sleef run-v13-perf run-v13-perf-sleef help compare

all: $(PLANNER_V13)

help:
	@echo "Available targets:"
	@echo "  make v13               - Build with correctness flags (scalar trig)"
	@echo "  make v13-sleef         - Build with SLEEF vectorized trig (-O2)"
	@echo "  make v13-perf          - Build with performance flags (scalar trig, -O3)"
	@echo "  make v13-perf-sleef    - Build with SLEEF + performance flags (-O3)"
	@echo "  make run-v13           - Run scalar correctness build"
	@echo "  make run-v13-sleef     - Run SLEEF correctness build"
	@echo "  make run-v13-perf      - Run scalar performance build"
	@echo "  make run-v13-perf-sleef - Run SLEEF performance build"
	@echo "  make compare           - Compare scalar vs SLEEF performance"
	@echo "  make clean             - Remove build directory"

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PLANNER_V13): $(PLANNER_V13_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_CORRECT) -o $(PLANNER_V13) $(PLANNER_V13_SRC)

$(PLANNER_V13_SLEEF): $(PLANNER_V13_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_CORRECT) $(SLEEF_FLAGS) -o $(PLANNER_V13_SLEEF) $(PLANNER_V13_SRC)

$(PLANNER_V13_PERF): $(PLANNER_V13_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_PERF) -o $(PLANNER_V13_PERF) $(PLANNER_V13_SRC)

$(PLANNER_V13_PERF_SLEEF): $(PLANNER_V13_SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_PERF) $(SLEEF_FLAGS) -o $(PLANNER_V13_PERF_SLEEF) $(PLANNER_V13_SRC)

v13: $(PLANNER_V13)

v13-sleef: $(PLANNER_V13_SLEEF)

v13-perf: $(PLANNER_V13_PERF)

v13-perf-sleef: $(PLANNER_V13_PERF_SLEEF)

run-v13: $(PLANNER_V13)
	$(PLANNER_V13)

run-v13-sleef: $(PLANNER_V13_SLEEF)
	$(PLANNER_V13_SLEEF)

run-v13-perf: $(PLANNER_V13_PERF)
	$(PLANNER_V13_PERF)

run-v13-perf-sleef: $(PLANNER_V13_PERF_SLEEF)
	$(PLANNER_V13_PERF_SLEEF)

compare: $(PLANNER_V13) $(PLANNER_V13_SLEEF)
	@echo "=== SCALAR (no SLEEF) ==="
	@$(PLANNER_V13) 2>&1 | head -25
	@echo ""
	@echo "=== SLEEF (vectorized trig) ==="
	@$(PLANNER_V13_SLEEF) 2>&1 | head -25

clean:
	rm -rf $(BUILD_DIR)

