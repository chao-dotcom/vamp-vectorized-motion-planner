# Use Ubuntu with GCC for C++ compilation
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials and required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source files
COPY src/ /app/src/

# Compile Planner_v1
RUN g++ -std=c++17 -O3 -mavx2 -march=native -o planner_v1 src/Planner_v1.cpp

# Compile Planner_v2
RUN g++ -std=c++17 -O3 -mavx2 -march=native -o planner_v2 src/Planner_v2.cpp

# Compile Planner_v3
RUN g++ -std=c++17 -O3 -mavx2 -march=native -o planner_v3 src/Planner_v3.cpp

# Compile Planner_v4
RUN g++ -std=c++17 -O3 -mavx2 -march=native -o planner_v4 src/Planner_v4.cpp

# Compile Planner_v5
RUN g++ -std=c++17 -O3 -mavx2 -march=native -o planner_v5 src/Planner_v5.cpp

# Default command runs Planner_v1
CMD ["./planner_v1"]

