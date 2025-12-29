FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /lstm-from-scratch

# Copy entire repository
COPY . .

# Create build directory and build the project
WORKDIR /lstm-from-scratch/cpp/build
RUN cmake .. && make

# Set the working directory back to /lstm-from-scratch
WORKDIR /lstm-from-scratch

# Default command to run the LSTM executable
CMD ["./build/LSTM"]
