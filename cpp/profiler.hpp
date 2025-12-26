#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

class Profiler {
  public:
    static Profiler &instance() {
        static Profiler profiler;
        return profiler;
    }

    void start(const std::string &name) {
        timers[name] = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string &name) {
        auto end = std::chrono::high_resolution_clock::now();
        auto start = timers[name];
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        if (totals.find(name) == totals.end()) {
            totals[name] = 0.0;
            counts[name] = 0;
        }

        totals[name] += duration.count();
        counts[name]++;
    }

    void print_report() const {
        std::cout << "\n============== Profiling Report ==============" << std::endl;
        std::cout << "Operation                          | Total (ms) |  Avg (us)  | Count |   %"
                  << std::endl;
        std::cout
            << "------------------------------------------------------------------------------"
            << std::endl;

        // Calculate total time
        double grand_total = 0.0;
        for (const auto &[name, total] : totals) {
            grand_total += total;
        }

        // Create sorted vector of entries by percentage (descending)
        std::vector<std::tuple<std::string, double, int, double>> entries;
        for (const auto &[name, total] : totals) {
            double percentage = (total / grand_total) * 100.0;
            entries.push_back({name, total, counts.at(name), percentage});
        }

        // Sort by percentage (descending)
        std::sort(entries.begin(), entries.end(),
                  [](const auto &a, const auto &b) { return std::get<3>(a) > std::get<3>(b); });

        // Print sorted entries
        for (const auto &[name, total, count, percentage] : entries) {
            double total_ms = total / 1000.0;
            double avg_us = total / count;

            printf("%-34s | %010.2f | %010.2f | %05d | %04.1f%%\n", name.c_str(), total_ms, avg_us,
                   count, percentage);
        }

        std::cout
            << "------------------------------------------------------------------------------"
            << std::endl;
        std::cout << "Total time: " << (grand_total / 1000.0) << " ms" << std::endl;
        std::cout << "==============================================\n" << std::endl;
    }

    void reset() {
        timers.clear();
        totals.clear();
        counts.clear();
    }

  private:
    Profiler() = default;

    std::map<std::string, std::chrono::high_resolution_clock::time_point> timers;
    std::map<std::string, double> totals;
    std::map<std::string, int> counts;
};

// RAII timer for automatic start/stop
class ScopedTimer {
  public:
    ScopedTimer(const std::string &name) : name(name) {
        Profiler::instance().start(name);
    }

    ~ScopedTimer() {
        Profiler::instance().stop(name);
    }

  private:
    std::string name;
};

// Macros for easy profiling
#define PROFILE_SCOPE(name) ScopedTimer _timer_##__LINE__(name)
#define PROFILE_START(name) Profiler::instance().start(name)
#define PROFILE_STOP(name) Profiler::instance().stop(name)
#define PROFILE_REPORT() Profiler::instance().print_report()
#define PROFILE_RESET() Profiler::instance().reset()
