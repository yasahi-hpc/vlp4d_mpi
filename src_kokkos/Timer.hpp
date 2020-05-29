#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>
#include <vector>
#include <iostream>

struct Timer {
private:
  std::string label_;
  double accumulated_time_;
  int calls_;
  std::chrono::high_resolution_clock::time_point begin_, end_;

public:
  Timer() : accumulated_time_(0.0), calls_(0), label_(""){};
  Timer(const std::string label) : accumulated_time_(0.0), calls_(0), label_(label){};
  virtual ~Timer(){};

  void begin() {
    begin_ = std::chrono::high_resolution_clock::now();
  }

  void end() {
    end_ = std::chrono::high_resolution_clock::now();
    accumulated_time_ += std::chrono::duration_cast<std::chrono::duration<double> >(end_ - begin_).count();
    calls_++;
  }

  double seconds(){return accumulated_time_;};
  double milliseconds(){return accumulated_time_*1.e3;};
  int calls(){return calls_;};
  std::string label(){return label_;};
  void reset(){accumulated_time_ = 0.; calls_ = 0;};
};

enum TimerEnum : int {Total,
                      Halo_fill_A,
                      Halo_fill_B,
                      Halo_comm,
                      Advec2D,
                      Advec4D,
                      Field,
                      Diag,
                      Splinecoeff_xy,
                      Splinecoeff_vxvy,
                      Nb_timers};

static void defineTimers(std::vector<Timer*> &timers) {
  // Set timers
  timers.resize(Nb_timers);
  timers[Total]            = new Timer("total");
  timers[Halo_fill_A]      = new Timer("halo_fill_A");
  timers[Halo_fill_B]      = new Timer("halo_fill_B");
  timers[Halo_comm]        = new Timer("halo_comm");
  timers[Advec2D]          = new Timer("advec2D");
  timers[Advec4D]          = new Timer("advec4D");
  timers[Field]            = new Timer("field");
  timers[Diag]             = new Timer("diag");
  timers[Splinecoeff_xy]   = new Timer("splinecoeff_xy");
  timers[Splinecoeff_vxvy] = new Timer("splinecoeff_vxvy");
  timers[Nb_timers]        = new Timer("nb_timers");
}

static void printTimers(std::vector<Timer*> &timers) {
  // Print timer information
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    std::cout << (*it)->label() << " " << (*it)->seconds() << " [s], " << (*it)->calls() << " calls" << std::endl;
  }
}
        
static void freeTimers(std::vector<Timer*> &timers) {
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    delete *it;
  }
};

#endif
