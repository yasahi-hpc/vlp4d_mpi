# Device can be v100
DEVICES = $(DEVICE)

# Nvidia P100 on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring p100_tsubame3,$(DEVICES)))
CXX = clang++
CXXFLAGS = -O3 -std=c++11 -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp=libomp --cuda-path=${CUDA_HOME} -Xopenmp-target -march=sm_60 -Rpass-analysis=vector -DENABLE_OPENMP_OFFLOAD
LDFLAGS = -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_60 -fopenmp=libomp -I/apps/t3/sles12sp2/cuda/9.2.148/include

ifneq (,$(findstring STEP1,$(OPTIMIZATION)))
  CXXFLAGS += -DLONG_ENOUGH_BUFFER
endif

TARGET = comm.tsubame3.0_p100_omp4.5
endif

# Nvidia Tesla V100 on Marconi100 (Cineca, Italy)
ifneq (, $(findstring v100_marconi,$(DEVICES)))
CXX = mpic++
#CXX = nvc++
CXXFLAGS = -O3 -std=c++17 -mp=gpu -mcmodel=medium -fast -Minfo=accel -Mcudalib=cufft,cublas -DENABLE_OPENMP_OFFLOAD
LDFLAGS = -mp=gpu -Mcudalib=cufft,cublas

ifneq (,$(findstring STEP1,$(OPTIMIZATION)))
  CXXFLAGS += -DLONG_ENOUGH_BUFFER
endif

TARGET = vlp4d.m100_v100_omp4.5
LINKER = mpic++
endif

# Nvidia Tesla A100 on Wisteria (Univ. Tokyo, Japan)
ifneq (,$(findstring A100,$(DEVICES)))
CXX = nvc++
CXXFLAGS = -O3 -std=c++17 -mp=gpu -mcmodel=medium -fast -Minfo=accel -Mcudalib=cufft,cublas -DENABLE_OPENMP_OFFLOAD
LDFLAGS = -mp=gpu -Mcudalib=cufft,cublas

ifneq (,$(findstring STEP1,$(OPTIMIZATION)))
  CXXFLAGS += -DLONG_ENOUGH_BUFFER
endif

TARGET = vlp4d.WISTERIA_A100_omp4.5
LINKER = mpic++
endif

# Source files
SRCS = $(wildcard ./*.cpp)

# Object files
OBJS = $(patsubst %.cpp,%.o,$(SRCS))

# Targets
all: $(TARGET)

$(TARGET): $(OBJS)
	$(LINKER) -o $@ $^ $(LDFLAGS)
	mv $(TARGET) ../wk

# Compiler .cpp files to create object files
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.SUFFIXES: .c .cpp .cu .o
.PHONY: all clean
.PHONY: clean
clean:
	rm -f *.o *.lst *.optrpt $(TARGET) ../wk/$(TARGET)
