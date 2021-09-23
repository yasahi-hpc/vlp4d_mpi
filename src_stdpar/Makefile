# Device can be v100
DEVICES = $(DEVICE)

# Nvidia P100 on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring p100_tsubame3,$(DEVICES)))
CXX = nvc++
CXXFLAGS = -O3 -std=c++17 -stdpar=gpu -cuda -gpu=cuda11.4,cc60
LDFLAGS = -Mcudalib=cufft,cublas -stdpar=gpu -cuda -gpu=cuda11.4,cc60
TARGET = vlp4d.tsubame3.0_P100_stdpar
LINKER = mpic++
endif

# Nvidia Tesla V100 on Marconi100 (Cineca, Italy)
ifneq (, $(findstring v100_marconi,$(DEVICES)))
CXX = nvc++
CXXFLAGS = -O3 -std=c++17 -stdpar=gpu -cuda -gpu=cuda11.2,cc70
LDFLAGS = -stdpar=gpu
TARGET = vlp4d.m100_V100_stdpar
LINKER = mpic++
endif

# Nvidia Tesla A100 on Wisteria (Univ. Tokyo, Japan)
ifneq (,$(findstring A100,$(DEVICES)))
CXX = nvc++
CXXFLAGS = -O3 -std=c++17 -stdpar=gpu -cuda -gpu=cuda11.2,cc80 -w -fast -Mnouniform -Mfprelaxed -Mcudalib=cufft,cublas
LDFLAGS = -stdpar=gpu -cuda -gpu=cuda11.2,cc80 -Mcudalib=cufft,cublas
TARGET = vlp4d.WISTERIA_A100_stdpar
LINKER = mpic++
endif

# Intel Ice Lake on Wisteria (Univ. Tokyo, Japan)
ifneq (,$(findstring IceLake,$(DEVICES)))
CXX = nvc++
CXXFLAGS = -O3 -std=c++17 -stdpar=multicore -mp -w -fast
LDFLAGS = -stdpar=multicore -mp
TARGET = vlp4d.WISTERIA_${DEVICES}_stdpar
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

.PHONY: clean
clean:
	rm -f $(TARGET) ../wk/$(TARGET) *.o *.lst
