# Device can be p100, bdw, knl
DEVICES = $(DEVICE)

# Nvidia P100 on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring p100,$(DEVICES)))
CXX      = mpicxx
CXXFLAGS = -O3 -ta=nvidia:cc60 -Minfo=accel -Mcudalib=cufft,cublas -std=c++11 -DENABLE_OPENACC -DLAYOUT_LEFT
LDFLAGS  = -Mcudalib=cufft,cublas -ta=nvidia:cc60 -acc
TARGET   = vlp4d.tsubame3.0_p100_openacc
endif

# Intel Broadwell on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring bdw,$(DEVICES)))
CXXFLAGS=-O3 -std=c++11 -qopenmp -xCORE-AVX2 -DLAYOUT_LEFT
CXX=mpicxx
LDFLAGS = -lm -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f -qopenmp
TARGET = vlp4d.tsubame3.0_bdw_openmp
endif

# Intel Knights Landing on Oakforest-PACS (Univ. Tokyo, Japan)
ifneq (,$(findstring knl,$(DEVICES)))
CXXFLAGS=-O3 -std=c++11 -qopenmp -axMIC-AVX512 -DLAYOUT_LEFT
CXX=mpiicpc
LDFLAGS = -lm -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f -qopenmp -lstdc++
TARGET = vlp4d.pacs_knl_openmp
endif

# Linker
LINKER = $(CXX)
	         
# Source files
SRCS = $(wildcard ./*.cpp)
		     
# Object files
OBJS = $(patsubst %.cpp,%.o,$(SRCS))
		        
# Targets
all: $(TARGET)
			   
$(TARGET) : $(OBJS)
	$(LINKER) -o $@ $^ $(LDFLAGS)
	mv $(TARGET) ../wk
			      
# Compiler .cpp files to create object files
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
				  
.SUFFIXES: .c .cpp .cu .o
.PHONY: all clean
.PHONY: clean
					     
clean:
	rm -f *.o $(TARGET) ../wk/$(TARGET)
