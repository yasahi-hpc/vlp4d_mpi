# Device can be p100, bdw, knl
DEVICES = $(DEVICE)

# Nvidia V100 on Marconi100 (Cineca, Italy)
ifneq (,$(findstring v100_marconi,$(DEVICES)))
  CXX      = mpicxx
  CXXFLAGS = -O3 -ta=nvidia:cc70 -Minfo=accel -Mcudalib=cufft,cublas -std=c++11 -DENABLE_OPENACC -DLAYOUT_LEFT
  ifneq (,$(findstring BUFFER,$(OPTIMIZATION)))
    CXXFLAGS += -DLONG_ENOUGH_BUFFER
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    CXXFLAGS += -DNO_ERROR_CHECK -DLONG_ENOUGH_BUFFER
  endif
  LDFLAGS  = -Mcudalib=cufft,cublas -ta=nvidia:cc70 -acc
  TARGET   = vlp4d.m100_v100_openacc
endif

# Nvidia A100 on A100 on Wisteria (Univ. Tokyo, Japan)
ifneq (,$(findstring A100_Wisteria,$(DEVICES)))
  CXX      = mpicxx
  CXXFLAGS = -O3 -ta=nvidia:cc80 -Minfo=accel -Mcudalib=cufft,cublas -std=c++11 -DENABLE_OPENACC -DLAYOUT_LEFT
  ifneq (,$(findstring BUFFER,$(OPTIMIZATION)))
    CXXFLAGS += -DLONG_ENOUGH_BUFFER
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    CXXFLAGS += -DNO_ERROR_CHECK -DLONG_ENOUGH_BUFFER
  endif
  LDFLAGS  = -Mcudalib=cufft,cublas -ta=nvidia:cc80 -acc
  TARGET   = vlp4d.WISTERIA_A100_openacc
endif

# Nvidia P100 on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring p100,$(DEVICES)))
  CXX      = mpicxx
  CXXFLAGS = -O3 -ta=nvidia:cc60 -Minfo=accel -Mcudalib=cufft,cublas -std=c++11 -DENABLE_OPENACC -DLAYOUT_LEFT
  ifneq (,$(findstring BUFFER,$(OPTIMIZATION)))
    CXXFLAGS += -DLONG_ENOUGH_BUFFER
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    CXXFLAGS += -DNO_ERROR_CHECK -DLONG_ENOUGH_BUFFER
  endif
  LDFLAGS  = -Mcudalib=cufft,cublas -ta=nvidia:cc60 -acc
  TARGET   = vlp4d.tsubame3.0_p100_openacc
endif

# Intel Broadwell on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring bdw,$(DEVICES)))
  CXX=mpicxx
  CXXFLAGS=-O3 -std=c++11 -qopenmp -xCORE-AVX2 -DLAYOUT_LEFT
  ifneq (,$(findstring SIMD,$(OPTIMIZATION)))
    CXXFLAGS += -DSIMD
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    CXXFLAGS += -DNO_ERROR_CHECK -DSIMD
  endif
  LDFLAGS = -lm -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f -qopenmp
  TARGET = vlp4d.tsubame3.0_bdw_openmp
endif

# Intel Knights Landing on Oakforest-PACS (Univ. Tokyo, Japan)
ifneq (,$(findstring knl,$(DEVICES)))
  CXX=mpiicpc
  CXXFLAGS=-O3 -std=c++11 -qopenmp -axMIC-AVX512 -DLAYOUT_LEFT
  ifneq (,$(findstring SIMD,$(OPTIMIZATION)))
    CXXFLAGS += -DSIMD
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    CXXFLAGS += -DNO_ERROR_CHECK -DSIMD
  endif
  LDFLAGS = -lm -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f -qopenmp -lstdc++
  TARGET = vlp4d.pacs_knl_openmp
endif

# Fujitsu A64FX on Flow (Nagoya Univ, Japan)
ifneq (, $(findstring a64fx_flow,$(DEVICES)))
  #Fujitsu
  CXX = mpiFCCpx
  CXXFLAGS = -O3 -g -std=c++14 -Kfast,openmp,optmsg=2 -march=armv8.2-a+sve -Nlst=t -DLAYOUT_LEFT -Krestp=all -Kocl
  ifneq (,$(findstring SIMD,$(OPTIMIZATION)))
    CXXFLAGS += -DSIMD -DFUJI
  endif
  ifneq (,$(findstring BUFFER,$(OPTIMIZATION)))
    CXXFLAGS += -DLONG_ENOUGH_BUFFER -DSIMD -DFUJI
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    CXXFLAGS += -DNO_ERROR_CHECK -DSIMD -DFUJI
  endif
  LDFLAGS  = -Kfast,openmp -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f -lstdc++
  TARGET   = vlp4d.flow_a64fx_openmp
endif

# Intel Skylake on JFRS-1 (IFERC-CSC, Japan)
ifneq (, $(findstring skx,$(DEVICES)))
  CXX = CC
  CXXFLAGS = -O3 -std=c++11 -qopenmp -xCORE-AVX512 -DLAYOUT_LEFT
  ifneq (,$(findstring SIMD,$(OPTIMIZATION)))
    CXXFLAGS += -DSIMD
  endif
  ifneq (,$(findstring BUFFER,$(OPTIMIZATION)))
    CXXFLAGS += -DLONG_ENOUGH_BUFFER -DSIMD
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    CXXFLAGS += -DNO_ERROR_CHECK -DSIMD
  endif
  LDFLAGS  = -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f -lstdc++ -qopenmp
  TARGET   = vlp4d.jfrs1_skx_openmp
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
	rm -f *.o *.lst $(TARGET) ../wk/$(TARGET)
