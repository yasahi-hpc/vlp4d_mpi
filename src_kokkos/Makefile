# Device can be p100, v100, bdw, skx, tx2
KOKKOS_DEVICES = $(DEVICE)
SRC = $(wildcard ./*.cpp)
vpath %.cpp $(sort $(dir $(SRC)))

default: build
	echo "Start Build"

# Nvidia Tesla V100 on Marconi100 (Cineca, Italy)
ifneq (,$(findstring v100_marconi,$(KOKKOS_DEVICES)))
  CXX = mpicxx
  CXXFLAGS = -O3
  ifneq (,$(findstring TILE_SIZE,$(OPTIMIZATION)))
    # enable auto-tuning against tile sizes
    CXXFLAGS += -DTILE_SIZE_TUNING
  endif
  ifneq (,$(findstring ALL,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DTILE_SIZE_TUNING -DNO_ERROR_CHECK
  endif
  LINK = mpicxx
  LDFLAGS = -lcufft -lcublas
  EXE = vlp4d.m100_v100_kokkos
  KOKKOS_DEVICES = "Cuda,OpenMP"
  KOKKOS_ARCH = "POWER9,VOLTA70"
  KOKKOS_CUDA_OPTIONS += "enable_lambda"
endif

# Nvidia TeslaP100 on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring p100,$(KOKKOS_DEVICES)))
  CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
  CXXFLAGS = -O3
  ifneq (,$(findstring TILE_SIZE,$(OPTIMIZATION)))
    # enable auto-tuning against tile sizes
    CXXFLAGS += -DTILE_SIZE_TUNING
  endif
  ifneq (,$(findstring ALL,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DTILE_SIZE_TUNING -DNO_ERROR_CHECK
  endif
  LINK = mpicxx
  LDFLAGS = -lcufft -lcublas
  EXE = vlp4d.tsubame3.0_p100_kokkos
  KOKKOS_DEVICES = "Cuda,OpenMP"
  KOKKOS_ARCH = "BDW,Pascal60"
  KOKKOS_CUDA_OPTIONS += "enable_lambda"
endif

# Intel Broadwell on Tsubame3.0 (Tokyo Tech, Japan)
ifneq (,$(findstring bdw,$(KOKKOS_DEVICES)))
  CXX = icpc
  CXXFLAGS = -O3
  ifneq (,$(findstring TILE_SIZE,$(OPTIMIZATION)))
    # enable auto-tuning against tile sizes
    CXXFLAGS += -DTILE_SIZE_TUNING
  endif
  ifneq (,$(findstring LAYOUT_LEFT,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DLAYOUT_LEFT
  endif
  ifneq (,$(findstring NO_SCATTER_VIEW,$(OPTIMIZATION)))
    # disable scatter views and use parallel reduce for error check
    CXXFLAGS += -DNO_SCATTER_VIEWS
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DLAYOUT_LEFT -DTILE_SIZE_TUNING -DNO_SCATTER_VIEWS
  endif
  ifneq (,$(findstring ALL,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DLAYOUT_LEFT -DTILE_SIZE_TUNING -DNO_ERROR_CHECK
  endif
  LINK = mpicxx
  LDFLAGS = -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f
  EXE = vlp4d.tsubame3.0_bdw_kokkos
  KOKKOS_DEVICES = "OpenMP"
  KOKKOS_ARCH = "BDW"
endif

# Intel Skylake on JFRS-1 (IFERC-CSC, Japan)
ifneq (,$(findstring skx,$(KOKKOS_DEVICES)))
  CXX = CC
  CXXFLAGS = -O3
  ifneq (,$(findstring TILE_SIZE,$(OPTIMIZATION)))
    # enable auto-tuning against tile sizes
    CXXFLAGS += -DTILE_SIZE_TUNING
  endif
  ifneq (,$(findstring LAYOUT_LEFT,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DLAYOUT_LEFT
  endif
  ifneq (,$(findstring NO_SCATTER_VIEW,$(OPTIMIZATION)))
    # disable scatter views and use parallel reduce for error check
    CXXFLAGS += -DNO_SCATTER_VIEWS
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DLAYOUT_LEFT -DTILE_SIZE_TUNING -DNO_SCATTER_VIEWS
  endif
  ifneq (,$(findstring ALL,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DLAYOUT_LEFT -DTILE_SIZE_TUNING -DNO_ERROR_CHECK
  endif
  LINK = ${CXX}
  LDFLAGS = -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f
  EXE = vlp4d.jfrs1_skx_kokkos
  KOKKOS_DEVICES = "OpenMP"
  KOKKOS_ARCH = "SKX"
endif

# Fujitsu A64FX on flow (Nagoya Univ, Japan)
ifneq (,$(findstring a64fx_flow,$(KOKKOS_DEVICES)))
  CXX = mpiFCCpx
  CXXFLAGS = -O3 -Kfast,openmp -Krestp=all
  ifneq (,$(findstring TILE_SIZE,$(OPTIMIZATION)))
    # enable auto-tuning against tile sizes
    CXXFLAGS += -DTILE_SIZE_TUNING
  endif
  ifneq (,$(findstring LAYOUT_LEFT,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DLAYOUT_LEFT
  endif
  ifneq (,$(findstring NO_SCATTER_VIEW,$(OPTIMIZATION)))
    # disable scatter views and use parallel reduce for error check
    CXXFLAGS += -DNO_SCATTER_VIEWS
  endif
  ifneq (,$(findstring BOTH,$(OPTIMIZATION)))
    # enable all the optimizations
    CXXFLAGS += -DLAYOUT_LEFT -DTILE_SIZE_TUNING -DNO_SCATTER_VIEWS
  endif
  ifneq (,$(findstring ALL,$(OPTIMIZATION)))
    # enable layout-tuning
    CXXFLAGS += -DLAYOUT_LEFT -DTILE_SIZE_TUNING -DNO_ERROR_CHECK
  endif
  LINK = ${CXX}
  LDFLAGS = -lfftw3_threads -lfftw3 -lfftw3f_threads -lfftw3f -Kfast,openmp -lstdc++
  EXE = vlp4d.flow_a64fx_kokkos
  KOKKOS_DEVICES = "OpenMP"
  KOKKOS_ARCH = "A64FX"
endif

# Nvidia TeslaV100 on Summit (OLCF, US)
ifneq (,$(findstring v100,$(KOKKOS_DEVICES)))
KOKKOS_PATH = /ccs/home/yasahi/lib/kokkos
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -O3
ifneq (,$(findstring TILE_SIZE,$(OPTIMIZATION)))
  # enable auto-tuning against tile sizes
  CXXFLAGS += -DTILE_SIZE_TUNING
endif
LINK = mpicxx
LDFLAGS = -lcufft -lcublas
EXE = vlp4d.summit_v100_kokkos
KOKKOS_DEVICES = "Cuda,OpenMP"
KOKKOS_ARCH = "Power9,Volta70"
KOKKOS_CUDA_OPTIONS += "enable_lambda"
endif

ifneq (,$(findstring tx2,$(KOKKOS_DEVICES)))
# Marvell Thunder X2 on CEA Computing Complex (CEA, France)
KOKKOS_PATH = /ccc/home/cont002/qst/asahiy/lib/src/kokkos
CXX=armclang++
CXXFLAGS=-O3 -armpl
ifneq (,$(findstring TILE_SIZE,$(OPTIMIZATION)))
  # enable auto-tuning against tile sizes
  CXXFLAGS += -DTILE_SIZE_TUNING
endif
LINK = ${CXX}
LDFLAGS=-armpl
# Execution files
EXE = vlp4d.ccc_tx2_kokkos
KOKKOS_DEVICES = "OpenMP"
KOKKOS_ARCH = "ARMv8-TX2"
endif

KOKKOS_SRC_PATH = ${KOKKOS_PATH}
DEPFLAGS = -M

OBJ = $(notdir $(SRC:.cpp=.o))
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

test: $(EXE)
	./$(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE) $(LDFLAGS)
	mv $(EXE) ../wk

clean: kokkos-clean 
	rm -f *.o vlp4d.*kokkos ../wk/vlp4d.*kokkos

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $< -o $(notdir $@)
