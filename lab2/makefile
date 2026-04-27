SRCS = lib/gemm.h lib/gemm.cpp lib/main.cpp mpi.cpp

UNAME_S := $(shell uname -s)
ifeq ($(shell uname -m),arm64)
    SRCS += lib/gemm-baseline-aarch64-g++.a lib/gemm-baseline-aarch64-llvm.a
else
	ifeq ($(UNAME_S),Darwin)
		SRCS += lib/gemm-baseline-darwin-g++.a lib/gemm-baseline-darwin-llvm.a
	else
		SRCS += lib/gemm-baseline-linux-g++.a
	endif
endif

REQUIRED_FILES = mpi.cpp lab2-report.pdf

np ?= 4

test: gemm
	mpiexec -np $(np) ./$^

gemm: $(SRCS)
	$(CXX) $(CXXFLAGS) $(LAB2_CXX_FLAGS) -o $@ $(filter %.cpp %.a %.o, $^)

clean:
	$(RM) gemm


include ../common/makefile.inc
CXX = MPICH_CXX=$(MPICH_CXX) mpicxx  # specify your compiler here
LDFLAGS +=  # specify your library linking options here
