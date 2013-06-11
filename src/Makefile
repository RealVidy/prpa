UNAME = $(shell uname)

# running on hive machines
ifeq ($(UNAME),Linux)
CXX = g++ -std=c++11
endif

INCLUDES =
OMP = -fopenmp
LIBS = -lpthread
CBLAS = -lblas -lm
# a pretty good flag selection for this machine...
CFLAGS = -msse4 -O3 -pipe -fno-omit-frame-pointer
OUT_PATH = ../

all:	bench-naive bench-all bench-para
	#bench-openmp

# triple nested loop implementation
bench-naive: benchmark.o sgemm-naive.o
	$(CXX) -o $(OUT_PATH)$@ $(LIBS) benchmark.o sgemm-naive.o $(CBLAS)
# your implementation for component 1
bench-small: benchmark.o sgemm-small.o
	$(CXX) -o $(OUT_PATH)$@ $(LIBS) benchmark.o sgemm-small.o $(CBLAS)
# your implementation for component 2
bench-all: benchmark.o sgemm-all.o
	$(CXX) -o $(OUT_PATH)$@ $(LIBS) benchmark.o sgemm-all.o $(CBLAS)
bench-para: benchmark.o sgemm-para.o
	$(CXX) -o $(OUT_PATH)$@ $(LIBS) benchmark.o sgemm-para.o $(CBLAS)
# your implementation for component 3
bench-openmp: benchmark.o sgemm-openmp.o
	$(CXX) -o $(OUT_PATH)$@ $(LIBS) $(OMP) benchmark.o sgemm-openmp.o $(CBLAS)

%.o: %.c
	$(CXX) -c $(CFLAGS) $(INCLUDES) $<
%.o: %.cc
	$(CXX) -c $(CFLAGS) $(INCLUDES) $<

clean:
	rm -f *~ *.o
	cd $(OUT_PATH) && rm -f bench-naive bench-small bench-all \
		bench-openmp bench-para
