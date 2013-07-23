CC = gcc -std=gnu99
GOTOLIB = -lm -lblas -lprofiler
CPU_TYPE = native
OMP = -fopenmp
CFLAGS = -msse4 -O3 $(DEBUG) -pipe -fno-omit-frame-pointer $(OMP) -march=$(CPU_TYPE)
OUT = ../

all: bench-naive bench-naive-para bench-normal bench-openmp

bench-naive: benchmark.o sgemm-naive.o
	$(CC) -o $(OUT)$@ benchmark.o sgemm-naive.o $(GOTOLIB)
bench-naive-para: benchmark.o sgemm-naive-para.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-naive-para.o $(GOTOLIB)
bench-normal: benchmark.o sgemm-normal.o
	$(CC) -o $(OUT)$@ benchmark.o sgemm-normal.o $(GOTOLIB)
bench-openmp: benchmark.o sgemm-openmp.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-openmp.o $(GOTOLIB)


%.o: %.c
	$(CC) -c $(CFLAGS) $<

# Get assembly code with corresponding c code
#%.o: %.c
#	$(CC) -c -g -Wa,-a,-ad $(CFLAGS) $(INCLUDES) $< > $(OUT)$<.lst

clean:
	@cd ..; rm -f bench-naive bench-naive-para bench-openmp bench-normal *.lst
	@rm -f *.o
