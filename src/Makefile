CC = gcc -std=gnu99
GOTOLIB = -lm -lblas -lprofiler
CPU_TYPE = native
OMP = -fopenmp
CFLAGS = -msse4 -O3 $(DEBUG) -pipe -fno-omit-frame-pointer $(OMP) -march=$(CPU_TYPE)
OUT = ../

all: bench-all bench-para bench-new bench-naive bench-naive-para

bench-naive: benchmark.o sgemm-naive.o
	$(CC) -o $(OUT)$@ benchmark.o sgemm-naive.o $(GOTOLIB)
bench-naive-para: benchmark.o sgemm-naive-para.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-naive-para.o $(GOTOLIB)
bench-all: benchmark.o sgemm-all.o
	$(CC) -o $(OUT)$@ benchmark.o sgemm-all.o $(GOTOLIB)
bench-para: benchmark.o sgemm-para.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-para.o $(GOTOLIB)
bench-new: benchmark.o sgemm-new.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-new.o $(GOTOLIB)
bench-openmp: benchmark.o sgemm-openmp.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-openmp.o $(GOTOLIB)


%.o: %.c
	$(CC) -c $(CFLAGS) $<

# Get assembly code with corresponding c code
#%.o: %.c
#	$(CC) -c -g -Wa,-a,-ad $(CFLAGS) $(INCLUDES) $< > $(OUT)$<.lst

clean:
	@cd ..; rm -f bench-naive bench-naive-para bench-small bench-all bench-openmp bench-para bench-new *.lst
	@rm -f *.o
