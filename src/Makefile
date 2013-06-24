HOME = /home/ff/cs61c
UNAME = $(shell uname)

# running on hive machines
ifeq ($(UNAME),Linux)
CC = gcc -std=gnu99
#GOTO = $(HOME)/bin/GotoBLAS2_Linux
GOTOLIB = -lm -lblas
endif

# running on 200 SD machines
ifeq ($(UNAME),Darwin)
CC = gcc -std=gnu99
#GOTO = $(HOME)/bin/GotoBLAS2
GOTOLIB = -lm -lblas
endif

CPU_TYPE = core2
OMP = -fopenmp
CFLAGS = -msse4 -O3 -pipe $(OMP) -march=$(CPU_TYPE)
OUT = ../

all: bench-naive bench-all bench-para

bench-naive: benchmark.o sgemm-naive.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-naive.o $(GOTOLIB)
bench-all: benchmark.o sgemm-all.o
	$(CC) -o $(OUT)$@ benchmark.o sgemm-all.o $(GOTOLIB)
bench-para: benchmark.o sgemm-para.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-para.o $(GOTOLIB)
bench-openmp: benchmark.o sgemm-openmp.o
	$(CC) -o $(OUT)$@ $(OMP) benchmark.o sgemm-openmp.o $(GOTOLIB)


%.o: %.c
	$(CC) -c $(CFLAGS) $<

# Get assembly code with corresponding c code
# %.o: %.c
# 	$(CC) -c -g -Wa,-a,-ad $(CFLAGS) $(INCLUDES) $< > $<.lst

clean:
	@cd ..; rm -f bench-naive bench-small bench-all bench-openmp bench-para *.o *.lst
