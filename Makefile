include makefile_include

all:
	make -C src/ DEBUG=$(DEBUG)

clean:
	make -C src/ $@
