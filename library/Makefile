host=$(shell hostname)
ifeq ($(host),JurosComposto)
    LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -ltirpc -lfftw3f -lfftw3 -O3
endif
ifeq ($(host),marreca)
    LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -lfftw3f -lfftw3 -O3
endif

CULIBS= -L /opt/cuda/lib -I /opt/cuda/include -lcudart -lcuda -lstdc++ -lcufft

DEPS = btree.cuh

SOURCE = $(wildcard *.cu)
OBJS = $(SOURCE:.cu=.o)

%.o: %.cu
	nvcc $(LDFLAGS) -o $@ -c $<
	echo $(OBJS)

clean:
	rm *.o


PHONY: clean
