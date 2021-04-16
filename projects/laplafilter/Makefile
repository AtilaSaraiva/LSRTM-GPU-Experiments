host=$(shell hostname)
ifeq ($(host),JurosComposto)
    LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -ltirpc -lfftw3f -lfftw3 -O3
endif
ifeq ($(host),marreca)
    LDFLAGS= -I$(RSFROOT)/include -L$(RSFROOT)/lib -lrsf++ -lrsf -lm -lfftw3f -lfftw3 -O3
endif

CULIBS= -L /opt/cuda/lib -I /opt/cuda/include -lcudart -lcuda -lstdc++ -lcufft

lap: laplafilter.cu
	nvcc laplafilter.cu $(LDFLAGS) -o lap

run: lap
	#sfspike n1=425 n2=368 d1=8 d2=25 mag=3300 > vel1.rsf
	#sfspike n1=425 n2=368 d1=8 d2=25 mag=1000 k1=250 l1=425 > vel2.rsf
	sfspike n1=400 n2=400 d1=8 d2=25 mag=3300 > vel1.rsf
	sfspike n1=400 n2=400 d1=8 d2=25 mag=1000 k1=250 l1=400 > vel2.rsf
	sfadd mode=a <vel1.rsf vel2.rsf >imagem.rsf
	./lap img=imagem.rsf lap=lapla.rsf
	sfgrey <lapla.rsf >lapla.vpl
