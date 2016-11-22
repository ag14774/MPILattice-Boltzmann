# Makefile

EXE=d2q9-bgk

CC=mpiicc
CFLAGS= -std=c99 -Wall -Ofast -static_mpi -xAVX -march=native -mtune=native -m64 -fopenmp -D NOFUNCCALL -qopt-report=5 -qopt-report-phase=vec
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
	rm -f machine.file.*
	rm -f host.file.*
	rm d2q9-bgk.80s-*

profile: 
	$(CC) -g $(CFLAGS) $(EXE).c $(LIBS) -o $(EXE)

debug: 
	$(CC) $(CFLAGS) -g -DDEBUG $(EXE).c $(LIBS) -o $(EXE)
