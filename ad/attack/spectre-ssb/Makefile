CC=gcc
CYCLES?=100
#CFLAGS=-Wall -DHIT_THRESHOLD=$(CYCLES) -DNO_INTERRUPTS
CFLAGS=-Wall -DHIT_THRESHOLD=$(CYCLES)

ssb: ssb.c
	$(CC) -o ssb_$(CYCLES) ssb.c $(CFLAGS)

clean:
	rm ssb_*
