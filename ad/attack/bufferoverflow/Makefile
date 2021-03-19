override CFLAGS += -fno-stack-protector -z execstack -mpreferred-stack-boundary=4
CC=gcc

SOURCES := $(wildcard *.c)
BINARIES := $(SOURCES:%.c=%)

all: $(BINARIES)

test_call_shell: test_call_shell.c
	$(CC) $(CFLAGS) -s test_call_shell.c -o test_call_shell.s
	$(CC) $(CFLAGS) -o test_call_shell.c test_call_shell.o

clean:
	rm -f *.o $(BINARIES)
	make clean -C libkdump
