override CFLAGS += -fno-stack-protector -z execstack -mpreferred-stack-boundary=4 -D_FORTIFY_SOURCE=0
CC=gcc

SOURCES := $(wildcard *.c)
BINARIES := $(SOURCES:%.c=%)

all: $(BINARIES)

victim: victim.c
		$(CC) $(CFLAGS) -S victim.c -o victim.s
		$(CC) $(CFLAGS) victim.c -o victim.o

test_call_shell: test_call_shell.c
	$(CC) $(CFLAGS) -S test_call_shell.c -o test_call_shell.s
	$(CC) $(CFLAGS) test_call_shell.c -o test_call_shell.o

clean:
	rm -f *.o $(BINARIES) a.out
