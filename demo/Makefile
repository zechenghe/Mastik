FILES=$(wildcard *.c)
TARGETS=$(FILES:.c=)
OBJS=${FILES:.c=.o}
CFLAGS=-g -std=gnu99 -I../src
LDFLAGS=-L../src/ -g
<<<<<<< HEAD
LDLIBS=-lmastik -ldwarf -lelf -lbfd 
=======
LDLIBS=-lmastik 
>>>>>>> 1641ac23e7dce4236a1e8d5fd168dacff2d848a0


all: ${TARGETS}

${TARGETS}: %: %.o 
	${CC} ${LDFLAGS} -o $@ $@.o ${LDLIBS}


clean:
	rm -f ${TARGETS} ${OBJS} *.sig out

distclean: clean
	rm Makefile
