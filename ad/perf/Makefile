KDIR := /lib/modules/$(shell uname -r)/build
CC = gcc
TARGET = event_open_user
obj-m += event_open_kernel.o

all: kernel user gettimer

kernel:
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) modules

user: $(TARGET).c
	$(CC) -o $(TARGET) $(TARGET).c

gettimer: gettimer.c
	$(CC) -o gettimer gettimer.c

clean:
	rm -rf *.o *.ko *.mod.* *.cmd .module* modules* Module* .*.cmd .tmp* $(TARGET) gettimer
