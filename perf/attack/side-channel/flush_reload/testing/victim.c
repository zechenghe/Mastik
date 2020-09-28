#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>

#define FLUSH_RELOAD_CYCLE 300

unsigned long get_tsc_val(void)
{
        unsigned int higher32bits, lower32bits;
        __asm__ volatile("rdtsc":"=a"(lower32bits),"=d"(higher32bits));
        return lower32bits;
}

void delay(int cycles)
{
        unsigned long tsc = get_tsc_val();
        while(get_tsc_val() - tsc < cycles);
}

int main(int argc, char** argv)
{
        int fd = open("/mnt/hugepages/nebula1", O_CREAT|O_RDWR, 0755);
        unsigned long *buf = mmap(0, 1024*1024*1024, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        int i;
        while (1) {
                buf[0]++;
                delay(FLUSH_RELOAD_CYCLE);
        }

        munmap(buf, 1024*1024*1024);
        return 0;
}
