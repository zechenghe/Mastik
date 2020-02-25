#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>

#define PAGE_SIZE 4096
#define CACHELINE_SIZE 64
#define NPAGES 1024

int main(int argc, char **argv) {
  char temp = 0;
  int fd = open(argv[1]);
  char* buffer = (char*)mmap(0, NPAGES * PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, fd, 0);
  if (!buffer){
    printf("mmap error");
    exit(1);
  }

  srand(0);
  for (int i = 0; i < NPAGES * PAGE_SIZE; i++){
    asm volatile ("clflush 0(%0)": : "r" (buffer + i):);
  }

  asm volatile("mfence");
  asm volatile("mfence");

  printf("RAND_MAX %d\n", RAND_MAX);
  printf("Start For Loop\n");

  while(1){
      temp = buffer[rand() % (NPAGES * PAGE_SIZE)];
      asm volatile("mfence");
  }
}
