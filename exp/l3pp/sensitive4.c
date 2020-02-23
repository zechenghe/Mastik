#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define PAGE_SIZE 4096
#define CACHELINE_SIZE 64

int main(int ac, char **av) {
  int npages = 1024;
  long long i = 0;
  int temp = 0;

  char* buffer = (char*)mmap(0, npages * PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
  if (!buffer){
    printf("mmap error");
    exit(1);
  }

  srand(0);
  for (i = 0; i < npages * PAGE_SIZE; i++){
    asm volatile ("clflush 0(%0)": : "r" (buffer + i):);
  }

  asm volatile("mfence");
  asm volatile("mfence");

  printf("RAND_MAX", RAND_MAX);
  printf("Start For Loop\n");

  while(1){
      temp = buffer[rand() % npages * PAGE_SIZE];
  }
}
