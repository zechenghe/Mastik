#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define PAGE_SIZE 4096
#define NPAGES 1024 * 16

int main(int ac, char **av) {
  int idx = 0;
  char temp = 0;
  char* buffer = (char*)mmap(0, NPAGES * PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
  if (!buffer){
    printf("mmap error");
    exit(1);
  }

  srand(0);
  for (int i = 0; i < NPAGES * PAGE_SIZE; i++){
    asm volatile ("clflush 0(%0)": : "r" (buffer + i):);
  }

  asm volatile("lfence");
  asm volatile("lfence");

  printf("RAND_MAX %d\n", RAND_MAX);
  printf("Start For Loop\n");

  while(1){
      for (int i = 0; i < NPAGES * PAGE_SIZE; i++){
        temp = buffer[i];
        temp = temp * 2 + 1024;
      }
  }
}
