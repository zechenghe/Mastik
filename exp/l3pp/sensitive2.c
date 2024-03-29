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
  char* buffer = (char*)mmap(0, npages * PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
  if (!buffer){
    printf("mmap error");
    exit(1);
  }

  for (;;) {
    for (int i = 0; i < 65536; i++){
      buffer[i*CACHELINE_SIZE + 31] += i;
      buffer[i*CACHELINE_SIZE + 41] += i;
    }
  }
}
