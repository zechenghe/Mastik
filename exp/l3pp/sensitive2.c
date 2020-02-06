#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define PAGE_SIZE 4096

volatile char buffer[PAGE_SIZE*1024];

int main(int ac, char **av) {
  buffer = mmap(0, npages * PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
  for (;;) {
    for (int i = 0; i < 1024; i++){
      buffer[i*PAGE_SIZE + 31] += i;
      buffer[i*PAGE_SIZE + 168] += i;
    }
  }
}
