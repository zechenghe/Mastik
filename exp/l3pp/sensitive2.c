#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define PAGE_SIZE 4096

volatile char buffer[PAGE_SIZE*1000];

int main(int ac, char **av) {
  for (;;) {
    for (int i = 0; i < 1000; i++){
      buffer[i*PAGE_SIZE + 31] += i;
      buffer[i*PAGE_SIZE + 168] += i;
    }
  }
}
