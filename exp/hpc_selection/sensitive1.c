#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

volatile char buffer[4096];

int main(int ac, char **av) {
  for (;;) {
    for (int i = 0; i < 64000; i++)
      buffer[666] += i;
    for (int i = 0; i < 64000; i++)
      buffer[1666] += i;
  }
}
