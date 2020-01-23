/*
 * Copyright 2016 CSIRO
 *
 * This file is part of Mastik.
 *
 * Mastik is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Mastik is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Mastik.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <symbol.h>
#include <ff.h>
#include <util.h>
#include <vlist.h>

#define SAMPLES 100000
#define SLOT	2000
#define THRESHOLD 100

struct ff {
  vlist_t vl;
  int modified;
  uint16_t *thresholds;
  int thresholdcap;
};

char *monitor[] = {
  "mpih-mul.c:85",
  "mpih-mul.c:271",
  "mpih-div.c:356"
};
int nmonitor = sizeof(monitor)/sizeof(monitor[0]);

void usage(const char *prog) {
  fprintf(stderr, "Usage: %s <gpg-binary>\n", prog);
  exit(1);
}


int main(int ac, char **av) {
  char *binary = av[1];
  if (binary == NULL)
    usage(av[0]);

  delayloop(2000000000);

  ff_t ff = ff_prepare();
  for (int i = 0; i < nmonitor; i++) {
    uint64_t offset = sym_getsymboloffset(binary, monitor[i]);
    if (offset == ~0ULL) {
      fprintf(stderr, "Cannot find %s in %s\n", monitor[i], binary);
      exit(1);
    }
    ff_monitor(ff, map_offset(binary, offset));
  }

  uint16_t *res = malloc(SAMPLES * nmonitor * sizeof(uint16_t));
  for (int i = 0; i < SAMPLES * nmonitor ; i+= 4096/sizeof(uint16_t))
    res[i] = 1;
  ff_probe(ff, res);

  int l;
  do {
    // int ff_trace(ff_t ff, int max_records, uint16_t *results, int slot, int threshold, int max_idle)
    // return count;
    l = ff_trace(ff, SAMPLES, res, SLOT, THRESHOLD, 500);

    for (int i=0; i<vl_len(ff->vl); i++){
      printf("%d ", vl_get(ff->thresholds, i));
    }
    putchar('\n');

  } while (l < 10000);

/*
  for (int i = 0; i < l; i++) {
    for (int j = 0; j < nmonitor; j++)
      printf("%d ", res[i * nmonitor + j]);
    putchar('\n');
  }
*/

  free(res);
  ff_release(ff);
}
