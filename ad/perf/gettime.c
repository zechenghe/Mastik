#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>


int main(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    printf("%f \n", t.tv_sec);
    printf("%f %f\n", t.tv_usec, (float)(t.tv_usec) / 1000000);
    printf("%f \n", t.tv_sec + (float)(t.tv_usec) / 1000000);
}
