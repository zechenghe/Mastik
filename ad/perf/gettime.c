#include <sys/time.h>

struct timeval stop, start;
gettimeofday(&start, NULL);
//do stuff
gettimeofday(&stop, NULL);

printf("start.tv_sec %lu \n", start.tv_sec);
printf("start.tv_usec %lu \n", start.tv_usec);
printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);