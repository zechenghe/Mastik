#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

uint64_t rdtsc(void) {
        uint64_t a, d;
        __asm__ volatile ("rdtsc" : "=a" (a), "=d" (d));
        return (d<<32) | a;
}


int main(int argc, char **argv) {
        struct perf_event_attr pe;
        int fd;

        int pid = atoi(argv[1]);
        uint64_t INTERVAL = 300000*atoi(argv[2]);
        int ROUND = 500;

        int j;

        uint64_t* time;
        time = (uint64_t*)malloc(ROUND*sizeof(uint64_t));
        uint64_t* value;
        value = (uint64_t*)malloc(ROUND*sizeof(uint64_t));

        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HW_CACHE;
        pe.config = (PERF_COUNT_HW_CACHE_LL)|(PERF_COUNT_HW_CACHE_OP_READ << 8)|(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        pe.size = sizeof(struct perf_event_attr);
        pe.disabled = 1;
        pe.inherit = 0;
        pe.pinned = 1;
        pe.exclude_kernel = 0;
        pe.exclude_user = 0;
        pe.exclude_hv = 0;
        pe.exclude_host = 0;
        pe.exclude_guest = 0;

        uint64_t start_cycle;

        fd = syscall(__NR_perf_event_open, &pe, pid, -1, -1, 0);
        uint64_t begin_stamp = rdtsc();
        int x[1];
	while (1) {
//      for (j=0; j<ROUND; j++) {
                ioctl(fd, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

	        x[0] = pid;
                syscall(189, x);

                start_cycle = rdtsc();
                while(rdtsc()-start_cycle<INTERVAL);

                ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
//                read(fd, &value[j], sizeof(long long));
//                time[j] = (start_cycle-begin_stamp)/1000;
        }
        close(fd);

        FILE *output = fopen("output", "a");
        for (j=0; j<ROUND; j++) {
                fprintf(output, "%lu:           %lu\n", time[j], value[j]);
        }

        fclose(output);
        return 0;
}
