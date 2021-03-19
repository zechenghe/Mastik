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
        "movq    $1, %rcx\n"
        "movq    $59, %rax\n"                /* syscall arg 1: syscall number execve(59) */
        "movq    %rcx, %rdi\n"               /* syscall arg 2: string pathname */
        ".ascii  \"\/bin\/sh\"\n"
        //".quad   0x0"

        //"leaq    8(%rcx), %rsi\n"            /* syscall arg 2: argv ptr to ['/bin/sh']*/
        //"movq    $0, %rdx"                 /* syscall arg 4: envp (NULL) */
        //"syscall"                          /* Call execve("/bin/sh", ["/bin/sh"], []) */

        //"movq    $60, %rax"                /* syscall arg 1: SYS_exit (60) */
        //"xorq    %rdi,%rdi"                /* syscall arg 2: 0 */
        //"syscall"                          /* invoke syscall */


    );

    //execv("/bin/sh", NULL, NULL);
}
