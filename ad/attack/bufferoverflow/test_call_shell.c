/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    execv("/bin/sh", "/bin/sh", NULL);
}
