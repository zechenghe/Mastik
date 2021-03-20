/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

/* This program has a buffer overflow vulnerability. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
int vulnerable_func(char *str)
{
	char buffer[72];

	/* The following strcpy function has a buffer overflow problem */
	strcpy(buffer, str);
	return 0;
}

int main(int argc, char **argv)
{
	char str[1024];
	FILE *badfile;
    int n=0;
	badfile = fopen("badfile", "r");
	n=fread(str, sizeof(char), 1024, badfile);
    printf("%d %c%c%c%c", n, str[0], str[1], str[2], str[3]);

	//vulnerable_func(str);
	printf("If you see this, the program returned correctly\n");
	return 0;
}
