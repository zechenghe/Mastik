/* victim.c */
/* Author: Zecheng He @ Princeton University */
/* Modified from https://github.com/npapernot/buffer-overflow-attack to support 64-bit machines*/

/* This program has a buffer overflow vulnerability. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
int vulnerable_func(char *str)
{
	char buffer[24];

	/* The following strcpy function has a buffer overflow problem */
	strcpy(buffer, str);
	return 1;
}

int main(int argc, char **argv)
{
	char str[1024];
	FILE *badfile;
	badfile = fopen("badfile", "r");
	fread(str, sizeof(char), 1024, badfile);
	vulnerable_func(str);
	printf("If you see this, the program returned correctly\n");
	return 0;
}
