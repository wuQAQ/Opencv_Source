#include <stdio.h>
int main()
{
    char str[5];
    int ret = snprintf(str, 3, "%s", "abcdefg");
    printf("%d\n",ret);
    printf("%s",str);
    return 0;
}