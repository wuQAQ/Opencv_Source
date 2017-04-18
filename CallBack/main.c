#include <stdio.h>

void PrintNum(int n);
void ShowNum(int n, void (*ptr)());

void PrintMessage1();
void PrintMessage2();
void PrintMessage3();
void ShowMessage(void (* ptr)());

int main() 
{
  ShowNum(11111, PrintNum);
  ShowNum(22222, PrintNum);
  ShowMessage(PrintMessage1);
  ShowMessage(PrintMessage2);
  ShowMessage(PrintMessage3);
}

void PrintNum(int n)
{
  printf("Test1 is called, the number is %d\n", n);
}

void ShowNum(int n, void (* ptr)())
{
  (* ptr)(n);
}

void PrintMessage1()
{
  printf("This is the message 1!\n");
}

void PrintMessage2()
{
  printf("This is the message 2!\n");
}

void PrintMessage3()
{
  printf("This is the message 3!\n");
}

void ShowMessage(void (*ptr)())
{
  (* ptr)();
}