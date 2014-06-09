#include <stdio.h>
#include <stdlib.h>
#include "Timer_Test.h"

int main_Seconds(int argc, char** argv)
{
    // Number of elements in the array
    for (unsigned int N = 1; N < 10; N++)
    {
        Test::TimerSeconds(N);
    }

    return 0;
}


int main_MilliSeconds(int argc, char** argv)
{
    // Number of elements in the array
    for (unsigned int N = 1; N < 10; N++)
    {
        Test::TimerMilliSeconds(N);
    }

    return 0;
}


int main_MicroSeconds(int argc, char** argv)
{
    // Number of elements in the array
    for (unsigned int N = 1; N < 10; N++)
    {
        Test::TimerMicroSeconds(N);
    }

    return 0;
}


int main(int argc, char** argv)
{
    main_MilliSeconds(argc, argv);
    return 0;
}
