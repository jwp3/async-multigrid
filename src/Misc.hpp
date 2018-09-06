#ifndef MISC_HPP
#define MISC_HPP

#include "Main.hpp"

void PrintOutput(AllData all_data);

double RandDouble(double low, double high);

double Norm2(double *x, int n);

int SumInt(int *x, int n);

double SumDbl(double *x, int n);

void QuicksortPair_int_dbl(int *x, double *y, int first, int last);

void SMEM_LevelBarrier(AllData *all_data,
                       int level);

#endif
