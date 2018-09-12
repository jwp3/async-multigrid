#ifndef MISC_HPP
#define MISC_HPP

#include "Main.hpp"

void PrintOutput(AllData all_data);

double RandDouble(double low, double high);

double Norm2(double *x, int n);

double Parfor_Norm2(double *x, int n);

void Par_Norm2(AllData *all_data,
               double *r,
               int thread_level,
               int ns, int ne);

int SumInt(int *x, int n);

double SumDbl(double *x, int n);

void QuicksortPair_int_double(int *x, double *y, int first, int last);

void BubblesortPair_int_double(int *x, double *y, int n);

int SMEM_LevelBarrier(AllData *all_data,
                      int **barrier_flags,
                      int level);

int CheckConverge(AllData *all_data, int thread_level);

void InitSolve(AllData *all_data);

#endif
