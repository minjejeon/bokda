//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_Gen_Gamma_mex.h
//
// Code generation for function '_coder_Gen_Gamma_mex'
//

#pragma once


// Include files 
#include "rtwtypes.h"
#include "emlrt.h"
#include "mex.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Function Declarations 
void Gen_Gamma_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs, const mxArray *prhs[6]);

MEXFUNCTION_LINKAGE void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray *prhs[]);

emlrtCTX mexFunctionCreateRootTLS();



// End of code generation (_coder_Gen_Gamma_mex.h) 
