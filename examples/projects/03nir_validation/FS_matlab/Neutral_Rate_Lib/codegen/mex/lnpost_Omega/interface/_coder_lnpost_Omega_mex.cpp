//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_lnpost_Omega_mex.cpp
//
// Code generation for function '_coder_lnpost_Omega_mex'
//

// Include files
#include "_coder_lnpost_Omega_mex.h"
#include "_coder_lnpost_Omega_api.h"
#include "lnpost_Omega.h"
#include "lnpost_Omega_data.h"
#include "lnpost_Omega_initialize.h"
#include "lnpost_Omega_terminate.h"
#include "rt_nonfinite.h"
#include <stdexcept>

void emlrtExceptionBridge();
void emlrtExceptionBridge()
{
  throw std::runtime_error("");
}
// Function Definitions
void lnpost_Omega_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs,
                              const mxArray *prhs[6])
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  const mxArray *outputs;
  st.tls = emlrtRootTLSGlobal;
  // Check for proper number of arguments.
  if (nrhs != 6) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 6, 4,
                        12, "lnpost_Omega");
  }
  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 12,
                        "lnpost_Omega");
  }
  // Call the function.
  lnpost_Omega_api(prhs, &outputs);
  // Copy over outputs to the caller.
  emlrtReturnArrays(1, &plhs[0], &outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs,
                 const mxArray *prhs[])
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mexAtExit(&lnpost_Omega_atexit);
  emlrtLoadLibrary(
      (const char_T *)"C:\\ProgramData\\MATLAB\\SupportPackages\\R2021a\\3P."
                      "instrset\\mingw_w64.instrset\\bin\\libgomp-1.dll");
  // Initialize the memory manager.
  omp_init_lock(&emlrtLockGlobal);
  omp_init_nest_lock(&emlrtNestLockGlobal);
  // Module initialization.
  lnpost_Omega_initialize();
  st.tls = emlrtRootTLSGlobal;
  try {
    emlrtShouldCleanupOnError((emlrtCTX *)emlrtRootTLSGlobal, false);
    // Dispatch the entry-point.
    lnpost_Omega_mexFunction(nlhs, plhs, nrhs, prhs);
    // Module termination.
    lnpost_Omega_terminate();
    omp_destroy_lock(&emlrtLockGlobal);
    omp_destroy_nest_lock(&emlrtNestLockGlobal);
  } catch (...) {
    omp_destroy_lock(&emlrtLockGlobal);
    omp_destroy_nest_lock(&emlrtNestLockGlobal);
    emlrtReportParallelRunTimeError(&st);
    emlrtCleanupOnException((emlrtCTX *)emlrtRootTLSGlobal);
    throw;
  }
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLSR2021a(&emlrtRootTLSGlobal, &emlrtContextGlobal,
                           &emlrtLockerFunction, omp_get_num_procs(),
                           (void *)&emlrtExceptionBridge);
  return emlrtRootTLSGlobal;
}

// End of code generation (_coder_lnpost_Omega_mex.cpp)
