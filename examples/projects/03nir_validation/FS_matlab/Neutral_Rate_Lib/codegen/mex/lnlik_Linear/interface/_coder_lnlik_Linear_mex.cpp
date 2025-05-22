//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_lnlik_Linear_mex.cpp
//
// Code generation for function '_coder_lnlik_Linear_mex'
//

// Include files
#include "_coder_lnlik_Linear_mex.h"
#include "_coder_lnlik_Linear_api.h"
#include "lnlik_Linear_data.h"
#include "lnlik_Linear_initialize.h"
#include "lnlik_Linear_terminate.h"
#include "rt_nonfinite.h"
#include <stdexcept>

void emlrtExceptionBridge();
void emlrtExceptionBridge()
{
  throw std::runtime_error("");
}
// Function Definitions
void lnlik_Linear_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs,
                              const mxArray *prhs[4])
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  const mxArray *outputs;
  st.tls = emlrtRootTLSGlobal;
  // Check for proper number of arguments.
  if (nrhs != 4) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 4, 4,
                        12, "lnlik_Linear");
  }
  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 12,
                        "lnlik_Linear");
  }
  // Call the function.
  lnlik_Linear_api(prhs, &outputs);
  // Copy over outputs to the caller.
  emlrtReturnArrays(1, &plhs[0], &outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs,
                 const mxArray *prhs[])
{
  mexAtExit(&lnlik_Linear_atexit);
  // Module initialization.
  lnlik_Linear_initialize();
  try {
    emlrtShouldCleanupOnError((emlrtCTX *)emlrtRootTLSGlobal, false);
    // Dispatch the entry-point.
    lnlik_Linear_mexFunction(nlhs, plhs, nrhs, prhs);
    // Module termination.
    lnlik_Linear_terminate();
  } catch (...) {
    emlrtCleanupOnException((emlrtCTX *)emlrtRootTLSGlobal);
    throw;
  }
}

emlrtCTX mexFunctionCreateRootTLS()
{
  emlrtCreateRootTLSR2021a(&emlrtRootTLSGlobal, &emlrtContextGlobal, nullptr, 1,
                           (void *)&emlrtExceptionBridge);
  return emlrtRootTLSGlobal;
}

// End of code generation (_coder_lnlik_Linear_mex.cpp)
