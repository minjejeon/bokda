//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_lnpost_beta_mex.cpp
//
// Code generation for function '_coder_lnpost_beta_mex'
//

// Include files
#include "_coder_lnpost_beta_mex.h"
#include "_coder_lnpost_beta_api.h"
#include "lnpost_beta_data.h"
#include "lnpost_beta_initialize.h"
#include "lnpost_beta_terminate.h"
#include "rt_nonfinite.h"
#include <stdexcept>

void emlrtExceptionBridge();
void emlrtExceptionBridge()
{
  throw std::runtime_error("");
}
// Function Definitions
void lnpost_beta_mexFunction(int32_T nlhs, mxArray *plhs[1], int32_T nrhs,
                             const mxArray *prhs[8])
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  const mxArray *outputs;
  st.tls = emlrtRootTLSGlobal;
  // Check for proper number of arguments.
  if (nrhs != 8) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 8, 4,
                        11, "lnpost_beta");
  }
  if (nlhs > 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 11,
                        "lnpost_beta");
  }
  // Call the function.
  lnpost_beta_api(prhs, &outputs);
  // Copy over outputs to the caller.
  emlrtReturnArrays(1, &plhs[0], &outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs,
                 const mxArray *prhs[])
{
  mexAtExit(&lnpost_beta_atexit);
  // Module initialization.
  lnpost_beta_initialize();
  try {
    emlrtShouldCleanupOnError((emlrtCTX *)emlrtRootTLSGlobal, false);
    // Dispatch the entry-point.
    lnpost_beta_mexFunction(nlhs, plhs, nrhs, prhs);
    // Module termination.
    lnpost_beta_terminate();
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

// End of code generation (_coder_lnpost_beta_mex.cpp)
