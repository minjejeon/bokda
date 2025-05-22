//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnlik_Linear_terminate.cpp
//
// Code generation for function 'lnlik_Linear_terminate'
//

// Include files
#include "lnlik_Linear_terminate.h"
#include "_coder_lnlik_Linear_mex.h"
#include "lnlik_Linear_data.h"
#include "rt_nonfinite.h"

// Function Definitions
void lnlik_Linear_atexit()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void lnlik_Linear_terminate()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  st.tls = emlrtRootTLSGlobal;
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

// End of code generation (lnlik_Linear_terminate.cpp)
