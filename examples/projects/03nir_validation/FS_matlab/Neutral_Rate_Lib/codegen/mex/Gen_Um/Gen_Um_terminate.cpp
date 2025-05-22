//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Um_terminate.cpp
//
// Code generation for function 'Gen_Um_terminate'
//

// Include files
#include "Gen_Um_terminate.h"
#include "Gen_Um_data.h"
#include "_coder_Gen_Um_mex.h"
#include "rt_nonfinite.h"

// Function Definitions
void Gen_Um_atexit()
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

void Gen_Um_terminate()
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

// End of code generation (Gen_Um_terminate.cpp)
