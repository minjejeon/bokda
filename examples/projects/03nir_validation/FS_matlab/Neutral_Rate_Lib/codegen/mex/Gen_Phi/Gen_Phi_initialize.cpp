//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Phi_initialize.cpp
//
// Code generation for function 'Gen_Phi_initialize'
//

// Include files
#include "Gen_Phi_initialize.h"
#include "Gen_Phi_data.h"
#include "_coder_Gen_Phi_mex.h"
#include "rt_nonfinite.h"

// Function Definitions
void Gen_Phi_initialize()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtBreakCheckR2012bFlagVar = emlrtGetBreakCheckFlagAddressR2012b();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, nullptr);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

// End of code generation (Gen_Phi_initialize.cpp)
