//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpost_beta_initialize.cpp
//
// Code generation for function 'lnpost_beta_initialize'
//

// Include files 
#include "lnpost_beta_initialize.h"
#include "_coder_lnpost_beta_mex.h"
#include "lnpost_beta_data.h"
#include "rt_nonfinite.h"

// Function Definitions 
void lnpost_beta_initialize()
{
    emlrtStack st{ nullptr,     // site
nullptr,     // tls
nullptr    // prev
 };
    mex_InitInfAndNan();
    mexFunctionCreateRootTLS();
    emlrtBreakCheckR2012bFlagVar = emlrtGetBreakCheckFlagAddressR2012b();
    st.tls = emlrtRootTLSGlobal;
    emlrtClearAllocCountR2012b(&st, false, 0U, nullptr);
    emlrtEnterRtStackR2012b(&st);
    emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}


// End of code generation (lnpost_beta_initialize.cpp) 
