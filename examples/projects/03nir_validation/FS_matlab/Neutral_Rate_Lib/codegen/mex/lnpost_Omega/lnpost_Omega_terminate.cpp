//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpost_Omega_terminate.cpp
//
// Code generation for function 'lnpost_Omega_terminate'
//

// Include files 
#include "lnpost_Omega_terminate.h"
#include "_coder_lnpost_Omega_mex.h"
#include "lnpost_Omega_data.h"
#include "rt_nonfinite.h"

// Function Definitions 
void lnpost_Omega_atexit()
{
    emlrtStack st{ nullptr,     // site
nullptr,     // tls
nullptr    // prev
 };
    mexFunctionCreateRootTLS();
    st.tls = emlrtRootTLSGlobal;
    emlrtEnterRtStackR2012b(&st);
    emlrtLeaveRtStackR2012b(&st);
    emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
    emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void lnpost_Omega_terminate()
{
    emlrtStack st{ nullptr,     // site
nullptr,     // tls
nullptr    // prev
 };
    st.tls = emlrtRootTLSGlobal;
    emlrtLeaveRtStackR2012b(&st);
    emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}


// End of code generation (lnpost_Omega_terminate.cpp) 
