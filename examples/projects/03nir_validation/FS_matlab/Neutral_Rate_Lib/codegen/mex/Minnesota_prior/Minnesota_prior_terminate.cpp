//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Minnesota_prior_terminate.cpp
//
// Code generation for function 'Minnesota_prior_terminate'
//

// Include files 
#include "Minnesota_prior_terminate.h"
#include "Minnesota_prior_data.h"
#include "_coder_Minnesota_prior_mex.h"
#include "rt_nonfinite.h"

// Function Definitions 
void Minnesota_prior_atexit()
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

void Minnesota_prior_terminate()
{
    emlrtStack st{ nullptr,     // site
nullptr,     // tls
nullptr    // prev
 };
    st.tls = emlrtRootTLSGlobal;
    emlrtLeaveRtStackR2012b(&st);
    emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}


// End of code generation (Minnesota_prior_terminate.cpp) 
