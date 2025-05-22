//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpost_Omega_mexutil.cpp
//
// Code generation for function 'lnpost_Omega_mexutil'
//

// Include files
#include "lnpost_Omega_mexutil.h"
#include "rt_nonfinite.h"

// Function Definitions
const mxArray *emlrt_marshallOut(const real_T u)
{
  const mxArray *m;
  const mxArray *y;
  y = nullptr;
  m = emlrtCreateDoubleScalar(u);
  emlrtAssign(&y, m);
  return y;
}

// End of code generation (lnpost_Omega_mexutil.cpp)
