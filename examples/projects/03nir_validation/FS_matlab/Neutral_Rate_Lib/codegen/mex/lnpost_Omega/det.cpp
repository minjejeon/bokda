//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// det.cpp
//
// Code generation for function 'det'
//

// Include files
#include "det.h"
#include "lnpost_Omega_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "lapacke.h"
#include "mwmathutil.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo cd_emlrtRSI{
    21,    // lineNo
    "det", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\det.m" // pathName
};

static emlrtRTEInfo r_emlrtRTEI{
    12,    // lineNo
    15,    // colNo
    "det", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\det.m" // pName
};

// Function Definitions
namespace coder {
real_T det(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x)
{
  array<ptrdiff_t, 1U> ipiv_t;
  array<ptrdiff_t, 1U> r;
  array<real_T, 2U> b_x;
  array<int32_T, 2U> ipiv;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T y;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  if (x.size(0) != x.size(1)) {
    emlrtErrorWithMessageIdR2018a(sp, &r_emlrtRTEI, "Coder:MATLAB:square",
                                  "Coder:MATLAB:square", 0);
  }
  if ((x.size(0) == 0) || (x.size(1) == 0)) {
    y = 1.0;
  } else {
    ptrdiff_t info_t;
    int32_T i;
    int32_T loop_ub;
    int32_T m;
    int32_T n;
    boolean_T isodd;
    m = x.size(0);
    n = x.size(1);
    st.site = &cd_emlrtRSI;
    b_x.set_size(&mc_emlrtRTEI, &st, x.size(0), x.size(1));
    loop_ub = x.size(0) * x.size(1);
    for (i = 0; i < loop_ub; i++) {
      b_x[i] = x[i];
    }
    b_st.site = &dd_emlrtRSI;
    c_st.site = &fd_emlrtRSI;
    info_t = (ptrdiff_t)0.0;
    m = muIntScalarMin_sint32(m, n);
    r.set_size(&pc_emlrtRTEI, &c_st, m);
    for (i = 0; i < m; i++) {
      r[i] = info_t;
    }
    ipiv_t.set_size(&nc_emlrtRTEI, &b_st, r.size(0));
    info_t = LAPACKE_dgetrf_work(102, (ptrdiff_t)x.size(0),
                                 (ptrdiff_t)x.size(1), &(b_x.data())[0],
                                 (ptrdiff_t)x.size(0), &(ipiv_t.data())[0]);
    m = (int32_T)info_t;
    ipiv.set_size(&oc_emlrtRTEI, &b_st, 1, ipiv_t.size(0));
    c_st.site = &ed_emlrtRSI;
    if (m < 0) {
      if (m == -1010) {
        emlrtErrorWithMessageIdR2018a(&c_st, &q_emlrtRTEI, "MATLAB:nomem",
                                      "MATLAB:nomem", 0);
      } else {
        emlrtErrorWithMessageIdR2018a(
            &c_st, &p_emlrtRTEI, "Coder:toolbox:LAPACKCallErrorInfo",
            "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 19, &cv[0], 12, m);
      }
    }
    i = ipiv_t.size(0) - 1;
    for (m = 0; m <= i; m++) {
      ipiv[m] = (int32_T)ipiv_t[m];
    }
    y = b_x[0];
    i = b_x.size(0);
    for (m = 0; m <= i - 2; m++) {
      y *= b_x[(m + b_x.size(0) * (m + 1)) + 1];
    }
    isodd = false;
    i = ipiv.size(1);
    for (m = 0; m <= i - 2; m++) {
      if (ipiv[m] > m + 1) {
        isodd = !isodd;
      }
    }
    if (isodd) {
      y = -y;
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
  return y;
}

} // namespace coder

// End of code generation (det.cpp)
