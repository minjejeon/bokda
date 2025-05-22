//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// chol.cpp
//
// Code generation for function 'chol'
//

// Include files
#include "chol.h"
#include "Gen_Mu_data.h"
#include "eml_int_forloop_overflow_check.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "lapacke.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo hc_emlrtRSI{
    74,         // lineNo
    "cholesky", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\chol.m" // pathName
};

static emlrtRSInfo ic_emlrtRSI{
    91,         // lineNo
    "cholesky", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\chol.m" // pathName
};

static emlrtRSInfo jc_emlrtRSI{
    92,         // lineNo
    "cholesky", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\chol.m" // pathName
};

static emlrtRSInfo kc_emlrtRSI{
    79,             // lineNo
    "ceval_xpotrf", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xpotrf.m" // pathName
};

static emlrtRSInfo lc_emlrtRSI{
    13,       // lineNo
    "xpotrf", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xpotrf.m" // pathName
};

static emlrtRTEInfo t_emlrtRTEI{
    54,         // lineNo
    15,         // colNo
    "cholesky", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\chol.m" // pName
};

static emlrtRTEInfo u_emlrtRTEI{
    80,         // lineNo
    23,         // colNo
    "cholesky", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\chol.m" // pName
};

// Function Definitions
namespace coder {
void cholesky(const emlrtStack *sp, ::coder::array<real_T, 2U> &A)
{
  static const char_T fname[19]{'L', 'A', 'P', 'A', 'C', 'K', 'E',
                                '_', 'd', 'p', 'o', 't', 'r', 'f',
                                '_', 'w', 'o', 'r', 'k'};
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  int32_T jmax;
  int32_T n;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  n = A.size(1);
  if (A.size(0) != A.size(1)) {
    emlrtErrorWithMessageIdR2018a(sp, &t_emlrtRTEI, "Coder:MATLAB:square",
                                  "Coder:MATLAB:square", 0);
  }
  if (A.size(1) != 0) {
    ptrdiff_t info_t;
    int32_T info;
    st.site = &hc_emlrtRSI;
    b_st.site = &lc_emlrtRSI;
    info_t = LAPACKE_dpotrf_work(102, 'U', (ptrdiff_t)A.size(1), &(A.data())[0],
                                 (ptrdiff_t)A.size(1));
    info = (int32_T)info_t;
    c_st.site = &kc_emlrtRSI;
    if (info < 0) {
      if (info == -1010) {
        emlrtErrorWithMessageIdR2018a(&c_st, &g_emlrtRTEI, "MATLAB:nomem",
                                      "MATLAB:nomem", 0);
      } else {
        emlrtErrorWithMessageIdR2018a(
            &c_st, &h_emlrtRTEI, "Coder:toolbox:LAPACKCallErrorInfo",
            "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 19, &fname[0], 12, info);
      }
    }
    if (info == 0) {
      jmax = n;
    } else {
      emlrtErrorWithMessageIdR2018a(sp, &u_emlrtRTEI, "Coder:MATLAB:posdef",
                                    "Coder:MATLAB:posdef", 0);
    }
    st.site = &ic_emlrtRSI;
    if ((1 <= jmax) && (jmax > 2147483646)) {
      b_st.site = &cb_emlrtRSI;
      check_forloop_overflow_error(&b_st);
    }
    for (info = 0; info < jmax; info++) {
      n = info + 2;
      st.site = &jc_emlrtRSI;
      for (int32_T i{n}; i <= jmax; i++) {
        A[(i + A.size(0) * info) - 1] = 0.0;
      }
    }
  }
}

} // namespace coder

// End of code generation (chol.cpp)
