//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// mrdivide_helper.cpp
//
// Code generation for function 'mrdivide_helper'
//

// Include files
#include "mrdivide_helper.h"
#include "eml_int_forloop_overflow_check.h"
#include "lnpost_Omega_data.h"
#include "qrsolve.h"
#include "rt_nonfinite.h"
#include "warning.h"
#include "blas.h"
#include "coder_array.h"
#include "lapacke.h"
#include "mwmathutil.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo id_emlrtRSI{
    42,      // lineNo
    "mrdiv", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_"
    "helper.m" // pathName
};

static emlrtRSInfo jd_emlrtRSI{
    44,      // lineNo
    "mrdiv", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_"
    "helper.m" // pathName
};

static emlrtRSInfo kd_emlrtRSI{
    67,        // lineNo
    "lusolve", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\lusolve.m" // pathName
};

static emlrtRSInfo ld_emlrtRSI{
    112,          // lineNo
    "lusolveNxN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\lusolve.m" // pathName
};

static emlrtRSInfo md_emlrtRSI{
    107,          // lineNo
    "lusolveNxN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\lusolve.m" // pathName
};

static emlrtRSInfo nd_emlrtRSI{
    135,          // lineNo
    "XtimesInvA", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\lusolve.m" // pathName
};

static emlrtRSInfo od_emlrtRSI{
    140,          // lineNo
    "XtimesInvA", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\lusolve.m" // pathName
};

static emlrtRSInfo pd_emlrtRSI{
    142,          // lineNo
    "XtimesInvA", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\lusolve.m" // pathName
};

static emlrtRSInfo qd_emlrtRSI{
    147,          // lineNo
    "XtimesInvA", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\lusolve.m" // pathName
};

static emlrtRSInfo rd_emlrtRSI{
    67,      // lineNo
    "xtrsm", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+blas\\xtrsm."
    "m" // pathName
};

static emlrtRSInfo td_emlrtRSI{
    90,              // lineNo
    "warn_singular", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\lusolve.m" // pathName
};

static emlrtRTEInfo qc_emlrtRTEI{
    44,                // lineNo
    32,                // colNo
    "mrdivide_helper", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_"
    "helper.m" // pName
};

static emlrtRTEInfo rc_emlrtRTEI{
    44,                // lineNo
    35,                // colNo
    "mrdivide_helper", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_"
    "helper.m" // pName
};

static emlrtRTEInfo sc_emlrtRTEI{
    44,                // lineNo
    5,                 // colNo
    "mrdivide_helper", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_"
    "helper.m" // pName
};

static emlrtRTEInfo tc_emlrtRTEI{
    42,                // lineNo
    5,                 // colNo
    "mrdivide_helper", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_"
    "helper.m" // pName
};

static emlrtRTEInfo uc_emlrtRTEI{
    31,                // lineNo
    5,                 // colNo
    "mrdivide_helper", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_"
    "helper.m" // pName
};

// Function Definitions
namespace coder {
namespace internal {
void mrdiv(const emlrtStack *sp, const ::coder::array<real_T, 2U> &A,
           const ::coder::array<real_T, 2U> &B, ::coder::array<real_T, 2U> &Y)
{
  ptrdiff_t info_t;
  ptrdiff_t lda_t;
  ptrdiff_t ldb_t;
  ptrdiff_t n_t;
  array<ptrdiff_t, 1U> ipiv_t;
  array<ptrdiff_t, 1U> r;
  array<real_T, 2U> b_A;
  array<real_T, 2U> b_B;
  array<real_T, 2U> c_A;
  array<int32_T, 2U> ipiv;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack st;
  real_T temp;
  char_T DIAGA1;
  char_T SIDE1;
  char_T TRANSA1;
  char_T UPLO1;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  e_st.prev = &d_st;
  e_st.tls = d_st.tls;
  f_st.prev = &e_st;
  f_st.tls = e_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  if ((A.size(0) == 0) || (A.size(1) == 0) ||
      ((B.size(0) == 0) || (B.size(1) == 0))) {
    int32_T loop_ub;
    Y.set_size(&uc_emlrtRTEI, sp, A.size(0), B.size(0));
    loop_ub = A.size(0) * B.size(0);
    for (int32_T i{0}; i < loop_ub; i++) {
      Y[i] = 0.0;
    }
  } else if (B.size(0) == B.size(1)) {
    int32_T i;
    int32_T loop_ub;
    int32_T n;
    st.site = &id_emlrtRSI;
    b_st.site = &kd_emlrtRSI;
    Y.set_size(&tc_emlrtRTEI, &b_st, A.size(0), A.size(1));
    loop_ub = A.size(0) * A.size(1);
    for (i = 0; i < loop_ub; i++) {
      Y[i] = A[i];
    }
    c_st.site = &md_emlrtRSI;
    n = B.size(1);
    d_st.site = &nd_emlrtRSI;
    c_A.set_size(&mc_emlrtRTEI, &d_st, B.size(0), B.size(1));
    loop_ub = B.size(0) * B.size(1);
    for (i = 0; i < loop_ub; i++) {
      c_A[i] = B[i];
    }
    e_st.site = &dd_emlrtRSI;
    f_st.site = &fd_emlrtRSI;
    info_t = (ptrdiff_t)0.0;
    n = muIntScalarMin_sint32(n, n);
    r.set_size(&pc_emlrtRTEI, &f_st, n);
    for (i = 0; i < n; i++) {
      r[i] = info_t;
    }
    ipiv_t.set_size(&nc_emlrtRTEI, &e_st, r.size(0));
    info_t = LAPACKE_dgetrf_work(102, (ptrdiff_t)B.size(1),
                                 (ptrdiff_t)B.size(1), &(c_A.data())[0],
                                 (ptrdiff_t)B.size(1), &(ipiv_t.data())[0]);
    loop_ub = (int32_T)info_t;
    ipiv.set_size(&oc_emlrtRTEI, &e_st, 1, ipiv_t.size(0));
    f_st.site = &ed_emlrtRSI;
    if (loop_ub < 0) {
      if (loop_ub == -1010) {
        emlrtErrorWithMessageIdR2018a(&f_st, &q_emlrtRTEI, "MATLAB:nomem",
                                      "MATLAB:nomem", 0);
      } else {
        emlrtErrorWithMessageIdR2018a(
            &f_st, &p_emlrtRTEI, "Coder:toolbox:LAPACKCallErrorInfo",
            "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 19, &cv[0], 12, loop_ub);
      }
    }
    i = ipiv_t.size(0) - 1;
    for (n = 0; n <= i; n++) {
      ipiv[n] = (int32_T)ipiv_t[n];
    }
    n = Y.size(0);
    d_st.site = &od_emlrtRSI;
    e_st.site = &rd_emlrtRSI;
    temp = 1.0;
    DIAGA1 = 'N';
    TRANSA1 = 'N';
    UPLO1 = 'U';
    SIDE1 = 'R';
    info_t = (ptrdiff_t)Y.size(0);
    n_t = (ptrdiff_t)B.size(1);
    lda_t = (ptrdiff_t)B.size(1);
    ldb_t = (ptrdiff_t)Y.size(0);
    dtrsm(&SIDE1, &UPLO1, &TRANSA1, &DIAGA1, &info_t, &n_t, &temp,
          &(c_A.data())[0], &lda_t, &(Y.data())[0], &ldb_t);
    d_st.site = &pd_emlrtRSI;
    e_st.site = &rd_emlrtRSI;
    temp = 1.0;
    DIAGA1 = 'U';
    TRANSA1 = 'N';
    UPLO1 = 'L';
    SIDE1 = 'R';
    info_t = (ptrdiff_t)n;
    n_t = (ptrdiff_t)B.size(1);
    lda_t = (ptrdiff_t)B.size(1);
    ldb_t = (ptrdiff_t)n;
    dtrsm(&SIDE1, &UPLO1, &TRANSA1, &DIAGA1, &info_t, &n_t, &temp,
          &(c_A.data())[0], &lda_t, &(Y.data())[0], &ldb_t);
    i = B.size(1) - 1;
    for (int32_T j{i}; j >= 1; j--) {
      int32_T i1;
      i1 = ipiv[j - 1];
      if (i1 != j) {
        d_st.site = &qd_emlrtRSI;
        if (n > 2147483646) {
          e_st.site = &i_emlrtRSI;
          check_forloop_overflow_error(&e_st);
        }
        for (int32_T b_i{0}; b_i < n; b_i++) {
          temp = Y[b_i + Y.size(0) * (j - 1)];
          Y[b_i + Y.size(0) * (j - 1)] = Y[b_i + Y.size(0) * (i1 - 1)];
          Y[b_i + Y.size(0) * (i1 - 1)] = temp;
        }
      }
    }
    if (((B.size(0) != 1) || (B.size(1) != 1)) && (loop_ub > 0)) {
      c_st.site = &ld_emlrtRSI;
      if (!emlrtSetWarningFlag(&c_st)) {
        d_st.site = &td_emlrtRSI;
        warning(&d_st);
      }
    }
  } else {
    int32_T i;
    int32_T i1;
    int32_T loop_ub;
    int32_T n;
    b_B.set_size(&qc_emlrtRTEI, sp, B.size(1), B.size(0));
    loop_ub = B.size(0);
    for (i = 0; i < loop_ub; i++) {
      n = B.size(1);
      for (i1 = 0; i1 < n; i1++) {
        b_B[i1 + b_B.size(0) * i] = B[i + B.size(0) * i1];
      }
    }
    b_A.set_size(&rc_emlrtRTEI, sp, A.size(1), A.size(0));
    loop_ub = A.size(0);
    for (i = 0; i < loop_ub; i++) {
      n = A.size(1);
      for (i1 = 0; i1 < n; i1++) {
        b_A[i1 + b_A.size(0) * i] = A[i + A.size(0) * i1];
      }
    }
    st.site = &jd_emlrtRSI;
    qrsolve(&st, b_B, b_A, c_A);
    Y.set_size(&sc_emlrtRTEI, sp, c_A.size(1), c_A.size(0));
    loop_ub = c_A.size(0);
    for (i = 0; i < loop_ub; i++) {
      n = c_A.size(1);
      for (i1 = 0; i1 < n; i1++) {
        Y[i1 + Y.size(0) * i] = c_A[i + c_A.size(0) * i1];
      }
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

} // namespace internal
} // namespace coder

// End of code generation (mrdivide_helper.cpp)
