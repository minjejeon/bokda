//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// qrsolve.cpp
//
// Code generation for function 'qrsolve'
//

// Include files
#include "qrsolve.h"
#include "eml_int_forloop_overflow_check.h"
#include "lnpost_Omega_data.h"
#include "lnpost_Omega_mexutil.h"
#include "rt_nonfinite.h"
#include "warning.h"
#include "coder_array.h"
#include "lapacke.h"
#include "mwmathutil.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo ud_emlrtRSI{
    61,        // lineNo
    "qrsolve", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo vd_emlrtRSI{
    72,        // lineNo
    "qrsolve", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo wd_emlrtRSI{
    85,        // lineNo
    "qrsolve", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo xd_emlrtRSI{
    63,       // lineNo
    "xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo yd_emlrtRSI{
    98,             // lineNo
    "ceval_xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo ae_emlrtRSI{
    138,            // lineNo
    "ceval_xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo be_emlrtRSI{
    141,            // lineNo
    "ceval_xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo ce_emlrtRSI{
    143,            // lineNo
    "ceval_xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo de_emlrtRSI{
    148,            // lineNo
    "ceval_xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo ee_emlrtRSI{
    151,            // lineNo
    "ceval_xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo fe_emlrtRSI{
    154,            // lineNo
    "ceval_xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo ge_emlrtRSI{
    158,            // lineNo
    "ceval_xgeqp3", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pathName
};

static emlrtRSInfo he_emlrtRSI{
    173,          // lineNo
    "rankFromQR", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo ie_emlrtRSI{
    172,          // lineNo
    "rankFromQR", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo je_emlrtRSI{
    119,         // lineNo
    "LSQFromQR", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo ke_emlrtRSI{
    126,         // lineNo
    "LSQFromQR", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo le_emlrtRSI{
    128,         // lineNo
    "LSQFromQR", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo me_emlrtRSI{
    138,         // lineNo
    "LSQFromQR", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pathName
};

static emlrtRSInfo ne_emlrtRSI{
    31,         // lineNo
    "xunormqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xunormqr.m" // pathName
};

static emlrtRSInfo oe_emlrtRSI{
    102,              // lineNo
    "ceval_xunormqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xunormqr.m" // pathName
};

static emlrtRSInfo pe_emlrtRSI{
    108,              // lineNo
    "ceval_xunormqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xunormqr.m" // pathName
};

static emlrtRSInfo qe_emlrtRSI{
    18,          // lineNo
    "xzunormqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "reflapack\\xzunormqr.m" // pathName
};

static emlrtRSInfo re_emlrtRSI{
    21,          // lineNo
    "xzunormqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "reflapack\\xzunormqr.m" // pathName
};

static emlrtRSInfo se_emlrtRSI{
    23,          // lineNo
    "xzunormqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "reflapack\\xzunormqr.m" // pathName
};

static emlrtRSInfo te_emlrtRSI{
    29,          // lineNo
    "xzunormqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "reflapack\\xzunormqr.m" // pathName
};

static emlrtMCInfo g_emlrtMCI{
    53,        // lineNo
    19,        // colNo
    "flt2str", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\flt2str.m" // pName
};

static emlrtRTEInfo vc_emlrtRTEI{
    1,        // lineNo
    32,       // colNo
    "xgeqp3", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pName
};

static emlrtRTEInfo wc_emlrtRTEI{
    61,       // lineNo
    9,        // colNo
    "xgeqp3", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pName
};

static emlrtRTEInfo xc_emlrtRTEI{
    92,       // lineNo
    22,       // colNo
    "xgeqp3", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pName
};

static emlrtRTEInfo yc_emlrtRTEI{
    105,      // lineNo
    1,        // colNo
    "xgeqp3", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pName
};

static emlrtRTEInfo ad_emlrtRTEI{
    97,       // lineNo
    5,        // colNo
    "xgeqp3", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeqp3.m" // pName
};

static emlrtRTEInfo bd_emlrtRTEI{
    85,        // lineNo
    26,        // colNo
    "qrsolve", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pName
};

static emlrtRTEInfo cd_emlrtRTEI{
    109,       // lineNo
    1,         // colNo
    "qrsolve", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pName
};

static emlrtRTEInfo dd_emlrtRTEI{
    119,       // lineNo
    5,         // colNo
    "qrsolve", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\qrsolve.m" // pName
};

static emlrtRSInfo jf_emlrtRSI{
    53,        // lineNo
    "flt2str", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\flt2str.m" // pathName
};

// Function Declarations
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId, char_T ret[14]);

static const mxArray *b_sprintf(const emlrtStack *sp, const mxArray *b,
                                const mxArray *c, emlrtMCInfo *location);

namespace coder {
namespace internal {
static void LSQFromQR(const emlrtStack *sp, const ::coder::array<real_T, 2U> &A,
                      const ::coder::array<real_T, 1U> &tau,
                      const ::coder::array<int32_T, 2U> &jpvt,
                      ::coder::array<real_T, 2U> &B, int32_T rankA,
                      ::coder::array<real_T, 2U> &Y);

}
} // namespace coder
static void emlrt_marshallIn(const emlrtStack *sp,
                             const mxArray *a__output_of_sprintf_,
                             const char_T *identifier, char_T y[14]);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId, char_T y[14]);

// Function Definitions
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId, char_T ret[14])
{
  static const int32_T dims[2]{1, 14};
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"char",
                          false, 2U, (void *)&dims[0]);
  emlrtImportCharArrayR2015b((emlrtCTX)sp, src, &ret[0], 14);
  emlrtDestroyArray(&src);
}

static const mxArray *b_sprintf(const emlrtStack *sp, const mxArray *b,
                                const mxArray *c, emlrtMCInfo *location)
{
  const mxArray *pArrays[2];
  const mxArray *m;
  pArrays[0] = b;
  pArrays[1] = c;
  return emlrtCallMATLABR2012b((emlrtCTX)sp, 1, &m, 2, &pArrays[0],
                               (const char_T *)"sprintf", true, location);
}

namespace coder {
namespace internal {
static void LSQFromQR(const emlrtStack *sp, const ::coder::array<real_T, 2U> &A,
                      const ::coder::array<real_T, 1U> &tau,
                      const ::coder::array<int32_T, 2U> &jpvt,
                      ::coder::array<real_T, 2U> &B, int32_T rankA,
                      ::coder::array<real_T, 2U> &Y)
{
  static const char_T fname[14]{'L', 'A', 'P', 'A', 'C', 'K', 'E',
                                '_', 'd', 'o', 'r', 'm', 'q', 'r'};
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack st;
  int32_T b_nb;
  int32_T i;
  int32_T info;
  int32_T j;
  int32_T k;
  int32_T nb;
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
  nb = B.size(1);
  Y.set_size(&cd_emlrtRTEI, sp, A.size(1), B.size(1));
  info = A.size(1) * B.size(1);
  for (b_nb = 0; b_nb < info; b_nb++) {
    Y[b_nb] = 0.0;
  }
  st.site = &je_emlrtRSI;
  b_st.site = &ne_emlrtRSI;
  if ((A.size(0) != 0) && (A.size(1) != 0) &&
      ((B.size(0) != 0) && (B.size(1) != 0))) {
    ptrdiff_t nrc_t;
    int32_T a_tmp;
    int32_T mn;
    boolean_T p;
    nrc_t = (ptrdiff_t)B.size(0);
    mn = A.size(0);
    a_tmp = A.size(1);
    nrc_t = LAPACKE_dormqr(102, 'L', 'T', nrc_t, (ptrdiff_t)B.size(1),
                           (ptrdiff_t)muIntScalarMin_sint32(mn, a_tmp),
                           &(((::coder::array<real_T, 2U> *)&A)->data())[0],
                           (ptrdiff_t)A.size(0),
                           &(((::coder::array<real_T, 1U> *)&tau)->data())[0],
                           &(B.data())[0], nrc_t);
    info = (int32_T)nrc_t;
    c_st.site = &oe_emlrtRSI;
    if (info != 0) {
      boolean_T b_p;
      p = true;
      b_p = false;
      if (info == -7) {
        b_p = true;
      } else if (info == -9) {
        b_p = true;
      } else if (info == -10) {
        b_p = true;
      }
      if (!b_p) {
        if (info == -1010) {
          emlrtErrorWithMessageIdR2018a(&c_st, &q_emlrtRTEI, "MATLAB:nomem",
                                        "MATLAB:nomem", 0);
        } else {
          emlrtErrorWithMessageIdR2018a(&c_st, &p_emlrtRTEI,
                                        "Coder:toolbox:LAPACKCallErrorInfo",
                                        "Coder:toolbox:LAPACKCallErrorInfo", 5,
                                        4, 14, &fname[0], 12, info);
        }
      }
    } else {
      p = false;
    }
    if (p) {
      if ((info == -10) && (B.size(1) > 1)) {
        c_st.site = &pe_emlrtRSI;
        info = A.size(0);
        b_nb = B.size(1);
        mn = A.size(0);
        a_tmp = A.size(1);
        mn = muIntScalarMin_sint32(mn, a_tmp);
        d_st.site = &qe_emlrtRSI;
        if (mn > 2147483646) {
          e_st.site = &i_emlrtRSI;
          check_forloop_overflow_error(&e_st);
        }
        for (j = 0; j < mn; j++) {
          if (tau[j] != 0.0) {
            d_st.site = &re_emlrtRSI;
            if (b_nb > 2147483646) {
              e_st.site = &i_emlrtRSI;
              check_forloop_overflow_error(&e_st);
            }
            for (k = 0; k < b_nb; k++) {
              real_T wj;
              wj = B[j + B.size(0) * k];
              a_tmp = j + 2;
              d_st.site = &se_emlrtRSI;
              if ((j + 2 <= info) && (info > 2147483646)) {
                e_st.site = &i_emlrtRSI;
                check_forloop_overflow_error(&e_st);
              }
              for (i = a_tmp; i <= info; i++) {
                wj += A[(i + A.size(0) * j) - 1] * B[(i + B.size(0) * k) - 1];
              }
              wj *= tau[j];
              if (wj != 0.0) {
                B[j + B.size(0) * k] = B[j + B.size(0) * k] - wj;
                d_st.site = &te_emlrtRSI;
                for (i = a_tmp; i <= info; i++) {
                  B[(i + B.size(0) * k) - 1] = B[(i + B.size(0) * k) - 1] -
                                               A[(i + A.size(0) * j) - 1] * wj;
                }
              }
            }
          }
        }
      } else {
        info = B.size(0);
        b_nb = B.size(1);
        B.set_size(&dd_emlrtRTEI, &b_st, info, b_nb);
        info *= b_nb;
        for (b_nb = 0; b_nb < info; b_nb++) {
          B[b_nb] = rtNaN;
        }
      }
    }
  }
  st.site = &ke_emlrtRSI;
  if ((1 <= nb) && (nb > 2147483646)) {
    b_st.site = &i_emlrtRSI;
    check_forloop_overflow_error(&b_st);
  }
  for (k = 0; k < nb; k++) {
    st.site = &le_emlrtRSI;
    if ((1 <= rankA) && (rankA > 2147483646)) {
      b_st.site = &i_emlrtRSI;
      check_forloop_overflow_error(&b_st);
    }
    for (i = 0; i < rankA; i++) {
      Y[(jpvt[i] + Y.size(0) * k) - 1] = B[i + B.size(0) * k];
    }
    for (j = rankA; j >= 1; j--) {
      b_nb = jpvt[j - 1];
      Y[(b_nb + Y.size(0) * k) - 1] =
          Y[(b_nb + Y.size(0) * k) - 1] / A[(j + A.size(0) * (j - 1)) - 1];
      st.site = &me_emlrtRSI;
      for (i = 0; i <= j - 2; i++) {
        Y[(jpvt[i] + Y.size(0) * k) - 1] =
            Y[(jpvt[i] + Y.size(0) * k) - 1] -
            Y[(jpvt[j - 1] + Y.size(0) * k) - 1] * A[i + A.size(0) * (j - 1)];
      }
    }
  }
}

} // namespace internal
} // namespace coder
static void emlrt_marshallIn(const emlrtStack *sp,
                             const mxArray *a__output_of_sprintf_,
                             const char_T *identifier, char_T y[14])
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  emlrt_marshallIn(sp, emlrtAlias(a__output_of_sprintf_), &thisId, y);
  emlrtDestroyArray(&a__output_of_sprintf_);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId, char_T y[14])
{
  b_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

namespace coder {
namespace internal {
void qrsolve(const emlrtStack *sp, const ::coder::array<real_T, 2U> &A,
             const ::coder::array<real_T, 2U> &B, ::coder::array<real_T, 2U> &Y)
{
  static const int32_T iv[2]{1, 6};
  static const char_T fname[14]{'L', 'A', 'P', 'A', 'C', 'K', 'E',
                                '_', 'd', 'g', 'e', 'q', 'p', '3'};
  static const char_T rfmt[6]{'%', '1', '4', '.', '6', 'e'};
  array<ptrdiff_t, 1U> jpvt_t;
  array<real_T, 2U> b_A;
  array<real_T, 2U> b_B;
  array<real_T, 1U> tau;
  array<int32_T, 2U> jpvt;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack st;
  const mxArray *m;
  const mxArray *y;
  real_T tol;
  int32_T b_na;
  int32_T i;
  int32_T ma;
  int32_T minmana;
  int32_T na;
  int32_T rankA;
  char_T str[14];
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  st.site = &ud_emlrtRSI;
  b_A.set_size(&vc_emlrtRTEI, &st, A.size(0), A.size(1));
  na = A.size(0) * A.size(1);
  for (i = 0; i < na; i++) {
    b_A[i] = A[i];
  }
  rankA = b_A.size(0);
  b_na = b_A.size(1);
  jpvt.set_size(&wc_emlrtRTEI, &st, 1, b_A.size(1));
  na = b_A.size(1);
  for (i = 0; i < na; i++) {
    jpvt[i] = 0;
  }
  b_st.site = &xd_emlrtRSI;
  ma = b_A.size(0);
  na = b_A.size(1);
  minmana = muIntScalarMin_sint32(ma, na);
  tau.set_size(&xc_emlrtRTEI, &b_st, minmana);
  if ((b_A.size(0) == 0) || (b_A.size(1) == 0)) {
    tau.set_size(&ad_emlrtRTEI, &b_st, minmana);
    for (i = 0; i < minmana; i++) {
      tau[i] = 0.0;
    }
    c_st.site = &yd_emlrtRSI;
    if ((1 <= b_A.size(1)) && (b_A.size(1) > 2147483646)) {
      d_st.site = &i_emlrtRSI;
      check_forloop_overflow_error(&d_st);
    }
    for (rankA = 0; rankA < b_na; rankA++) {
      jpvt[rankA] = rankA + 1;
    }
  } else {
    boolean_T p;
    jpvt_t.set_size(&yc_emlrtRTEI, &b_st, b_A.size(1));
    na = b_A.size(1);
    for (i = 0; i < na; i++) {
      jpvt_t[i] = (ptrdiff_t)0;
    }
    ptrdiff_t info_t;
    info_t = LAPACKE_dgeqp3(102, (ptrdiff_t)b_A.size(0), (ptrdiff_t)b_A.size(1),
                            &(b_A.data())[0], (ptrdiff_t)b_A.size(0),
                            &(jpvt_t.data())[0], &(tau.data())[0]);
    na = (int32_T)info_t;
    c_st.site = &ae_emlrtRSI;
    if (na != 0) {
      p = true;
      if (na != -4) {
        if (na == -1010) {
          emlrtErrorWithMessageIdR2018a(&c_st, &q_emlrtRTEI, "MATLAB:nomem",
                                        "MATLAB:nomem", 0);
        } else {
          emlrtErrorWithMessageIdR2018a(
              &c_st, &p_emlrtRTEI, "Coder:toolbox:LAPACKCallErrorInfo",
              "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 14, &fname[0], 12, na);
        }
      }
    } else {
      p = false;
    }
    if (p) {
      c_st.site = &be_emlrtRSI;
      if ((1 <= b_na) && (b_na > 2147483646)) {
        d_st.site = &i_emlrtRSI;
        check_forloop_overflow_error(&d_st);
      }
      for (na = 0; na < b_na; na++) {
        c_st.site = &ce_emlrtRSI;
        if ((1 <= rankA) && (rankA > 2147483646)) {
          d_st.site = &i_emlrtRSI;
          check_forloop_overflow_error(&d_st);
        }
        for (i = 0; i < rankA; i++) {
          b_A[na * ma + i] = rtNaN;
        }
      }
      ma = muIntScalarMin_sint32(rankA, b_na);
      c_st.site = &de_emlrtRSI;
      for (rankA = 0; rankA < ma; rankA++) {
        tau[rankA] = rtNaN;
      }
      na = ma + 1;
      c_st.site = &ee_emlrtRSI;
      if ((ma + 1 <= minmana) && (minmana > 2147483646)) {
        d_st.site = &i_emlrtRSI;
        check_forloop_overflow_error(&d_st);
      }
      for (rankA = na; rankA <= minmana; rankA++) {
        tau[rankA - 1] = 0.0;
      }
      c_st.site = &fe_emlrtRSI;
      for (rankA = 0; rankA < b_na; rankA++) {
        jpvt[rankA] = rankA + 1;
      }
    } else {
      c_st.site = &ge_emlrtRSI;
      if ((1 <= b_na) && (b_na > 2147483646)) {
        d_st.site = &i_emlrtRSI;
        check_forloop_overflow_error(&d_st);
      }
      for (rankA = 0; rankA < b_na; rankA++) {
        jpvt[rankA] = (int32_T)jpvt_t[rankA];
      }
    }
  }
  st.site = &vd_emlrtRSI;
  rankA = 0;
  tol = 0.0;
  if (b_A.size(0) < b_A.size(1)) {
    ma = b_A.size(0);
    na = b_A.size(1);
  } else {
    ma = b_A.size(1);
    na = b_A.size(0);
  }
  if (ma > 0) {
    tol = muDoubleScalarMin(1.4901161193847656E-8,
                            2.2204460492503131E-15 * static_cast<real_T>(na)) *
          muDoubleScalarAbs(b_A[0]);
    while ((rankA < ma) &&
           (!(muDoubleScalarAbs(b_A[rankA + b_A.size(0) * rankA]) <= tol))) {
      rankA++;
    }
  }
  if ((rankA < ma) && (!emlrtSetWarningFlag(&st))) {
    b_st.site = &he_emlrtRSI;
    y = nullptr;
    m = emlrtCreateCharArray(2, &iv[0]);
    emlrtInitCharArrayR2013a(&b_st, 6, m, &rfmt[0]);
    emlrtAssign(&y, m);
    c_st.site = &jf_emlrtRSI;
    emlrt_marshallIn(&c_st,
                     b_sprintf(&c_st, y, emlrt_marshallOut(tol), &g_emlrtMCI),
                     "<output of sprintf>", str);
    b_st.site = &ie_emlrtRSI;
    warning(&b_st, rankA, str);
  }
  b_B.set_size(&bd_emlrtRTEI, sp, B.size(0), B.size(1));
  na = B.size(0) * B.size(1) - 1;
  for (i = 0; i <= na; i++) {
    b_B[i] = B[i];
  }
  st.site = &wd_emlrtRSI;
  LSQFromQR(&st, b_A, tau, jpvt, b_B, rankA, Y);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

} // namespace internal
} // namespace coder

// End of code generation (qrsolve.cpp)
