//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// eig.cpp
//
// Code generation for function 'eig'
//

// Include files
#include "eig.h"
#include "Gen_Phi_data.h"
#include "anyNonFinite.h"
#include "eml_int_forloop_overflow_check.h"
#include "rt_nonfinite.h"
#include "warning.h"
#include "coder_array.h"
#include "lapacke.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo id_emlrtRSI{
    93,    // lineNo
    "eig", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\eig.m" // pathName
};

static emlrtRSInfo jd_emlrtRSI{
    139,   // lineNo
    "eig", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\eig.m" // pathName
};

static emlrtRSInfo kd_emlrtRSI{
    147,   // lineNo
    "eig", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\eig.m" // pathName
};

static emlrtRSInfo od_emlrtRSI{
    21,                     // lineNo
    "eigHermitianStandard", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eigHerm"
    "itianStandard.m" // pathName
};

static emlrtRSInfo pd_emlrtRSI{
    22,                     // lineNo
    "eigHermitianStandard", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eigHerm"
    "itianStandard.m" // pathName
};

static emlrtRSInfo qd_emlrtRSI{
    35,      // lineNo
    "schur", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\schur.m" // pathName
};

static emlrtRSInfo rd_emlrtRSI{
    43,      // lineNo
    "schur", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\schur.m" // pathName
};

static emlrtRSInfo sd_emlrtRSI{
    52,      // lineNo
    "schur", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\schur.m" // pathName
};

static emlrtRSInfo td_emlrtRSI{
    54,      // lineNo
    "schur", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\schur.m" // pathName
};

static emlrtRSInfo ud_emlrtRSI{
    83,      // lineNo
    "schur", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\schur.m" // pathName
};

static emlrtRSInfo vd_emlrtRSI{
    48,     // lineNo
    "triu", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\triu.m" // pathName
};

static emlrtRSInfo wd_emlrtRSI{
    47,     // lineNo
    "triu", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\triu.m" // pathName
};

static emlrtRSInfo xd_emlrtRSI{
    15,       // lineNo
    "xgehrd", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgehrd.m" // pathName
};

static emlrtRSInfo yd_emlrtRSI{
    85,             // lineNo
    "ceval_xgehrd", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgehrd.m" // pathName
};

static emlrtRSInfo ae_emlrtRSI{
    28,       // lineNo
    "xhseqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xhseqr.m" // pathName
};

static emlrtRSInfo be_emlrtRSI{
    128,            // lineNo
    "ceval_xhseqr", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xhseqr.m" // pathName
};

static emlrtRSInfo ce_emlrtRSI{
    47,                 // lineNo
    "mainDiagZeroImag", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eigHerm"
    "itianStandard.m" // pathName
};

static emlrtRSInfo de_emlrtRSI{
    59,            // lineNo
    "eigStandard", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eigStan"
    "dard.m" // pathName
};

static emlrtRSInfo ee_emlrtRSI{
    44,            // lineNo
    "eigStandard", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eigStan"
    "dard.m" // pathName
};

static emlrtRSInfo fe_emlrtRSI{
    38,      // lineNo
    "xgeev", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeev.m" // pathName
};

static emlrtRSInfo ge_emlrtRSI{
    148,           // lineNo
    "ceval_xgeev", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeev.m" // pathName
};

static emlrtRSInfo he_emlrtRSI{
    143,           // lineNo
    "ceval_xgeev", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeev.m" // pathName
};

static emlrtRTEInfo q_emlrtRTEI{
    47,          // lineNo
    13,          // colNo
    "infocheck", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\infocheck.m" // pName
};

static emlrtRTEInfo r_emlrtRTEI{
    44,          // lineNo
    13,          // colNo
    "infocheck", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\infocheck.m" // pName
};

static emlrtRTEInfo s_emlrtRTEI{
    18,      // lineNo
    15,      // colNo
    "schur", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\schur.m" // pName
};

static emlrtRTEInfo t_emlrtRTEI{
    62,    // lineNo
    27,    // colNo
    "eig", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\eig.m" // pName
};

static emlrtRTEInfo rc_emlrtRTEI{
    78,    // lineNo
    24,    // colNo
    "eig", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\eig.m" // pName
};

static emlrtRTEInfo sc_emlrtRTEI{
    38,      // lineNo
    33,      // colNo
    "xgeev", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeev.m" // pName
};

static emlrtRTEInfo tc_emlrtRTEI{
    83,      // lineNo
    24,      // colNo
    "xgeev", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeev.m" // pName
};

static emlrtRTEInfo uc_emlrtRTEI{
    86,      // lineNo
    21,      // colNo
    "xgeev", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeev.m" // pName
};

static emlrtRTEInfo vc_emlrtRTEI{
    115,     // lineNo
    29,      // colNo
    "xgeev", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeev.m" // pName
};

static emlrtRTEInfo wc_emlrtRTEI{
    116,     // lineNo
    29,      // colNo
    "xgeev", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgeev.m" // pName
};

static emlrtRTEInfo xc_emlrtRTEI{
    21,                     // lineNo
    19,                     // colNo
    "eigHermitianStandard", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eigHerm"
    "itianStandard.m" // pName
};

static emlrtRTEInfo yc_emlrtRTEI{
    76,       // lineNo
    22,       // colNo
    "xgehrd", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgehrd.m" // pName
};

static emlrtRTEInfo ad_emlrtRTEI{
    86,       // lineNo
    9,        // colNo
    "xgehrd", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgehrd.m" // pName
};

static emlrtRTEInfo bd_emlrtRTEI{
    111,      // lineNo
    29,       // colNo
    "xhseqr", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xhseqr.m" // pName
};

static emlrtRTEInfo cd_emlrtRTEI{
    112,      // lineNo
    29,       // colNo
    "xhseqr", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xhseqr.m" // pName
};

static emlrtRTEInfo dd_emlrtRTEI{
    129,      // lineNo
    9,        // colNo
    "xhseqr", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xhseqr.m" // pName
};

static emlrtRTEInfo ed_emlrtRTEI{
    21,                     // lineNo
    9,                      // colNo
    "eigHermitianStandard", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eigHerm"
    "itianStandard.m" // pName
};

static emlrtRTEInfo fd_emlrtRTEI{
    46,                     // lineNo
    20,                     // colNo
    "eigHermitianStandard", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eigHerm"
    "itianStandard.m" // pName
};

static emlrtRTEInfo gd_emlrtRTEI{
    139,   // lineNo
    9,     // colNo
    "eig", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\eig.m" // pName
};

static emlrtRTEInfo hd_emlrtRTEI{
    110,   // lineNo
    9,     // colNo
    "eig", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\eig.m" // pName
};

// Function Definitions
namespace coder {
void eig(const emlrtStack *sp, const ::coder::array<real_T, 2U> &A,
         ::coder::array<creal_T, 1U> &V)
{
  static const char_T b_fname[14]{'L', 'A', 'P', 'A', 'C', 'K', 'E',
                                  '_', 'd', 'g', 'e', 'h', 'r', 'd'};
  static const char_T c_fname[14]{'L', 'A', 'P', 'A', 'C', 'K', 'E',
                                  '_', 'd', 'h', 's', 'e', 'q', 'r'};
  static const char_T fname[14]{'L', 'A', 'P', 'A', 'C', 'K', 'E',
                                '_', 'd', 'g', 'e', 'e', 'v', 'x'};
  ptrdiff_t ihi_t;
  ptrdiff_t ilo_t;
  array<real_T, 2U> T;
  array<real_T, 2U> wi;
  array<real_T, 2U> wr;
  array<real_T, 1U> scale;
  array<real_T, 1U> wimag;
  array<real_T, 1U> wreal;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack st;
  real_T abnrm;
  real_T rconde;
  real_T rcondv;
  real_T vleft;
  real_T vright;
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
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  if (A.size(0) != A.size(1)) {
    emlrtErrorWithMessageIdR2018a(sp, &t_emlrtRTEI,
                                  "MATLAB:eig:inputMustBeSquareStandard",
                                  "MATLAB:eig:inputMustBeSquareStandard", 0);
  }
  V.set_size(&rc_emlrtRTEI, sp, A.size(0));
  if ((A.size(0) != 0) && (A.size(1) != 0)) {
    st.site = &id_emlrtRSI;
    if (internal::anyNonFinite(&st, A)) {
      int32_T n;
      V.set_size(&hd_emlrtRTEI, sp, A.size(0));
      n = A.size(0);
      for (int32_T m{0}; m < n; m++) {
        V[m].re = rtNaN;
        V[m].im = 0.0;
      }
    } else {
      int32_T i;
      int32_T j;
      boolean_T p;
      p = (A.size(0) == A.size(1));
      if (p) {
        boolean_T exitg2;
        j = 0;
        exitg2 = false;
        while ((!exitg2) && (j <= A.size(1) - 1)) {
          int32_T exitg1;
          i = 0;
          do {
            exitg1 = 0;
            if (i <= j) {
              if (!(A[i + A.size(0) * j] == A[j + A.size(0) * i])) {
                p = false;
                exitg1 = 1;
              } else {
                i++;
              }
            } else {
              j++;
              exitg1 = 2;
            }
          } while (exitg1 == 0);
          if (exitg1 == 1) {
            exitg2 = true;
          }
        }
      }
      if (p) {
        int32_T m;
        int32_T n;
        st.site = &jd_emlrtRSI;
        b_st.site = &od_emlrtRSI;
        T.set_size(&xc_emlrtRTEI, &b_st, A.size(0), A.size(1));
        n = A.size(0) * A.size(1);
        for (m = 0; m < n; m++) {
          T[m] = A[m];
        }
        if (T.size(0) != T.size(1)) {
          emlrtErrorWithMessageIdR2018a(&b_st, &s_emlrtRTEI,
                                        "Coder:MATLAB:square",
                                        "Coder:MATLAB:square", 0);
        }
        c_st.site = &qd_emlrtRSI;
        if (internal::anyNonFinite(&c_st, T)) {
          uint32_T unnamed_idx_0;
          uint32_T unnamed_idx_1;
          unnamed_idx_0 = static_cast<uint32_T>(T.size(0));
          unnamed_idx_1 = static_cast<uint32_T>(T.size(1));
          T.set_size(&ed_emlrtRTEI, &b_st, static_cast<int32_T>(unnamed_idx_0),
                     static_cast<int32_T>(unnamed_idx_1));
          n = static_cast<int32_T>(unnamed_idx_0) *
              static_cast<int32_T>(unnamed_idx_1);
          for (m = 0; m < n; m++) {
            T[m] = rtNaN;
          }
          c_st.site = &rd_emlrtRSI;
          m = static_cast<int32_T>(unnamed_idx_0);
          if (1 < static_cast<int32_T>(unnamed_idx_0)) {
            int32_T jend;
            n = 2;
            if (static_cast<int32_T>(unnamed_idx_0) - 2 <
                static_cast<int32_T>(unnamed_idx_1) - 1) {
              jend = static_cast<int32_T>(unnamed_idx_0) - 1;
            } else {
              jend = static_cast<int32_T>(unnamed_idx_1);
            }
            d_st.site = &wd_emlrtRSI;
            if ((1 <= jend) && (jend > 2147483646)) {
              e_st.site = &u_emlrtRSI;
              check_forloop_overflow_error(&e_st);
            }
            for (j = 0; j < jend; j++) {
              d_st.site = &vd_emlrtRSI;
              if ((n <= static_cast<int32_T>(unnamed_idx_0)) &&
                  (static_cast<int32_T>(unnamed_idx_0) > 2147483646)) {
                e_st.site = &u_emlrtRSI;
                check_forloop_overflow_error(&e_st);
              }
              for (i = n; i <= m; i++) {
                T[(i + T.size(0) * j) - 1] = 0.0;
              }
              n++;
            }
          }
        } else {
          ptrdiff_t info_t;
          int32_T jend;
          c_st.site = &sd_emlrtRSI;
          d_st.site = &xd_emlrtRSI;
          scale.set_size(&yc_emlrtRTEI, &d_st, T.size(0) - 1);
          if (T.size(0) > 1) {
            info_t = LAPACKE_dgehrd(102, (ptrdiff_t)T.size(0), (ptrdiff_t)1,
                                    (ptrdiff_t)T.size(0), &(T.data())[0],
                                    (ptrdiff_t)T.size(0), &(scale.data())[0]);
            jend = (int32_T)info_t;
            e_st.site = &yd_emlrtRSI;
            if (jend != 0) {
              p = true;
              if (jend != -5) {
                if (jend == -1010) {
                  emlrtErrorWithMessageIdR2018a(
                      &e_st, &r_emlrtRTEI, "MATLAB:nomem", "MATLAB:nomem", 0);
                } else {
                  emlrtErrorWithMessageIdR2018a(
                      &e_st, &q_emlrtRTEI, "Coder:toolbox:LAPACKCallErrorInfo",
                      "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 14,
                      &b_fname[0], 12, jend);
                }
              }
            } else {
              p = false;
            }
            if (p) {
              n = T.size(0);
              m = T.size(1);
              T.set_size(&ad_emlrtRTEI, &d_st, n, m);
              n *= m;
              for (m = 0; m < n; m++) {
                T[m] = rtNaN;
              }
            }
          }
          c_st.site = &td_emlrtRSI;
          d_st.site = &ae_emlrtRSI;
          vleft = 0.0;
          ilo_t = (ptrdiff_t)T.size(0);
          wr.set_size(&bd_emlrtRTEI, &d_st, 1, T.size(0));
          wi.set_size(&cd_emlrtRTEI, &d_st, 1, T.size(0));
          info_t = LAPACKE_dhseqr(102, 'S', 'N', ilo_t, (ptrdiff_t)1,
                                  (ptrdiff_t)T.size(0), &(T.data())[0], ilo_t,
                                  &wr[0], &wi[0], &vleft, (ptrdiff_t)T.size(0));
          jend = (int32_T)info_t;
          e_st.site = &be_emlrtRSI;
          if (jend < 0) {
            boolean_T b_p;
            p = true;
            b_p = false;
            if (jend == -7) {
              b_p = true;
            } else if (jend == -11) {
              b_p = true;
            }
            if (!b_p) {
              if (jend == -1010) {
                emlrtErrorWithMessageIdR2018a(
                    &e_st, &r_emlrtRTEI, "MATLAB:nomem", "MATLAB:nomem", 0);
              } else {
                emlrtErrorWithMessageIdR2018a(
                    &e_st, &q_emlrtRTEI, "Coder:toolbox:LAPACKCallErrorInfo",
                    "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 14, &c_fname[0],
                    12, jend);
              }
            }
          } else {
            p = false;
          }
          if (p) {
            n = T.size(0);
            m = T.size(1);
            T.set_size(&dd_emlrtRTEI, &d_st, n, m);
            n *= m;
            for (m = 0; m < n; m++) {
              T[m] = rtNaN;
            }
          }
          if (jend != 0) {
            c_st.site = &ud_emlrtRSI;
            internal::warning(&c_st);
          }
        }
        b_st.site = &pd_emlrtRSI;
        n = T.size(0);
        scale.set_size(&fd_emlrtRTEI, &b_st, T.size(0));
        c_st.site = &ce_emlrtRSI;
        if (T.size(0) > 2147483646) {
          d_st.site = &u_emlrtRSI;
          check_forloop_overflow_error(&d_st);
        }
        for (m = 0; m < n; m++) {
          scale[m] = T[m + T.size(0) * m];
        }
        V.set_size(&gd_emlrtRTEI, sp, scale.size(0));
        n = scale.size(0);
        for (m = 0; m < n; m++) {
          V[m].re = scale[m];
          V[m].im = 0.0;
        }
      } else {
        int32_T jend;
        int32_T n;
        st.site = &kd_emlrtRSI;
        b_st.site = &ee_emlrtRSI;
        c_st.site = &fe_emlrtRSI;
        T.set_size(&sc_emlrtRTEI, &c_st, A.size(0), A.size(1));
        n = A.size(0) * A.size(1);
        for (int32_T m{0}; m < n; m++) {
          T[m] = A[m];
        }
        ptrdiff_t info_t;
        n = A.size(1);
        scale.set_size(&tc_emlrtRTEI, &c_st, A.size(1));
        V.set_size(&uc_emlrtRTEI, &c_st, A.size(1));
        wreal.set_size(&vc_emlrtRTEI, &c_st, A.size(1));
        wimag.set_size(&wc_emlrtRTEI, &c_st, A.size(1));
        info_t = LAPACKE_dgeevx(
            102, 'B', 'N', 'N', 'N', (ptrdiff_t)A.size(1), &(T.data())[0],
            (ptrdiff_t)A.size(0), &(wreal.data())[0], &(wimag.data())[0],
            &vleft, (ptrdiff_t)1, &vright, (ptrdiff_t)1, &ilo_t, &ihi_t,
            &(scale.data())[0], &abnrm, &rconde, &rcondv);
        jend = (int32_T)info_t;
        d_st.site = &he_emlrtRSI;
        if (jend < 0) {
          if (jend == -1010) {
            emlrtErrorWithMessageIdR2018a(&d_st, &r_emlrtRTEI, "MATLAB:nomem",
                                          "MATLAB:nomem", 0);
          } else {
            emlrtErrorWithMessageIdR2018a(&d_st, &q_emlrtRTEI,
                                          "Coder:toolbox:LAPACKCallErrorInfo",
                                          "Coder:toolbox:LAPACKCallErrorInfo",
                                          5, 4, 14, &fname[0], 12, jend);
          }
        }
        d_st.site = &ge_emlrtRSI;
        if (A.size(1) > 2147483646) {
          e_st.site = &u_emlrtRSI;
          check_forloop_overflow_error(&e_st);
        }
        for (i = 0; i < n; i++) {
          V[i].re = wreal[i];
          V[i].im = wimag[i];
        }
        if (jend != 0) {
          b_st.site = &de_emlrtRSI;
          internal::b_warning(&b_st);
        }
      }
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

} // namespace coder

// End of code generation (eig.cpp)
