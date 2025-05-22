//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// inv.cpp
//
// Code generation for function 'inv'
//

// Include files
#include "inv.h"
#include "Gen_Mu_data.h"
#include "eml_int_forloop_overflow_check.h"
#include "norm.h"
#include "rt_nonfinite.h"
#include "warning.h"
#include "blas.h"
#include "coder_array.h"
#include "lapacke.h"
#include "mwmathutil.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo kb_emlrtRSI{
    21,    // lineNo
    "inv", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo lb_emlrtRSI{
    22,    // lineNo
    "inv", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo mb_emlrtRSI{
    173,      // lineNo
    "invNxN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo nb_emlrtRSI{
    174,      // lineNo
    "invNxN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo ob_emlrtRSI{
    177,      // lineNo
    "invNxN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo pb_emlrtRSI{
    180,      // lineNo
    "invNxN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo qb_emlrtRSI{
    183,      // lineNo
    "invNxN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo rb_emlrtRSI{
    190,      // lineNo
    "invNxN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo sb_emlrtRSI{
    27,       // lineNo
    "xgetrf", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pathName
};

static emlrtRSInfo tb_emlrtRSI{
    91,             // lineNo
    "ceval_xgetrf", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pathName
};

static emlrtRSInfo ub_emlrtRSI{
    58,             // lineNo
    "ceval_xgetrf", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pathName
};

static emlrtRSInfo vb_emlrtRSI{
    28,       // lineNo
    "repmat", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\repmat.m" // pathName
};

static emlrtRSInfo
    xb_emlrtRSI{
        82,      // lineNo
        "colon", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pathName
    };

static emlrtRSInfo
    yb_emlrtRSI{
        140,                            // lineNo
        "eml_integer_colon_dispatcher", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pathName
    };

static emlrtRSInfo
    ac_emlrtRSI{
        168,                        // lineNo
        "eml_signed_integer_colon", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pathName
    };

static emlrtRSInfo bc_emlrtRSI{
    14,              // lineNo
    "eml_ipiv2perm", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\private\\eml_"
    "ipiv2perm.m" // pathName
};

static emlrtRSInfo cc_emlrtRSI{
    67,      // lineNo
    "xtrsm", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+blas\\xtrsm."
    "m" // pathName
};

static emlrtRSInfo dc_emlrtRSI{
    81,           // lineNo
    "xtrsm_blas", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+blas\\xtrsm."
    "m" // pathName
};

static emlrtRSInfo ec_emlrtRSI{
    42,          // lineNo
    "checkcond", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtRSInfo fc_emlrtRSI{
    46,          // lineNo
    "checkcond", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pathName
};

static emlrtMCInfo c_emlrtMCI{
    53,        // lineNo
    19,        // colNo
    "flt2str", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\flt2str.m" // pName
};

static emlrtRTEInfo f_emlrtRTEI{
    14,    // lineNo
    15,    // colNo
    "inv", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pName
};

static emlrtRTEInfo ab_emlrtRTEI{
    21,    // lineNo
    5,     // colNo
    "inv", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pName
};

static emlrtRTEInfo bb_emlrtRTEI{
    1,        // lineNo
    37,       // colNo
    "xgetrf", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pName
};

static emlrtRTEInfo cb_emlrtRTEI{
    58,       // lineNo
    29,       // colNo
    "xgetrf", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pName
};

static emlrtRTEInfo db_emlrtRTEI{
    89,       // lineNo
    27,       // colNo
    "xgetrf", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pName
};

static emlrtRTEInfo
    eb_emlrtRTEI{
        164,     // lineNo
        20,      // colNo
        "colon", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pName
    };

static emlrtRTEInfo fb_emlrtRTEI{
    19,    // lineNo
    5,     // colNo
    "inv", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\inv.m" // pName
};

static emlrtRSInfo ne_emlrtRSI{
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
void inv(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x,
         ::coder::array<real_T, 2U> &y)
{
  static const int32_T iv[2]{1, 6};
  static const char_T fname[19]{'L', 'A', 'P', 'A', 'C', 'K', 'E',
                                '_', 'd', 'g', 'e', 't', 'r', 'f',
                                '_', 'w', 'o', 'r', 'k'};
  static const char_T rfmt[6]{'%', '1', '4', '.', '6', 'e'};
  ptrdiff_t info_t;
  ptrdiff_t lda_t;
  ptrdiff_t ldb_t;
  ptrdiff_t n_t;
  array<ptrdiff_t, 1U> ipiv_t;
  array<real_T, 2U> b_x;
  array<int32_T, 2U> ipiv;
  array<int32_T, 2U> p;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack g_st;
  emlrtStack h_st;
  emlrtStack st;
  const mxArray *b_y;
  const mxArray *c_y;
  const mxArray *m;
  real_T n1x;
  char_T str[14];
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
  g_st.prev = &f_st;
  g_st.tls = f_st.tls;
  h_st.prev = &g_st;
  h_st.tls = g_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  if (x.size(0) != x.size(1)) {
    emlrtErrorWithMessageIdR2018a(sp, &f_emlrtRTEI, "Coder:MATLAB:square",
                                  "Coder:MATLAB:square", 0);
  }
  if ((x.size(0) == 0) || (x.size(1) == 0)) {
    int32_T b_n;
    y.set_size(&fb_emlrtRTEI, sp, x.size(0), x.size(1));
    b_n = x.size(0) * x.size(1);
    for (int32_T i{0}; i < b_n; i++) {
      y[i] = x[i];
    }
  } else {
    real_T n1xinv;
    real_T rc;
    int32_T b_n;
    int32_T i;
    int32_T k;
    int32_T n;
    int32_T yk;
    st.site = &kb_emlrtRSI;
    n = x.size(0);
    y.set_size(&ab_emlrtRTEI, &st, x.size(0), x.size(1));
    b_n = x.size(0) * x.size(1);
    for (i = 0; i < b_n; i++) {
      y[i] = 0.0;
    }
    b_st.site = &mb_emlrtRSI;
    b_x.set_size(&bb_emlrtRTEI, &b_st, x.size(0), x.size(1));
    b_n = x.size(0) * x.size(1);
    for (i = 0; i < b_n; i++) {
      b_x[i] = x[i];
    }
    c_st.site = &sb_emlrtRSI;
    d_st.site = &ub_emlrtRSI;
    e_st.site = &vb_emlrtRSI;
    ipiv_t.set_size(&cb_emlrtRTEI, &d_st, muIntScalarMin_sint32(n, n));
    info_t = LAPACKE_dgetrf_work(102, (ptrdiff_t)x.size(0),
                                 (ptrdiff_t)x.size(0), &(b_x.data())[0],
                                 (ptrdiff_t)x.size(0), &(ipiv_t.data())[0]);
    b_n = (int32_T)info_t;
    ipiv.set_size(&db_emlrtRTEI, &c_st, 1, ipiv_t.size(0));
    d_st.site = &tb_emlrtRSI;
    if (b_n < 0) {
      if (b_n == -1010) {
        emlrtErrorWithMessageIdR2018a(&d_st, &g_emlrtRTEI, "MATLAB:nomem",
                                      "MATLAB:nomem", 0);
      } else {
        emlrtErrorWithMessageIdR2018a(
            &d_st, &h_emlrtRTEI, "Coder:toolbox:LAPACKCallErrorInfo",
            "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 19, &fname[0], 12, b_n);
      }
    }
    i = ipiv_t.size(0) - 1;
    for (k = 0; k <= i; k++) {
      ipiv[k] = (int32_T)ipiv_t[k];
    }
    b_st.site = &nb_emlrtRSI;
    c_st.site = &bc_emlrtRSI;
    d_st.site = &wb_emlrtRSI;
    e_st.site = &xb_emlrtRSI;
    f_st.site = &yb_emlrtRSI;
    b_n = x.size(0);
    p.set_size(&eb_emlrtRTEI, &f_st, 1, x.size(0));
    p[0] = 1;
    yk = 1;
    g_st.site = &ac_emlrtRSI;
    if ((2 <= x.size(0)) && (x.size(0) > 2147483646)) {
      h_st.site = &cb_emlrtRSI;
      check_forloop_overflow_error(&h_st);
    }
    for (k = 2; k <= b_n; k++) {
      yk++;
      p[k - 1] = yk;
    }
    i = ipiv.size(1);
    for (k = 0; k < i; k++) {
      b_n = ipiv[k];
      if (b_n > k + 1) {
        yk = p[b_n - 1];
        p[b_n - 1] = p[k];
        p[k] = yk;
      }
    }
    b_st.site = &ob_emlrtRSI;
    for (k = 0; k < n; k++) {
      i = p[k];
      y[k + y.size(0) * (i - 1)] = 1.0;
      b_st.site = &pb_emlrtRSI;
      if ((k + 1 <= n) && (n > 2147483646)) {
        c_st.site = &cb_emlrtRSI;
        check_forloop_overflow_error(&c_st);
      }
      for (yk = k + 1; yk <= n; yk++) {
        if (y[(yk + y.size(0) * (i - 1)) - 1] != 0.0) {
          b_n = yk + 1;
          b_st.site = &qb_emlrtRSI;
          for (int32_T b_i{b_n}; b_i <= n; b_i++) {
            y[(b_i + y.size(0) * (i - 1)) - 1] =
                y[(b_i + y.size(0) * (i - 1)) - 1] -
                y[(yk + y.size(0) * (i - 1)) - 1] *
                    b_x[(b_i + b_x.size(0) * (yk - 1)) - 1];
          }
        }
      }
    }
    b_st.site = &rb_emlrtRSI;
    c_st.site = &cc_emlrtRSI;
    d_st.site = &dc_emlrtRSI;
    n1x = 1.0;
    DIAGA1 = 'N';
    TRANSA1 = 'N';
    UPLO1 = 'U';
    SIDE1 = 'L';
    info_t = (ptrdiff_t)x.size(0);
    n_t = (ptrdiff_t)x.size(0);
    lda_t = (ptrdiff_t)x.size(0);
    ldb_t = (ptrdiff_t)x.size(0);
    dtrsm(&SIDE1, &UPLO1, &TRANSA1, &DIAGA1, &info_t, &n_t, &n1x,
          &(b_x.data())[0], &lda_t, &(y.data())[0], &ldb_t);
    st.site = &lb_emlrtRSI;
    n1x = b_norm(x);
    n1xinv = b_norm(y);
    rc = 1.0 / (n1x * n1xinv);
    if ((n1x == 0.0) || (n1xinv == 0.0) || (rc == 0.0)) {
      b_st.site = &ec_emlrtRSI;
      internal::warning(&b_st);
    } else if (muDoubleScalarIsNaN(rc) || (rc < 2.2204460492503131E-16)) {
      b_st.site = &fc_emlrtRSI;
      b_y = nullptr;
      m = emlrtCreateCharArray(2, &iv[0]);
      emlrtInitCharArrayR2013a(&b_st, 6, m, &rfmt[0]);
      emlrtAssign(&b_y, m);
      c_y = nullptr;
      m = emlrtCreateDoubleScalar(rc);
      emlrtAssign(&c_y, m);
      c_st.site = &ne_emlrtRSI;
      emlrt_marshallIn(&c_st, b_sprintf(&c_st, b_y, c_y, &c_emlrtMCI),
                       "<output of sprintf>", str);
      b_st.site = &fc_emlrtRSI;
      internal::warning(&b_st, str);
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

} // namespace coder

// End of code generation (inv.cpp)
