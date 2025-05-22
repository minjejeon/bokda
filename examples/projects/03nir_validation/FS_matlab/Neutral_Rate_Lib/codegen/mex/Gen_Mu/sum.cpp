//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// sum.cpp
//
// Code generation for function 'sum'
//

// Include files
#include "sum.h"
#include "Gen_Mu_data.h"
#include "eml_int_forloop_overflow_check.h"
#include "rt_nonfinite.h"
#include "sumMatrixIncludeNaN.h"
#include "coder_array.h"

// Variable Definitions
static emlrtRSInfo fd_emlrtRSI{
    20,    // lineNo
    "sum", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\sum.m" // pathName
};

static emlrtRSInfo gd_emlrtRSI{
    99,        // lineNo
    "sumprod", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumpro"
    "d.m" // pathName
};

static emlrtRSInfo hd_emlrtRSI{
    74,                      // lineNo
    "combineVectorElements", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\combin"
    "eVectorElements.m" // pathName
};

static emlrtRSInfo id_emlrtRSI{
    107,                // lineNo
    "blockedSummation", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\blocke"
    "dSummation.m" // pathName
};

static emlrtRSInfo jd_emlrtRSI{
    22,                    // lineNo
    "sumMatrixIncludeNaN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

static emlrtRSInfo kd_emlrtRSI{
    41,                 // lineNo
    "sumMatrixColumns", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

static emlrtRSInfo ld_emlrtRSI{
    42,                 // lineNo
    "sumMatrixColumns", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

static emlrtRSInfo md_emlrtRSI{
    50,                 // lineNo
    "sumMatrixColumns", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

static emlrtRSInfo nd_emlrtRSI{
    53,                 // lineNo
    "sumMatrixColumns", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

static emlrtRSInfo od_emlrtRSI{
    57,                 // lineNo
    "sumMatrixColumns", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

static emlrtRSInfo vd_emlrtRSI{
    99,                 // lineNo
    "blockedSummation", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\blocke"
    "dSummation.m" // pathName
};

static emlrtRTEInfo l_emlrtRTEI{
    76,        // lineNo
    9,         // colNo
    "sumprod", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumpro"
    "d.m" // pName
};

static emlrtRTEInfo m_emlrtRTEI{
    46,        // lineNo
    23,        // colNo
    "sumprod", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumpro"
    "d.m" // pName
};

static emlrtRTEInfo cc_emlrtRTEI{
    35,                    // lineNo
    20,                    // colNo
    "sumMatrixIncludeNaN", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pName
};

static emlrtRTEInfo dc_emlrtRTEI{
    20,    // lineNo
    1,     // colNo
    "sum", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\sum.m" // pName
};

// Function Definitions
namespace coder {
real_T sum(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x)
{
  array<real_T, 1U> c_x;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack st;
  real_T y;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &fd_emlrtRSI;
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
  b_st.site = &gd_emlrtRSI;
  c_st.site = &hd_emlrtRSI;
  if (x.size(1) == 0) {
    y = 0.0;
  } else {
    d_st.site = &vd_emlrtRSI;
    e_st.site = &jd_emlrtRSI;
    if (x.size(1) < 4096) {
      int32_T b_x;
      b_x = x.size(1);
      c_x = x.reshape(b_x);
      f_st.site = &ld_emlrtRSI;
      y = sumColumnB(&f_st, c_x, x.size(1));
    } else {
      int32_T b_x;
      int32_T inb;
      int32_T nfb;
      int32_T nleft;
      nfb = x.size(1) / 4096;
      inb = nfb << 12;
      nleft = x.size(1) - inb;
      b_x = x.size(1);
      c_x = x.reshape(b_x);
      y = sumColumnB4(c_x, 1);
      if (2 <= nfb) {
        b_x = x.size(1);
      }
      for (int32_T ib{2}; ib <= nfb; ib++) {
        c_x = x.reshape(b_x);
        y += sumColumnB4(c_x, ((ib - 1) << 12) + 1);
      }
      if (nleft > 0) {
        b_x = x.size(1);
        c_x = x.reshape(b_x);
        f_st.site = &od_emlrtRSI;
        y += sumColumnB(&f_st, c_x, nleft, inb + 1);
      }
    }
  }
  return y;
}

void sum(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x,
         ::coder::array<real_T, 2U> &y)
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack g_st;
  emlrtStack st;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &fd_emlrtRSI;
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
  if ((x.size(0) == 1) && (x.size(1) != 1)) {
    emlrtErrorWithMessageIdR2018a(&st, &m_emlrtRTEI,
                                  "Coder:toolbox:autoDimIncompatibility",
                                  "Coder:toolbox:autoDimIncompatibility", 0);
  }
  if ((x.size(0) == 0) && (x.size(1) == 0)) {
    emlrtErrorWithMessageIdR2018a(&st, &l_emlrtRTEI,
                                  "Coder:toolbox:UnsupportedSpecialEmpty",
                                  "Coder:toolbox:UnsupportedSpecialEmpty", 0);
  }
  b_st.site = &gd_emlrtRSI;
  c_st.site = &hd_emlrtRSI;
  if ((x.size(0) == 0) || (x.size(1) == 0)) {
    int32_T ncols;
    y.set_size(&dc_emlrtRTEI, &c_st, 1, x.size(1));
    ncols = x.size(1);
    for (int32_T nfb{0}; nfb < ncols; nfb++) {
      y[nfb] = 0.0;
    }
  } else {
    int32_T ncols;
    d_st.site = &id_emlrtRSI;
    e_st.site = &jd_emlrtRSI;
    y.set_size(&cc_emlrtRTEI, &e_st, 1, x.size(1));
    ncols = x.size(1) - 1;
    if (x.size(0) < 4096) {
      f_st.site = &kd_emlrtRSI;
      if (x.size(1) > 2147483646) {
        g_st.site = &cb_emlrtRSI;
        check_forloop_overflow_error(&g_st);
      }
      for (int32_T col{0}; col <= ncols; col++) {
        f_st.site = &ld_emlrtRSI;
        y[col] = sumColumnB(&f_st, x, col + 1, x.size(0));
      }
    } else {
      int32_T inb;
      int32_T nfb;
      int32_T nleft;
      nfb = x.size(0) / 4096;
      inb = nfb << 12;
      nleft = x.size(0) - inb;
      f_st.site = &md_emlrtRSI;
      if (x.size(1) > 2147483646) {
        g_st.site = &cb_emlrtRSI;
        check_forloop_overflow_error(&g_st);
      }
      for (int32_T col{0}; col <= ncols; col++) {
        real_T s;
        s = sumColumnB4(x, col + 1, 1);
        f_st.site = &nd_emlrtRSI;
        for (int32_T ib{2}; ib <= nfb; ib++) {
          s += sumColumnB4(x, col + 1, ((ib - 1) << 12) + 1);
        }
        if (nleft > 0) {
          f_st.site = &od_emlrtRSI;
          s += sumColumnB(&f_st, x, col + 1, nleft, inb + 1);
        }
        y[col] = s;
      }
    }
  }
}

} // namespace coder

// End of code generation (sum.cpp)
