//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// makeYX.cpp
//
// Code generation for function 'makeYX'
//

// Include files
#include "makeYX.h"
#include "eml_int_forloop_overflow_check.h"
#include "makeYX_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo emlrtRSI{
    20,       // lineNo
    "makeYX", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m" // pathName
};

static emlrtRSInfo
    b_emlrtRSI{
        50,    // lineNo
        "eye", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\eye.m" // pathName
    };

static emlrtRSInfo
    c_emlrtRSI{
        96,    // lineNo
        "eye", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\eye.m" // pathName
    };

static emlrtRSInfo e_emlrtRSI{
    21,                               // lineNo
    "eml_int_forloop_overflow_check", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\eml\\eml_int_forloop_"
    "overflow_check.m" // pathName
};

static emlrtRSInfo
    f_emlrtRSI{
        32,     // lineNo
        "kron", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\kron.m" // pathName
    };

static emlrtRSInfo
    g_emlrtRSI{
        33,     // lineNo
        "kron", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\kron.m" // pathName
    };

static emlrtRSInfo
    h_emlrtRSI{
        34,     // lineNo
        "kron", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\kron.m" // pathName
    };

static emlrtDCInfo emlrtDCI{
    7,        // lineNo
    8,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    1 // checkKind
};

static emlrtBCInfo emlrtBCI{
    -1,       // iFirst
    -1,       // iLast
    7,        // lineNo
    8,        // colNo
    "Y",      // aName
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    0 // checkKind
};

static emlrtBCInfo b_emlrtBCI{
    -1,       // iFirst
    -1,       // iLast
    7,        // lineNo
    12,       // colNo
    "Y",      // aName
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    0 // checkKind
};

static emlrtRTEInfo emlrtRTEI{
    11,       // lineNo
    9,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m" // pName
};

static emlrtDCInfo b_emlrtDCI{
    12,       // lineNo
    29,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    1 // checkKind
};

static emlrtBCInfo c_emlrtBCI{
    -1,       // iFirst
    -1,       // iLast
    12,       // lineNo
    29,       // colNo
    "Y",      // aName
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    0 // checkKind
};

static emlrtBCInfo d_emlrtBCI{
    -1,       // iFirst
    -1,       // iLast
    12,       // lineNo
    37,       // colNo
    "Y",      // aName
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    0 // checkKind
};

static emlrtRTEInfo b_emlrtRTEI{
    19,       // lineNo
    9,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m" // pName
};

static emlrtBCInfo e_emlrtBCI{
    -1,       // iFirst
    -1,       // iLast
    12,       // lineNo
    10,       // colNo
    "YL",     // aName
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    0 // checkKind
};

static emlrtBCInfo f_emlrtBCI{
    -1,       // iFirst
    -1,       // iLast
    12,       // lineNo
    20,       // colNo
    "YL",     // aName
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    0 // checkKind
};

static emlrtECInfo emlrtECI{
    -1,       // nDims
    12,       // lineNo
    5,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m" // pName
};

static emlrtBCInfo g_emlrtBCI{
    -1,       // iFirst
    -1,       // iLast
    20,       // lineNo
    26,       // colNo
    "YL",     // aName
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    0 // checkKind
};

static emlrtBCInfo h_emlrtBCI{
    -1,       // iFirst
    -1,       // iLast
    21,       // lineNo
    13,       // colNo
    "YLm",    // aName
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    0 // checkKind
};

static emlrtECInfo b_emlrtECI{
    -1,       // nDims
    21,       // lineNo
    5,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m" // pName
};

static emlrtDCInfo c_emlrtDCI{
    10,       // lineNo
    12,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    1 // checkKind
};

static emlrtDCInfo d_emlrtDCI{
    10,       // lineNo
    12,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    4 // checkKind
};

static emlrtDCInfo e_emlrtDCI{
    10,       // lineNo
    16,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    1 // checkKind
};

static emlrtDCInfo f_emlrtDCI{
    10,       // lineNo
    16,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    4 // checkKind
};

static emlrtDCInfo g_emlrtDCI{
    18,       // lineNo
    16,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    1 // checkKind
};

static emlrtDCInfo h_emlrtDCI{
    18,       // lineNo
    16,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    4 // checkKind
};

static emlrtDCInfo i_emlrtDCI{
    18,       // lineNo
    21,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    1 // checkKind
};

static emlrtDCInfo j_emlrtDCI{
    18,       // lineNo
    21,       // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    4 // checkKind
};

static emlrtDCInfo k_emlrtDCI{
    10,       // lineNo
    1,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    1 // checkKind
};

static emlrtDCInfo l_emlrtDCI{
    10,       // lineNo
    1,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    4 // checkKind
};

static emlrtDCInfo m_emlrtDCI{
    18,       // lineNo
    1,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    1 // checkKind
};

static emlrtDCInfo n_emlrtDCI{
    18,       // lineNo
    1,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m", // pName
    4 // checkKind
};

static emlrtRTEInfo d_emlrtRTEI{
    7,        // lineNo
    1,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m" // pName
};

static emlrtRTEInfo e_emlrtRTEI{
    10,       // lineNo
    1,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m" // pName
};

static emlrtRTEInfo f_emlrtRTEI{
    18,       // lineNo
    1,        // colNo
    "makeYX", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\makeYX.m" // pName
};

static emlrtRTEInfo
    g_emlrtRTEI{
        94,    // lineNo
        5,     // colNo
        "eye", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\eye.m" // pName
    };

static emlrtRTEInfo
    h_emlrtRTEI{
        30,     // lineNo
        20,     // colNo
        "kron", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\kron.m" // pName
    };

// Function Definitions
void makeYX(const emlrtStack *sp, const coder::array<real_T, 2U> &Y, real_T p,
            coder::array<real_T, 2U> &Y0, coder::array<real_T, 3U> &YLm)
{
  coder::array<real_T, 2U> YL;
  coder::array<real_T, 2U> xt;
  coder::array<int8_T, 2U> b_I;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T d;
  real_T d1;
  real_T kki;
  int32_T b_YL[2];
  int32_T iv[2];
  int32_T b_i;
  int32_T b_i1;
  int32_T i;
  int32_T i1;
  int32_T i2;
  int32_T j2;
  int32_T k;
  int32_T kidx;
  int32_T loop_ub;
  int32_T m;
  int32_T ma;
  int32_T na;
  int32_T nb;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  종속, 설명변수 만들기 %%%%%%%%%%%%%%%%%%%%%%%%%%
  //  number of columns
  k = Y.size(1);
  //  변수의 수
  //  시계열의 크기
  if (p + 1.0 > Y.size(0)) {
    i = 0;
    i1 = 0;
  } else {
    if (p + 1.0 != static_cast<int32_T>(muDoubleScalarFloor(p + 1.0))) {
      emlrtIntegerCheckR2012b(p + 1.0, &emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(p + 1.0) < 1) ||
        (static_cast<int32_T>(p + 1.0) > Y.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(p + 1.0), 1, Y.size(0),
                                    &emlrtBCI, (emlrtCTX)sp);
    }
    i = static_cast<int32_T>(p + 1.0) - 1;
    if (Y.size(0) < 1) {
      emlrtDynamicBoundsCheckR2012b(Y.size(0), 1, Y.size(0), &b_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    i1 = Y.size(0);
  }
  kidx = Y.size(1);
  ma = i1 - i;
  Y0.set_size(&d_emlrtRTEI, sp, ma, Y.size(1));
  for (i1 = 0; i1 < kidx; i1++) {
    for (b_i1 = 0; b_i1 < ma; b_i1++) {
      Y0[b_i1 + Y0.size(0) * i1] = Y[(i + b_i1) + Y.size(0) * i1];
    }
  }
  //  종속변수
  // 설명변수(=Y의 과거값) 만들기
  d = static_cast<real_T>(Y.size(0)) - p;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(d, &d_emlrtDCI, (emlrtCTX)sp);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &c_emlrtDCI, (emlrtCTX)sp);
  }
  YL.set_size(&e_emlrtRTEI, sp, static_cast<int32_T>(d), YL.size(1));
  d = p * static_cast<real_T>(Y.size(1));
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(d, &f_emlrtDCI, (emlrtCTX)sp);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &e_emlrtDCI, (emlrtCTX)sp);
  }
  YL.set_size(&e_emlrtRTEI, sp, YL.size(0), static_cast<int32_T>(d));
  d = static_cast<real_T>(Y.size(0)) - p;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(d, &l_emlrtDCI, (emlrtCTX)sp);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &k_emlrtDCI, (emlrtCTX)sp);
  }
  d1 = p * static_cast<real_T>(Y.size(1));
  if (!(d1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(d1, &l_emlrtDCI, (emlrtCTX)sp);
  }
  if (d1 != static_cast<int32_T>(muDoubleScalarFloor(d1))) {
    emlrtIntegerCheckR2012b(d1, &k_emlrtDCI, (emlrtCTX)sp);
  }
  kidx = static_cast<int32_T>(d) * static_cast<int32_T>(d1);
  for (i = 0; i < kidx; i++) {
    YL[i] = 0.0;
  }
  i = static_cast<int32_T>(p);
  emlrtForLoopVectorCheckR2021a(1.0, 1.0, p, mxDOUBLE_CLASS,
                                static_cast<int32_T>(p), &emlrtRTEI,
                                (emlrtCTX)sp);
  if (0 <= static_cast<int32_T>(p) - 1) {
    iv[1] = Y.size(1);
    na = Y.size(1);
  }
  for (b_i = 0; b_i < i; b_i++) {
    d = (p + 1.0) - (static_cast<real_T>(b_i) + 1.0);
    i1 = static_cast<int32_T>(static_cast<real_T>(Y.size(0)) -
                              (static_cast<real_T>(b_i) + 1.0));
    if (d > i1) {
      b_i1 = 0;
      i1 = 0;
    } else {
      if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
        emlrtIntegerCheckR2012b(d, &b_emlrtDCI, (emlrtCTX)sp);
      }
      if ((static_cast<int32_T>(d) < 1) ||
          (static_cast<int32_T>(d) > Y.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, Y.size(0),
                                      &c_emlrtBCI, (emlrtCTX)sp);
      }
      b_i1 = static_cast<int32_T>(d) - 1;
      if ((i1 < 1) || (i1 > Y.size(0))) {
        emlrtDynamicBoundsCheckR2012b(i1, 1, Y.size(0), &d_emlrtBCI,
                                      (emlrtCTX)sp);
      }
    }
    d = static_cast<real_T>(k) * ((static_cast<real_T>(b_i) + 1.0) - 1.0) + 1.0;
    d1 = static_cast<real_T>(k) * (static_cast<real_T>(b_i) + 1.0);
    if (d > d1) {
      ma = 0;
      j2 = 0;
    } else {
      if ((static_cast<int32_T>(d) < 1) ||
          (static_cast<int32_T>(d) > YL.size(1))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, YL.size(1),
                                      &e_emlrtBCI, (emlrtCTX)sp);
      }
      ma = static_cast<int32_T>(d) - 1;
      if ((static_cast<int32_T>(d1) < 1) ||
          (static_cast<int32_T>(d1) > YL.size(1))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d1), 1, YL.size(1),
                                      &f_emlrtBCI, (emlrtCTX)sp);
      }
      j2 = static_cast<int32_T>(d1);
    }
    b_YL[0] = YL.size(0);
    b_YL[1] = j2 - ma;
    kidx = i1 - b_i1;
    iv[0] = kidx;
    emlrtSubAssignSizeCheckR2012b(&b_YL[0], 2, &iv[0], 2, &emlrtECI,
                                  (emlrtCTX)sp);
    for (i1 = 0; i1 < na; i1++) {
      for (j2 = 0; j2 < kidx; j2++) {
        YL[j2 + YL.size(0) * (ma + i1)] = Y[(b_i1 + j2) + Y.size(0) * i1];
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  //  각 식에 있는 설명변수의 수
  kki = static_cast<real_T>(Y.size(1)) * (p * static_cast<real_T>(Y.size(1)));
  YLm.set_size(&f_emlrtRTEI, sp, Y.size(1), YLm.size(1), YLm.size(2));
  if (!(kki >= 0.0)) {
    emlrtNonNegativeCheckR2012b(kki, &h_emlrtDCI, (emlrtCTX)sp);
  }
  d = static_cast<int32_T>(muDoubleScalarFloor(kki));
  if (kki != d) {
    emlrtIntegerCheckR2012b(kki, &g_emlrtDCI, (emlrtCTX)sp);
  }
  YLm.set_size(&f_emlrtRTEI, sp, YLm.size(0), static_cast<int32_T>(kki),
               YLm.size(2));
  d1 = static_cast<real_T>(Y.size(0)) - p;
  if (!(d1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(d1, &j_emlrtDCI, (emlrtCTX)sp);
  }
  if (d1 != static_cast<int32_T>(muDoubleScalarFloor(d1))) {
    emlrtIntegerCheckR2012b(d1, &i_emlrtDCI, (emlrtCTX)sp);
  }
  YLm.set_size(&f_emlrtRTEI, sp, YLm.size(0), YLm.size(1),
               static_cast<int32_T>(d1));
  if (kki != d) {
    emlrtIntegerCheckR2012b(kki, &m_emlrtDCI, (emlrtCTX)sp);
  }
  d = static_cast<real_T>(Y.size(0)) - p;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(d, &n_emlrtDCI, (emlrtCTX)sp);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &m_emlrtDCI, (emlrtCTX)sp);
  }
  kidx = Y.size(1) * static_cast<int32_T>(kki) * static_cast<int32_T>(d);
  for (i = 0; i < kidx; i++) {
    YLm[i] = 0.0;
  }
  //  설명변수를 3차원으로 새롭게 저장할 방
  d = static_cast<real_T>(Y.size(0)) - p;
  i = static_cast<int32_T>(d);
  emlrtForLoopVectorCheckR2021a(1.0, 1.0, d, mxDOUBLE_CLASS,
                                static_cast<int32_T>(d), &b_emlrtRTEI,
                                (emlrtCTX)sp);
  if (0 <= static_cast<int32_T>(d) - 1) {
    m = Y.size(1);
    loop_ub = Y.size(1) * Y.size(1);
    nb = YL.size(1);
    i2 = YL.size(1);
  }
  for (int32_T t{0}; t < i; t++) {
    st.site = &emlrtRSI;
    b_st.site = &b_emlrtRSI;
    b_I.set_size(&g_emlrtRTEI, &st, k, k);
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_I[i1] = 0;
    }
    if (k > 0) {
      b_st.site = &c_emlrtRSI;
      if (k > 2147483646) {
        c_st.site = &e_emlrtRSI;
        coder::check_forloop_overflow_error(&c_st);
      }
      for (ma = 0; ma < m; ma++) {
        b_I[ma + b_I.size(0) * ma] = 1;
      }
    }
    st.site = &emlrtRSI;
    if ((t + 1 < 1) || (t + 1 > YL.size(0))) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, YL.size(0), &g_emlrtBCI, &st);
    }
    ma = b_I.size(0);
    na = b_I.size(1);
    xt.set_size(&h_emlrtRTEI, &st, b_I.size(0), i2 * b_I.size(1));
    kidx = -1;
    b_st.site = &f_emlrtRSI;
    if ((1 <= b_I.size(1)) && (b_I.size(1) > 2147483646)) {
      c_st.site = &e_emlrtRSI;
      coder::check_forloop_overflow_error(&c_st);
    }
    for (b_i = 0; b_i < na; b_i++) {
      b_st.site = &g_emlrtRSI;
      if ((1 <= nb) && (nb > 2147483646)) {
        c_st.site = &e_emlrtRSI;
        coder::check_forloop_overflow_error(&c_st);
      }
      for (j2 = 0; j2 < nb; j2++) {
        b_st.site = &h_emlrtRSI;
        if ((1 <= ma) && (ma > 2147483646)) {
          c_st.site = &e_emlrtRSI;
          coder::check_forloop_overflow_error(&c_st);
        }
        for (i1 = 0; i1 < ma; i1++) {
          xt[(kidx + i1) + 1] =
              static_cast<real_T>(b_I[i1 + b_I.size(0) * b_i]) *
              YL[t + YL.size(0) * j2];
        }
        kidx += ma;
      }
    }
    if ((t + 1 < 1) || (t + 1 > YLm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, YLm.size(2), &h_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    b_YL[0] = YLm.size(0);
    b_YL[1] = YLm.size(1);
    emlrtSubAssignSizeCheckR2012b(&b_YL[0], 2, xt.size(), 2, &b_emlrtECI,
                                  (emlrtCTX)sp);
    kidx = xt.size(1);
    for (i1 = 0; i1 < kidx; i1++) {
      ma = xt.size(0);
      for (b_i1 = 0; b_i1 < ma; b_i1++) {
        YLm[(b_i1 + YLm.size(0) * i1) + YLm.size(0) * YLm.size(1) * t] =
            xt[b_i1 + xt.size(0) * i1];
      }
    }
    //  p by k
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (makeYX.cpp)
