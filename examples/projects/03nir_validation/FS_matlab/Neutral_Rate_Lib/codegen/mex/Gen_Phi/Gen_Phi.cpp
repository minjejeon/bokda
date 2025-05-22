//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Phi.cpp
//
// Code generation for function 'Gen_Phi'
//

// Include files
#include "Gen_Phi.h"
#include "Gen_Phi_data.h"
#include "abs.h"
#include "assertValidSizeArg.h"
#include "cholmod.h"
#include "eig.h"
#include "eml_mtimes_helper.h"
#include "eye.h"
#include "invpd.h"
#include "maxc.h"
#include "mtimes.h"
#include "randn.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo emlrtRSI{
    7,         // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo b_emlrtRSI{
    8,         // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo c_emlrtRSI{
    12,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo d_emlrtRSI{
    13,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo e_emlrtRSI{
    16,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo f_emlrtRSI{
    19,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo g_emlrtRSI{
    21,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo h_emlrtRSI{
    22,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo i_emlrtRSI{
    24,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo j_emlrtRSI{
    25,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo k_emlrtRSI{
    28,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo l_emlrtRSI{
    29,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo m_emlrtRSI{
    32,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo n_emlrtRSI{
    34,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo o_emlrtRSI{
    38,        // lineNo
    "Gen_Phi", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pathName
};

static emlrtRSInfo ed_emlrtRSI{
    29,                  // lineNo
    "reshapeSizeChecks", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\reshapeSizeChecks.m" // pathName
};

static emlrtRSInfo fd_emlrtRSI{
    109,               // lineNo
    "computeDimsData", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\reshapeSizeChecks.m" // pathName
};

static emlrtRSInfo gd_emlrtRSI{
    24,    // lineNo
    "cat", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRSInfo hd_emlrtRSI{
    96,         // lineNo
    "cat_impl", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRTEInfo emlrtRTEI{
    271,                   // lineNo
    27,                    // colNo
    "check_non_axis_size", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pName
};

static emlrtRTEInfo b_emlrtRTEI{
    59,                  // lineNo
    23,                  // colNo
    "reshapeSizeChecks", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\reshapeSizeChecks.m" // pName
};

static emlrtRTEInfo c_emlrtRTEI{
    57,                  // lineNo
    23,                  // colNo
    "reshapeSizeChecks", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\reshapeSizeChecks.m" // pName
};

static emlrtRTEInfo d_emlrtRTEI{
    52,                  // lineNo
    13,                  // colNo
    "reshapeSizeChecks", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\reshapeSizeChecks.m" // pName
};

static emlrtDCInfo emlrtDCI{
    38,        // lineNo
    36,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    1 // checkKind
};

static emlrtDCInfo b_emlrtDCI{
    38,        // lineNo
    36,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    4 // checkKind
};

static emlrtDCInfo c_emlrtDCI{
    29,        // lineNo
    32,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    1 // checkKind
};

static emlrtDCInfo d_emlrtDCI{
    29,        // lineNo
    32,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    4 // checkKind
};

static emlrtECInfo emlrtECI{
    -1,        // nDims
    25,        // lineNo
    8,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtECInfo b_emlrtECI{
    -1,        // nDims
    21,        // lineNo
    5,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtECInfo c_emlrtECI{
    2,         // nDims
    20,        // lineNo
    11,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtECInfo d_emlrtECI{
    2,         // nDims
    18,        // lineNo
    15,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtECInfo e_emlrtECI{
    2,         // nDims
    17,        // lineNo
    10,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtECInfo f_emlrtECI{
    -1,        // nDims
    13,        // lineNo
    10,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtBCInfo emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    13,        // lineNo
    32,        // colNo
    "Y0",      // aName
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    0 // checkKind
};

static emlrtECInfo g_emlrtECI{
    2,         // nDims
    12,        // lineNo
    10,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtBCInfo b_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    11,        // lineNo
    16,        // colNo
    "X",       // aName
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    0 // checkKind
};

static emlrtDCInfo e_emlrtDCI{
    7,         // lineNo
    12,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    1 // checkKind
};

static emlrtDCInfo f_emlrtDCI{
    7,         // lineNo
    12,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    4 // checkKind
};

static emlrtDCInfo g_emlrtDCI{
    8,         // lineNo
    1,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    1 // checkKind
};

static emlrtDCInfo h_emlrtDCI{
    8,         // lineNo
    1,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m", // pName
    4 // checkKind
};

static emlrtRTEInfo u_emlrtRTEI{
    7,         // lineNo
    1,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo v_emlrtRTEI{
    8,         // lineNo
    1,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo w_emlrtRTEI{
    11,        // lineNo
    10,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo x_emlrtRTEI{
    13,        // lineNo
    29,        // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo y_emlrtRTEI{
    17,        // lineNo
    1,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo
    ab_emlrtRTEI{
        77,                  // lineNo
        13,                  // colNo
        "eml_mtimes_helper", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pName
    };

static emlrtRTEInfo bb_emlrtRTEI{
    20,        // lineNo
    1,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo cb_emlrtRTEI{
    28,        // lineNo
    1,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo db_emlrtRTEI{
    24,    // lineNo
    5,     // colNo
    "cat", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pName
};

static emlrtRTEInfo eb_emlrtRTEI{
    29,        // lineNo
    1,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo fb_emlrtRTEI{
    37,        // lineNo
    5,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

static emlrtRTEInfo gb_emlrtRTEI{
    38,        // lineNo
    5,         // colNo
    "Gen_Phi", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Phi.m" // pName
};

// Function Definitions
void Gen_Phi(const emlrtStack *sp, const coder::array<real_T, 2U> &Y0,
             const coder::array<real_T, 3U> &YLm,
             const coder::array<real_T, 2U> &Phi0, real_T p,
             const coder::array<real_T, 1U> &b_,
             const coder::array<real_T, 2U> &var_,
             const coder::array<real_T, 2U> &Omega_inv,
             coder::array<real_T, 2U> &Phi, coder::array<real_T, 2U> &Fm,
             coder::array<real_T, 1U> &beta, real_T *is_reject)
{
  coder::array<creal_T, 1U> eigF;
  coder::array<real_T, 2U> XX;
  coder::array<real_T, 2U> b_Y0;
  coder::array<real_T, 2U> b_YLm;
  coder::array<real_T, 2U> precb_;
  coder::array<real_T, 2U> r;
  coder::array<real_T, 1U> XY;
  coder::array<real_T, 1U> r1;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T d;
  real_T varargin_1;
  int32_T b_XX[2];
  int32_T iv[2];
  int32_T b_is_reject;
  int32_T i;
  int32_T i1;
  int32_T nx;
  int32_T sizes_idx_1;
  int32_T t;
  boolean_T empty_non_axis_sizes;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  Phi 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%
  //  = T-p
  //  number of columns
  //  설명변수, 3차원
  st.site = &emlrtRSI;
  b_st.site = &p_emlrtRSI;
  varargin_1 =
      p * (static_cast<real_T>(Y0.size(1)) * static_cast<real_T>(Y0.size(1)));
  d = p * (static_cast<real_T>(Y0.size(1)) * static_cast<real_T>(Y0.size(1)));
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(d, &f_emlrtDCI, (emlrtCTX)sp);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &e_emlrtDCI, (emlrtCTX)sp);
  }
  XX.set_size(&u_emlrtRTEI, sp, static_cast<int32_T>(varargin_1),
              static_cast<int32_T>(varargin_1));
  sizes_idx_1 =
      static_cast<int32_T>(varargin_1) * static_cast<int32_T>(varargin_1);
  for (i = 0; i < sizes_idx_1; i++) {
    XX[i] = 0.0;
  }
  st.site = &b_emlrtRSI;
  b_st.site = &p_emlrtRSI;
  varargin_1 =
      p * (static_cast<real_T>(Y0.size(1)) * static_cast<real_T>(Y0.size(1)));
  if (!(varargin_1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(varargin_1, &h_emlrtDCI, (emlrtCTX)sp);
  }
  d = static_cast<int32_T>(muDoubleScalarFloor(varargin_1));
  if (varargin_1 != d) {
    emlrtIntegerCheckR2012b(varargin_1, &g_emlrtDCI, (emlrtCTX)sp);
  }
  XY.set_size(&v_emlrtRTEI, sp, static_cast<int32_T>(varargin_1));
  if (varargin_1 != d) {
    emlrtIntegerCheckR2012b(varargin_1, &g_emlrtDCI, (emlrtCTX)sp);
  }
  sizes_idx_1 = static_cast<int32_T>(varargin_1);
  for (i = 0; i < sizes_idx_1; i++) {
    XY[i] = 0.0;
  }
  i = Y0.size(0);
  for (t = 0; t < i; t++) {
    if (t + 1 > YLm.size(2)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, YLm.size(2), &b_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    st.site = &c_emlrtRSI;
    sizes_idx_1 = YLm.size(0);
    nx = YLm.size(1);
    b_YLm.set_size(&w_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < nx; i1++) {
      for (b_is_reject = 0; b_is_reject < sizes_idx_1; b_is_reject++) {
        b_YLm[b_is_reject + b_YLm.size(0) * i1] =
            YLm[(b_is_reject + YLm.size(0) * i1) +
                YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &s_emlrtRSI;
    coder::dynamic_size_checks(&b_st, b_YLm, Omega_inv, YLm.size(0),
                               Omega_inv.size(0));
    sizes_idx_1 = YLm.size(0);
    nx = YLm.size(1);
    b_YLm.set_size(&w_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < nx; i1++) {
      for (b_is_reject = 0; b_is_reject < sizes_idx_1; b_is_reject++) {
        b_YLm[b_is_reject + b_YLm.size(0) * i1] =
            YLm[(b_is_reject + YLm.size(0) * i1) +
                YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &r_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, b_YLm, Omega_inv, precb_);
    st.site = &c_emlrtRSI;
    sizes_idx_1 = YLm.size(0);
    nx = YLm.size(1);
    b_YLm.set_size(&w_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < nx; i1++) {
      for (b_is_reject = 0; b_is_reject < sizes_idx_1; b_is_reject++) {
        b_YLm[b_is_reject + b_YLm.size(0) * i1] =
            YLm[(b_is_reject + YLm.size(0) * i1) +
                YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &s_emlrtRSI;
    coder::dynamic_size_checks(&b_st, precb_, b_YLm, precb_.size(1),
                               YLm.size(0));
    sizes_idx_1 = YLm.size(0);
    nx = YLm.size(1);
    b_YLm.set_size(&w_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < nx; i1++) {
      for (b_is_reject = 0; b_is_reject < sizes_idx_1; b_is_reject++) {
        b_YLm[b_is_reject + b_YLm.size(0) * i1] =
            YLm[(b_is_reject + YLm.size(0) * i1) +
                YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &r_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, precb_, b_YLm, r);
    iv[0] = (*(int32_T(*)[2])XX.size())[0];
    iv[1] = (*(int32_T(*)[2])XX.size())[1];
    b_XX[0] = (*(int32_T(*)[2])r.size())[0];
    b_XX[1] = (*(int32_T(*)[2])r.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_XX[0], &g_emlrtECI, (emlrtCTX)sp);
    sizes_idx_1 = XX.size(0) * XX.size(1);
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      XX[i1] = XX[i1] + r[i1];
    }
    if (t + 1 > Y0.size(0)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, Y0.size(0), &emlrtBCI,
                                    (emlrtCTX)sp);
    }
    st.site = &d_emlrtRSI;
    sizes_idx_1 = YLm.size(0);
    nx = YLm.size(1);
    b_YLm.set_size(&w_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < nx; i1++) {
      for (b_is_reject = 0; b_is_reject < sizes_idx_1; b_is_reject++) {
        b_YLm[b_is_reject + b_YLm.size(0) * i1] =
            YLm[(b_is_reject + YLm.size(0) * i1) +
                YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &s_emlrtRSI;
    coder::dynamic_size_checks(&b_st, b_YLm, Omega_inv, YLm.size(0),
                               Omega_inv.size(0));
    sizes_idx_1 = YLm.size(0);
    nx = YLm.size(1);
    b_YLm.set_size(&w_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < nx; i1++) {
      for (b_is_reject = 0; b_is_reject < sizes_idx_1; b_is_reject++) {
        b_YLm[b_is_reject + b_YLm.size(0) * i1] =
            YLm[(b_is_reject + YLm.size(0) * i1) +
                YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &r_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, b_YLm, Omega_inv, precb_);
    st.site = &d_emlrtRSI;
    sizes_idx_1 = Y0.size(1);
    b_Y0.set_size(&x_emlrtRTEI, &st, 1, Y0.size(1));
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      b_Y0[i1] = Y0[t + Y0.size(0) * i1];
    }
    b_st.site = &s_emlrtRSI;
    coder::b_dynamic_size_checks(&b_st, precb_, b_Y0, precb_.size(1),
                                 Y0.size(1));
    sizes_idx_1 = Y0.size(1);
    b_Y0.set_size(&x_emlrtRTEI, &st, 1, Y0.size(1));
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      b_Y0[i1] = Y0[t + Y0.size(0) * i1];
    }
    b_st.site = &r_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, precb_, b_Y0, r1);
    sizes_idx_1 = XY.size(0);
    if (XY.size(0) != r1.size(0)) {
      emlrtSizeEqCheck1DR2012b(XY.size(0), r1.size(0), &f_emlrtECI,
                               (emlrtCTX)sp);
    }
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      XY[i1] = XY[i1] + r1[i1];
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  st.site = &e_emlrtRSI;
  invpd(&st, var_, precb_);
  iv[0] = (*(int32_T(*)[2])precb_.size())[0];
  iv[1] = (*(int32_T(*)[2])precb_.size())[1];
  b_XX[0] = (*(int32_T(*)[2])XX.size())[0];
  b_XX[1] = (*(int32_T(*)[2])XX.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_XX[0], &e_emlrtECI, (emlrtCTX)sp);
  sizes_idx_1 = precb_.size(0) * precb_.size(1);
  XX.set_size(&y_emlrtRTEI, sp, precb_.size(0), precb_.size(1));
  for (i = 0; i < sizes_idx_1; i++) {
    XX[i] = precb_[i] + XX[i];
  }
  b_XX[0] = XX.size(1);
  b_XX[1] = XX.size(0);
  iv[0] = (*(int32_T(*)[2])XX.size())[0];
  iv[1] = (*(int32_T(*)[2])XX.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_XX[0], &d_emlrtECI, (emlrtCTX)sp);
  r.set_size(&ab_emlrtRTEI, sp, XX.size(0), XX.size(1));
  sizes_idx_1 = XX.size(1);
  for (i = 0; i < sizes_idx_1; i++) {
    nx = XX.size(0);
    for (i1 = 0; i1 < nx; i1++) {
      r[i1 + r.size(0) * i] =
          0.5 * (XX[i1 + XX.size(0) * i] + XX[i + XX.size(0) * i1]);
    }
  }
  st.site = &f_emlrtRSI;
  invpd(&st, r, XX);
  b_XX[0] = XX.size(1);
  b_XX[1] = XX.size(0);
  iv[0] = (*(int32_T(*)[2])XX.size())[0];
  iv[1] = (*(int32_T(*)[2])XX.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_XX[0], &c_emlrtECI, (emlrtCTX)sp);
  r.set_size(&ab_emlrtRTEI, sp, XX.size(0), XX.size(1));
  sizes_idx_1 = XX.size(1);
  for (i = 0; i < sizes_idx_1; i++) {
    nx = XX.size(0);
    for (i1 = 0; i1 < nx; i1++) {
      r[i1 + r.size(0) * i] =
          0.5 * (XX[i1 + XX.size(0) * i] + XX[i + XX.size(0) * i1]);
    }
  }
  XX.set_size(&bb_emlrtRTEI, sp, r.size(0), r.size(1));
  sizes_idx_1 = r.size(0) * r.size(1);
  for (i = 0; i < sizes_idx_1; i++) {
    XX[i] = r[i];
  }
  st.site = &g_emlrtRSI;
  b_st.site = &s_emlrtRSI;
  coder::b_dynamic_size_checks(&b_st, precb_, b_, precb_.size(1), b_.size(0));
  b_st.site = &r_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, precb_, b_, r1);
  if (XY.size(0) != r1.size(0)) {
    emlrtSizeEqCheck1DR2012b(XY.size(0), r1.size(0), &b_emlrtECI, (emlrtCTX)sp);
  }
  sizes_idx_1 = XY.size(0);
  for (i = 0; i < sizes_idx_1; i++) {
    XY[i] = XY[i] + r1[i];
  }
  //  b_ = B0
  st.site = &h_emlrtRSI;
  b_st.site = &s_emlrtRSI;
  coder::b_dynamic_size_checks(&b_st, XX, XY, XX.size(1), XY.size(0));
  b_st.site = &r_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, XX, XY, beta);
  //  full conditional mean
  st.site = &i_emlrtRSI;
  cholmod(&st, XX, r);
  st.site = &j_emlrtRSI;
  b_st.site = &j_emlrtRSI;
  coder::randn(&b_st,
               p * static_cast<real_T>(Y0.size(1)) *
                   static_cast<real_T>(Y0.size(1)),
               XY);
  b_st.site = &s_emlrtRSI;
  coder::b_dynamic_size_checks(&b_st, r, XY, r.size(0), XY.size(0));
  b_st.site = &r_emlrtRSI;
  coder::internal::blas::b_mtimes(&b_st, r, XY, r1);
  if (beta.size(0) != r1.size(0)) {
    emlrtSizeEqCheck1DR2012b(beta.size(0), r1.size(0), &emlrtECI, (emlrtCTX)sp);
  }
  sizes_idx_1 = beta.size(0);
  for (i = 0; i < sizes_idx_1; i++) {
    beta[i] = beta[i] + r1[i];
  }
  //  beta sampling 하기
  // F 행렬만들기
  st.site = &k_emlrtRSI;
  varargin_1 = p * static_cast<real_T>(Y0.size(1));
  nx = beta.size(0);
  b_st.site = &ed_emlrtRSI;
  c_st.site = &fd_emlrtRSI;
  coder::internal::assertValidSizeArg(&c_st, varargin_1);
  c_st.site = &fd_emlrtRSI;
  coder::internal::assertValidSizeArg(&c_st, static_cast<real_T>(Y0.size(1)));
  sizes_idx_1 = beta.size(0);
  if (1 > beta.size(0)) {
    sizes_idx_1 = 1;
  }
  nx = muIntScalarMax_sint32(nx, sizes_idx_1);
  if (static_cast<int32_T>(varargin_1) > nx) {
    emlrtErrorWithMessageIdR2018a(&st, &d_emlrtRTEI,
                                  "Coder:toolbox:reshape_emptyReshapeLimit",
                                  "Coder:toolbox:reshape_emptyReshapeLimit", 0);
  }
  if (Y0.size(1) > nx) {
    emlrtErrorWithMessageIdR2018a(&st, &d_emlrtRTEI,
                                  "Coder:toolbox:reshape_emptyReshapeLimit",
                                  "Coder:toolbox:reshape_emptyReshapeLimit", 0);
  }
  if (static_cast<int32_T>(varargin_1) < 0) {
    emlrtErrorWithMessageIdR2018a(&st, &c_emlrtRTEI,
                                  "MATLAB:checkDimCommon:nonnegativeSize",
                                  "MATLAB:checkDimCommon:nonnegativeSize", 0);
  }
  if (static_cast<int32_T>(varargin_1) * Y0.size(1) != beta.size(0)) {
    emlrtErrorWithMessageIdR2018a(
        &st, &b_emlrtRTEI, "Coder:MATLAB:getReshapeDims_notSameNumel",
        "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }
  b_XX[1] = Y0.size(1);
  Phi.set_size(&cb_emlrtRTEI, sp, Y0.size(1), static_cast<int32_T>(varargin_1));
  sizes_idx_1 = static_cast<int32_T>(varargin_1);
  for (i = 0; i < sizes_idx_1; i++) {
    nx = b_XX[1];
    for (i1 = 0; i1 < nx; i1++) {
      Phi[i1 + Phi.size(0) * i] =
          beta[i + static_cast<int32_T>(varargin_1) * i1];
    }
  }
  //  p*k by k
  st.site = &l_emlrtRSI;
  b_st.site = &l_emlrtRSI;
  coder::eye(&b_st, (p - 1.0) * static_cast<real_T>(Y0.size(1)), precb_);
  varargin_1 = static_cast<real_T>(Y0.size(1)) * (p - 1.0);
  if (!(varargin_1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(varargin_1, &d_emlrtDCI, &st);
  }
  if (varargin_1 != static_cast<int32_T>(muDoubleScalarFloor(varargin_1))) {
    emlrtIntegerCheckR2012b(varargin_1, &c_emlrtDCI, &st);
  }
  b_st.site = &gd_emlrtRSI;
  if ((precb_.size(0) != 0) && (precb_.size(1) != 0)) {
    t = precb_.size(0);
  } else if ((static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) *
                                   (p - 1.0)) != 0) &&
             (Y0.size(1) != 0)) {
    t = static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0));
  } else {
    t = precb_.size(0);
    if (static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0)) >
        precb_.size(0)) {
      t = static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0));
    }
  }
  c_st.site = &hd_emlrtRSI;
  if ((precb_.size(0) != t) &&
      ((precb_.size(0) != 0) && (precb_.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0)) !=
       t) &&
      ((static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0)) !=
        0) &&
       (Y0.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (t == 0);
  if (empty_non_axis_sizes ||
      ((precb_.size(0) != 0) && (precb_.size(1) != 0))) {
    nx = precb_.size(1);
  } else {
    nx = 0;
  }
  if (empty_non_axis_sizes ||
      ((static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0)) !=
        0) &&
       (Y0.size(1) != 0))) {
    sizes_idx_1 = Y0.size(1);
  } else {
    sizes_idx_1 = 0;
  }
  XX.set_size(&db_emlrtRTEI, &b_st, t, nx + sizes_idx_1);
  for (i = 0; i < nx; i++) {
    for (i1 = 0; i1 < t; i1++) {
      XX[i1 + XX.size(0) * i] = precb_[i1 + t * i];
    }
  }
  for (i = 0; i < sizes_idx_1; i++) {
    for (i1 = 0; i1 < t; i1++) {
      XX[i1 + XX.size(0) * (i + nx)] = 0.0;
    }
  }
  st.site = &l_emlrtRSI;
  b_st.site = &gd_emlrtRSI;
  if ((Phi.size(0) != 0) && (Phi.size(1) != 0)) {
    t = Phi.size(1);
  } else if ((XX.size(0) != 0) && (XX.size(1) != 0)) {
    t = XX.size(1);
  } else {
    t = Phi.size(1);
    if (XX.size(1) > Phi.size(1)) {
      t = XX.size(1);
    }
  }
  c_st.site = &hd_emlrtRSI;
  if ((Phi.size(1) != t) && ((Phi.size(0) != 0) && (Phi.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((XX.size(1) != t) && ((XX.size(0) != 0) && (XX.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (t == 0);
  if (empty_non_axis_sizes || ((Phi.size(0) != 0) && (Phi.size(1) != 0))) {
    nx = Phi.size(0);
  } else {
    nx = 0;
  }
  if (empty_non_axis_sizes || ((XX.size(0) != 0) && (XX.size(1) != 0))) {
    sizes_idx_1 = XX.size(0);
  } else {
    sizes_idx_1 = 0;
  }
  Fm.set_size(&eb_emlrtRTEI, &b_st, nx + sizes_idx_1, t);
  for (i = 0; i < t; i++) {
    for (i1 = 0; i1 < nx; i1++) {
      Fm[i1 + Fm.size(0) * i] = Phi[i1 + nx * i];
    }
  }
  for (i = 0; i < t; i++) {
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      Fm[(i1 + nx) + Fm.size(0) * i] = XX[i1 + sizes_idx_1 * i];
    }
  }
  //  p*k by p*k
  // 안정성 확인하기
  st.site = &m_emlrtRSI;
  coder::eig(&st, Fm, eigF);
  //  eigenvlaue 계산
  b_is_reject = 0;
  st.site = &n_emlrtRSI;
  coder::b_abs(&st, eigF, r1);
  st.site = &n_emlrtRSI;
  if (maxc(&st, r1) >= 1.0) {
    //      disp(maxc(abs(eigF)))
    b_is_reject = 1;
    Phi.set_size(&fb_emlrtRTEI, sp, Phi0.size(0), Phi0.size(1));
    sizes_idx_1 = Phi0.size(0) * Phi0.size(1);
    for (i = 0; i < sizes_idx_1; i++) {
      Phi[i] = Phi0[i];
    }
    st.site = &o_emlrtRSI;
    b_st.site = &o_emlrtRSI;
    coder::eye(&b_st, (p - 1.0) * static_cast<real_T>(Y0.size(1)), precb_);
    varargin_1 = static_cast<real_T>(Y0.size(1)) * (p - 1.0);
    if (!(varargin_1 >= 0.0)) {
      emlrtNonNegativeCheckR2012b(varargin_1, &b_emlrtDCI, &st);
    }
    if (varargin_1 != static_cast<int32_T>(muDoubleScalarFloor(varargin_1))) {
      emlrtIntegerCheckR2012b(varargin_1, &emlrtDCI, &st);
    }
    b_st.site = &gd_emlrtRSI;
    if ((precb_.size(0) != 0) && (precb_.size(1) != 0)) {
      t = precb_.size(0);
    } else if ((static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) *
                                     (p - 1.0)) != 0) &&
               (Y0.size(1) != 0)) {
      t = static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0));
    } else {
      t = precb_.size(0);
      if (static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0)) >
          precb_.size(0)) {
        t = static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0));
      }
    }
    c_st.site = &hd_emlrtRSI;
    if ((precb_.size(0) != t) &&
        ((precb_.size(0) != 0) && (precb_.size(1) != 0))) {
      emlrtErrorWithMessageIdR2018a(
          &c_st, &emlrtRTEI, "MATLAB:catenate:matrixDimensionMismatch",
          "MATLAB:catenate:matrixDimensionMismatch", 0);
    }
    if ((static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0)) !=
         t) &&
        ((static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0)) !=
          0) &&
         (Y0.size(1) != 0))) {
      emlrtErrorWithMessageIdR2018a(
          &c_st, &emlrtRTEI, "MATLAB:catenate:matrixDimensionMismatch",
          "MATLAB:catenate:matrixDimensionMismatch", 0);
    }
    empty_non_axis_sizes = (t == 0);
    if (empty_non_axis_sizes ||
        ((precb_.size(0) != 0) && (precb_.size(1) != 0))) {
      nx = precb_.size(1);
    } else {
      nx = 0;
    }
    if (empty_non_axis_sizes ||
        ((static_cast<int32_T>(static_cast<real_T>(Y0.size(1)) * (p - 1.0)) !=
          0) &&
         (Y0.size(1) != 0))) {
      sizes_idx_1 = Y0.size(1);
    } else {
      sizes_idx_1 = 0;
    }
    XX.set_size(&db_emlrtRTEI, &b_st, t, nx + sizes_idx_1);
    for (i = 0; i < nx; i++) {
      for (i1 = 0; i1 < t; i1++) {
        XX[i1 + XX.size(0) * i] = precb_[i1 + t * i];
      }
    }
    for (i = 0; i < sizes_idx_1; i++) {
      for (i1 = 0; i1 < t; i1++) {
        XX[i1 + XX.size(0) * (i + nx)] = 0.0;
      }
    }
    st.site = &o_emlrtRSI;
    b_st.site = &gd_emlrtRSI;
    if ((Phi0.size(0) != 0) && (Phi0.size(1) != 0)) {
      t = Phi0.size(1);
    } else if ((XX.size(0) != 0) && (XX.size(1) != 0)) {
      t = XX.size(1);
    } else {
      t = Phi0.size(1);
      if (XX.size(1) > Phi0.size(1)) {
        t = XX.size(1);
      }
    }
    c_st.site = &hd_emlrtRSI;
    if ((Phi0.size(1) != t) && ((Phi0.size(0) != 0) && (Phi0.size(1) != 0))) {
      emlrtErrorWithMessageIdR2018a(
          &c_st, &emlrtRTEI, "MATLAB:catenate:matrixDimensionMismatch",
          "MATLAB:catenate:matrixDimensionMismatch", 0);
    }
    if ((XX.size(1) != t) && ((XX.size(0) != 0) && (XX.size(1) != 0))) {
      emlrtErrorWithMessageIdR2018a(
          &c_st, &emlrtRTEI, "MATLAB:catenate:matrixDimensionMismatch",
          "MATLAB:catenate:matrixDimensionMismatch", 0);
    }
    empty_non_axis_sizes = (t == 0);
    if (empty_non_axis_sizes || ((Phi0.size(0) != 0) && (Phi0.size(1) != 0))) {
      nx = Phi0.size(0);
    } else {
      nx = 0;
    }
    if (empty_non_axis_sizes || ((XX.size(0) != 0) && (XX.size(1) != 0))) {
      sizes_idx_1 = XX.size(0);
    } else {
      sizes_idx_1 = 0;
    }
    Fm.set_size(&gb_emlrtRTEI, &b_st, nx + sizes_idx_1, t);
    for (i = 0; i < t; i++) {
      for (i1 = 0; i1 < nx; i1++) {
        Fm[i1 + Fm.size(0) * i] = Phi0[i1 + nx * i];
      }
    }
    for (i = 0; i < t; i++) {
      for (i1 = 0; i1 < sizes_idx_1; i1++) {
        Fm[(i1 + nx) + Fm.size(0) * i] = XX[i1 + sizes_idx_1 * i];
      }
    }
  }
  *is_reject = b_is_reject;
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (Gen_Phi.cpp)
