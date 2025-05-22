//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Minnesota_prior.cpp
//
// Code generation for function 'Minnesota_prior'
//

// Include files
#include "Minnesota_prior.h"
#include "Minnesota_prior_data.h"
#include "assertValidSizeArg.h"
#include "diag.h"
#include "eml_mtimes_helper.h"
#include "eye.h"
#include "inv.h"
#include "mean.h"
#include "mtimes.h"
#include "rt_nonfinite.h"
#include "sqrt.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo emlrtRSI{
    4,                 // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo b_emlrtRSI{
    8,                 // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo c_emlrtRSI{
    9,                 // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo d_emlrtRSI{
    17,                // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo e_emlrtRSI{
    18,                // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo f_emlrtRSI{
    19,                // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo g_emlrtRSI{
    20,                // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo h_emlrtRSI{
    29,                // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo i_emlrtRSI{
    31,                // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo j_emlrtRSI{
    38,                // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo k_emlrtRSI{
    39,                // lineNo
    "Minnesota_prior", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pathName
};

static emlrtRSInfo l_emlrtRSI{
    6,         // lineNo
    "demeanc", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\demeanc.m" // pathName
};

static emlrtRSInfo gb_emlrtRSI{
    24,    // lineNo
    "cat", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRSInfo hb_emlrtRSI{
    96,         // lineNo
    "cat_impl", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRSInfo ib_emlrtRSI{
    7,     // lineNo
    "vec", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\vec.m" // pathName
};

static emlrtRSInfo jb_emlrtRSI{
    109,               // lineNo
    "computeDimsData", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\reshapeSizeChecks.m" // pathName
};

static emlrtRSInfo kb_emlrtRSI{
    29,                  // lineNo
    "reshapeSizeChecks", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\reshapeSizeChecks.m" // pathName
};

static emlrtRSInfo
    lb_emlrtRSI{
        91,                  // lineNo
        "eml_mtimes_helper", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pathName
    };

static emlrtRSInfo
    mb_emlrtRSI{
        60,                  // lineNo
        "eml_mtimes_helper", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pathName
    };

static emlrtRSInfo nc_emlrtRSI{
    44,       // lineNo
    "mpower", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\mpower.m" // pathName
};

static emlrtRSInfo
    oc_emlrtRSI{
        71,      // lineNo
        "power", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\power.m" // pathName
    };

static emlrtECInfo emlrtECI{
    2,         // nDims
    8,         // lineNo
    5,         // colNo
    "demeanc", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\demeanc.m" // pName
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
    52,                  // lineNo
    13,                  // colNo
    "reshapeSizeChecks", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\reshapeSizeChecks.m" // pName
};

static emlrtRTEInfo d_emlrtRTEI{
    25,                // lineNo
    9,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtECInfo b_emlrtECI{
    2,                 // nDims
    18,                // lineNo
    9,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtECInfo c_emlrtECI{
    -1,                // nDims
    15,                // lineNo
    5,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtBCInfo emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    15,                // lineNo
    19,                // colNo
    "X",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo b_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    15,                // lineNo
    9,                 // colNo
    "X",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo c_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    15,                // lineNo
    34,                // colNo
    "y",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo d_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    15,                // lineNo
    28,                // colNo
    "y",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtDCInfo emlrtDCI{
    15,                // lineNo
    28,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtRTEInfo e_emlrtRTEI{
    14,                // lineNo
    9,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtBCInfo e_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    12,                // lineNo
    12,                // colNo
    "y",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo f_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    12,                // lineNo
    8,                 // colNo
    "y",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtDCInfo b_emlrtDCI{
    12,                // lineNo
    8,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtRTEInfo f_emlrtRTEI{
    102,    // lineNo
    19,     // colNo
    "diag", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pName
};

static emlrtDCInfo c_emlrtDCI{
    9,                 // lineNo
    29,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtDCInfo d_emlrtDCI{
    9,                 // lineNo
    29,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    4          // checkKind
};

static emlrtDCInfo e_emlrtDCI{
    13,                // lineNo
    11,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtDCInfo f_emlrtDCI{
    13,                // lineNo
    11,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    4          // checkKind
};

static emlrtDCInfo g_emlrtDCI{
    13,                // lineNo
    15,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtDCInfo h_emlrtDCI{
    13,                // lineNo
    15,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    4          // checkKind
};

static emlrtDCInfo i_emlrtDCI{
    24,                // lineNo
    17,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtDCInfo j_emlrtDCI{
    24,                // lineNo
    17,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    4          // checkKind
};

static emlrtDCInfo k_emlrtDCI{
    9,                 // lineNo
    21,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtDCInfo l_emlrtDCI{
    9,                 // lineNo
    21,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    4          // checkKind
};

static emlrtDCInfo m_emlrtDCI{
    13,                // lineNo
    1,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtDCInfo n_emlrtDCI{
    13,                // lineNo
    1,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    4          // checkKind
};

static emlrtDCInfo o_emlrtDCI{
    23,                // lineNo
    11,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtDCInfo p_emlrtDCI{
    23,                // lineNo
    11,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    4          // checkKind
};

static emlrtDCInfo q_emlrtDCI{
    24,                // lineNo
    1,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    1          // checkKind
};

static emlrtDCInfo r_emlrtDCI{
    24,                // lineNo
    1,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    4          // checkKind
};

static emlrtBCInfo g_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    31,                // lineNo
    47,                // colNo
    "sig",             // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo h_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    31,                // lineNo
    54,                // colNo
    "sig",             // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo i_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    31,                // lineNo
    17,                // colNo
    "v",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo j_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    29,                // lineNo
    17,                // colNo
    "v",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo k_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    33,                // lineNo
    36,                // colNo
    "v",               // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtBCInfo l_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    33,                // lineNo
    13,                // colNo
    "V_mat",           // aName
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m", // pName
    0          // checkKind
};

static emlrtRTEInfo q_emlrtRTEI{
    8,         // lineNo
    9,         // colNo
    "demeanc", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\demeanc.m" // pName
};

static emlrtRTEInfo r_emlrtRTEI{
    9,                 // lineNo
    14,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtRTEInfo s_emlrtRTEI{
    9,                 // lineNo
    1,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtRTEInfo t_emlrtRTEI{
    13,                // lineNo
    1,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtRTEInfo u_emlrtRTEI{
    12,                // lineNo
    6,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtRTEInfo v_emlrtRTEI{
    18,                // lineNo
    1,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtRTEInfo w_emlrtRTEI{
    109,    // lineNo
    24,     // colNo
    "diag", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pName
};

static emlrtRTEInfo x_emlrtRTEI{
    100,    // lineNo
    5,      // colNo
    "diag", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pName
};

static emlrtRTEInfo y_emlrtRTEI{
    23,                // lineNo
    5,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtRTEInfo ab_emlrtRTEI{
    24,                // lineNo
    1,                 // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

static emlrtRTEInfo bb_emlrtRTEI{
    38,                // lineNo
    16,                // colNo
    "Minnesota_prior", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Minnesota_"
    "prior.m" // pName
};

// Function Definitions
void Minnesota_prior(const emlrtStack *sp, coder::array<real_T, 2U> &y,
                     real_T p, real_T lambda_1, real_T lambda_2,
                     coder::array<real_T, 2U> &beta_0,
                     coder::array<real_T, 2U> &B_0,
                     coder::array<real_T, 2U> &Omega_hat)
{
  coder::array<real_T, 3U> v;
  coder::array<real_T, 2U> Phi_1;
  coder::array<real_T, 2U> X;
  coder::array<real_T, 2U> b_y;
  coder::array<real_T, 2U> meanX;
  coder::array<real_T, 2U> resid;
  coder::array<real_T, 1U> sig;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack st;
  real_T d;
  real_T rc;
  int32_T input_sizes[2];
  int32_T iv[2];
  int32_T result[2];
  int32_T b_i;
  int32_T i;
  int32_T i1;
  int32_T i2;
  int32_T i3;
  int32_T i4;
  int32_T loop_ub;
  int32_T m;
  int32_T maxdimlen;
  int32_T result_idx_1;
  int32_T sizes_idx_1;
  boolean_T empty_non_axis_sizes;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  Functions
  st.site = &emlrtRSI;
  //  demeaning vectors
  b_st.site = &l_emlrtRSI;
  coder::mean(&b_st, y, meanX);
  resid.set_size(&q_emlrtRTEI, &st, y.size(0), meanX.size(1));
  maxdimlen = meanX.size(1);
  for (i = 0; i < maxdimlen; i++) {
    sizes_idx_1 = y.size(0);
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      resid[i1 + resid.size(0) * i] = meanX[i];
    }
  }
  iv[0] = (*(int32_T(*)[2])y.size())[0];
  iv[1] = (*(int32_T(*)[2])y.size())[1];
  result[0] = (*(int32_T(*)[2])resid.size())[0];
  result[1] = (*(int32_T(*)[2])resid.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &result[0], &emlrtECI, &st);
  maxdimlen = y.size(0) * y.size(1);
  for (i = 0; i < maxdimlen; i++) {
    y[i] = y[i] - resid[i];
  }
  //  평균제거
  m = y.size(1);
  //  Step 0: Calculate prior mean
  st.site = &b_emlrtRSI;
  coder::eye(&st, static_cast<real_T>(y.size(1)), Phi_1);
  maxdimlen = Phi_1.size(0) * Phi_1.size(1);
  for (i = 0; i < maxdimlen; i++) {
    Phi_1[i] = 0.5 * Phi_1[i];
  }
  st.site = &c_emlrtRSI;
  rc = static_cast<real_T>(y.size(1)) * (p - 1.0);
  if (!(rc >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rc, &d_emlrtDCI, &st);
  }
  if (rc != static_cast<int32_T>(muDoubleScalarFloor(rc))) {
    emlrtIntegerCheckR2012b(rc, &c_emlrtDCI, &st);
  }
  rc = static_cast<real_T>(y.size(1)) * (p - 1.0);
  if (!(rc >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rc, &l_emlrtDCI, &st);
  }
  if (rc != static_cast<int32_T>(muDoubleScalarFloor(rc))) {
    emlrtIntegerCheckR2012b(rc, &k_emlrtDCI, &st);
  }
  b_st.site = &gb_emlrtRSI;
  if ((Phi_1.size(0) != 0) && (Phi_1.size(1) != 0)) {
    maxdimlen = Phi_1.size(0);
  } else if ((y.size(1) != 0) &&
             (static_cast<int32_T>(static_cast<real_T>(y.size(1)) *
                                   (p - 1.0)) != 0)) {
    maxdimlen = y.size(1);
  } else {
    maxdimlen = Phi_1.size(0);
    if (y.size(1) > Phi_1.size(0)) {
      maxdimlen = y.size(1);
    }
  }
  c_st.site = &hb_emlrtRSI;
  if ((Phi_1.size(0) != maxdimlen) &&
      ((Phi_1.size(0) != 0) && (Phi_1.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((y.size(1) != maxdimlen) &&
      ((y.size(1) != 0) &&
       (static_cast<int32_T>(static_cast<real_T>(y.size(1)) * (p - 1.0)) !=
        0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (maxdimlen == 0);
  if (empty_non_axis_sizes || ((Phi_1.size(0) != 0) && (Phi_1.size(1) != 0))) {
    input_sizes[1] = Phi_1.size(1);
  } else {
    input_sizes[1] = 0;
  }
  if (empty_non_axis_sizes ||
      ((y.size(1) != 0) &&
       (static_cast<int32_T>(static_cast<real_T>(y.size(1)) * (p - 1.0)) !=
        0))) {
    sizes_idx_1 =
        static_cast<int32_T>(static_cast<real_T>(y.size(1)) * (p - 1.0));
  } else {
    sizes_idx_1 = 0;
  }
  st.site = &c_emlrtRSI;
  result_idx_1 = input_sizes[1];
  result[1] = sizes_idx_1;
  beta_0.set_size(&r_emlrtRTEI, &st, input_sizes[1] + sizes_idx_1, maxdimlen);
  for (i = 0; i < maxdimlen; i++) {
    for (i1 = 0; i1 < result_idx_1; i1++) {
      beta_0[i1 + beta_0.size(0) * i] = Phi_1[i + maxdimlen * i1];
    }
  }
  for (i = 0; i < maxdimlen; i++) {
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      beta_0[(i1 + result_idx_1) + beta_0.size(0) * i] = 0.0;
    }
  }
  // Macro to compute unconditional variance of the factors
  rc =
      static_cast<real_T>(beta_0.size(0)) * static_cast<real_T>(beta_0.size(1));
  if (rc > 1.0) {
    b_st.site = &ib_emlrtRSI;
    sizes_idx_1 = beta_0.size(0) * beta_0.size(1);
    c_st.site = &kb_emlrtRSI;
    d_st.site = &jb_emlrtRSI;
    coder::internal::assertValidSizeArg(&d_st, rc);
    result_idx_1 = beta_0.size(0);
    if (beta_0.size(1) > beta_0.size(0)) {
      result_idx_1 = beta_0.size(1);
    }
    maxdimlen = muIntScalarMax_sint32(sizes_idx_1, result_idx_1);
    if (static_cast<int32_T>(rc) > maxdimlen) {
      emlrtErrorWithMessageIdR2018a(
          &b_st, &c_emlrtRTEI, "Coder:toolbox:reshape_emptyReshapeLimit",
          "Coder:toolbox:reshape_emptyReshapeLimit", 0);
    }
    if (1 > maxdimlen) {
      emlrtErrorWithMessageIdR2018a(
          &b_st, &c_emlrtRTEI, "Coder:toolbox:reshape_emptyReshapeLimit",
          "Coder:toolbox:reshape_emptyReshapeLimit", 0);
    }
    if (static_cast<int32_T>(rc) != sizes_idx_1) {
      emlrtErrorWithMessageIdR2018a(
          &b_st, &b_emlrtRTEI, "Coder:MATLAB:getReshapeDims_notSameNumel",
          "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
    }
    beta_0.set_size(&s_emlrtRTEI, &st, static_cast<int32_T>(rc), 1);
    maxdimlen = static_cast<int32_T>(rc);
    for (i = 0; i < 1; i++) {
      for (i1 = 0; i1 < maxdimlen; i1++) {
        beta_0[i1] = beta_0[i1];
      }
    }
  }
  //  Step 1: Run VAR to derive standard error for each equation
  if (p + 1.0 > y.size(0)) {
    i = 0;
    i1 = 0;
  } else {
    if (p + 1.0 != static_cast<int32_T>(muDoubleScalarFloor(p + 1.0))) {
      emlrtIntegerCheckR2012b(p + 1.0, &b_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(p + 1.0) < 1) ||
        (static_cast<int32_T>(p + 1.0) > y.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(p + 1.0), 1, y.size(0),
                                    &f_emlrtBCI, (emlrtCTX)sp);
    }
    i = static_cast<int32_T>(p + 1.0) - 1;
    if (y.size(0) < 1) {
      emlrtDynamicBoundsCheckR2012b(y.size(0), 1, y.size(0), &e_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    i1 = y.size(0);
  }
  rc = static_cast<real_T>(y.size(0)) - p;
  if (!(rc >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rc, &f_emlrtDCI, (emlrtCTX)sp);
  }
  if (rc != static_cast<int32_T>(muDoubleScalarFloor(rc))) {
    emlrtIntegerCheckR2012b(rc, &e_emlrtDCI, (emlrtCTX)sp);
  }
  X.set_size(&t_emlrtRTEI, sp, static_cast<int32_T>(rc), X.size(1));
  rc = static_cast<real_T>(y.size(1)) * p;
  if (!(rc >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rc, &h_emlrtDCI, (emlrtCTX)sp);
  }
  if (rc != static_cast<int32_T>(muDoubleScalarFloor(rc))) {
    emlrtIntegerCheckR2012b(rc, &g_emlrtDCI, (emlrtCTX)sp);
  }
  X.set_size(&t_emlrtRTEI, sp, X.size(0), static_cast<int32_T>(rc));
  rc = static_cast<real_T>(y.size(0)) - p;
  if (!(rc >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rc, &n_emlrtDCI, (emlrtCTX)sp);
  }
  if (rc != static_cast<int32_T>(muDoubleScalarFloor(rc))) {
    emlrtIntegerCheckR2012b(rc, &m_emlrtDCI, (emlrtCTX)sp);
  }
  d = static_cast<real_T>(y.size(1)) * p;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(d, &n_emlrtDCI, (emlrtCTX)sp);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &m_emlrtDCI, (emlrtCTX)sp);
  }
  maxdimlen = static_cast<int32_T>(rc) * static_cast<int32_T>(d);
  for (i2 = 0; i2 < maxdimlen; i2++) {
    X[i2] = 0.0;
  }
  i2 = static_cast<int32_T>(p);
  emlrtForLoopVectorCheckR2021a(1.0, 1.0, p, mxDOUBLE_CLASS,
                                static_cast<int32_T>(p), &e_emlrtRTEI,
                                (emlrtCTX)sp);
  if (0 <= static_cast<int32_T>(p) - 1) {
    result[1] = y.size(1);
    loop_ub = y.size(1);
  }
  for (b_i = 0; b_i < i2; b_i++) {
    rc = (p + 1.0) - (static_cast<real_T>(b_i) + 1.0);
    i3 = static_cast<int32_T>(static_cast<real_T>(y.size(0)) -
                              (static_cast<real_T>(b_i) + 1.0));
    if (rc > i3) {
      i4 = 0;
      i3 = 0;
    } else {
      if (rc != static_cast<int32_T>(muDoubleScalarFloor(rc))) {
        emlrtIntegerCheckR2012b(rc, &emlrtDCI, (emlrtCTX)sp);
      }
      if ((static_cast<int32_T>(rc) < 1) ||
          (static_cast<int32_T>(rc) > y.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(rc), 1, y.size(0),
                                      &d_emlrtBCI, (emlrtCTX)sp);
      }
      i4 = static_cast<int32_T>(rc) - 1;
      if ((i3 < 1) || (i3 > y.size(0))) {
        emlrtDynamicBoundsCheckR2012b(i3, 1, y.size(0), &c_emlrtBCI,
                                      (emlrtCTX)sp);
      }
    }
    rc =
        static_cast<real_T>(m) * ((static_cast<real_T>(b_i) + 1.0) - 1.0) + 1.0;
    d = static_cast<real_T>(m) * (static_cast<real_T>(b_i) + 1.0);
    if (rc > d) {
      maxdimlen = 0;
      result_idx_1 = 0;
    } else {
      if ((static_cast<int32_T>(rc) < 1) ||
          (static_cast<int32_T>(rc) > X.size(1))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(rc), 1, X.size(1),
                                      &b_emlrtBCI, (emlrtCTX)sp);
      }
      maxdimlen = static_cast<int32_T>(rc) - 1;
      if ((static_cast<int32_T>(d) < 1) ||
          (static_cast<int32_T>(d) > X.size(1))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, X.size(1),
                                      &emlrtBCI, (emlrtCTX)sp);
      }
      result_idx_1 = static_cast<int32_T>(d);
    }
    input_sizes[0] = X.size(0);
    input_sizes[1] = result_idx_1 - maxdimlen;
    sizes_idx_1 = i3 - i4;
    result[0] = sizes_idx_1;
    emlrtSubAssignSizeCheckR2012b(&input_sizes[0], 2, &result[0], 2,
                                  &c_emlrtECI, (emlrtCTX)sp);
    for (i3 = 0; i3 < loop_ub; i3++) {
      for (result_idx_1 = 0; result_idx_1 < sizes_idx_1; result_idx_1++) {
        X[result_idx_1 + X.size(0) * (maxdimlen + i3)] =
            y[(i4 + result_idx_1) + y.size(0) * i3];
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  st.site = &d_emlrtRSI;
  b_st.site = &mb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, X, X, X.size(0), X.size(0));
  b_st.site = &lb_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, X, X, resid);
  st.site = &d_emlrtRSI;
  b_st.site = &d_emlrtRSI;
  coder::inv(&b_st, resid, Phi_1);
  b_st.site = &mb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, Phi_1, X, Phi_1.size(1), X.size(1));
  b_st.site = &lb_emlrtRSI;
  coder::internal::blas::b_mtimes(&b_st, Phi_1, X, resid);
  st.site = &d_emlrtRSI;
  maxdimlen = y.size(1);
  sizes_idx_1 = i1 - i;
  b_y.set_size(&u_emlrtRTEI, &st, sizes_idx_1, y.size(1));
  for (i3 = 0; i3 < maxdimlen; i3++) {
    for (i4 = 0; i4 < sizes_idx_1; i4++) {
      b_y[i4 + b_y.size(0) * i3] = y[(i + i4) + y.size(0) * i3];
    }
  }
  b_st.site = &mb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, resid, b_y, resid.size(1), i1 - i);
  maxdimlen = y.size(1);
  b_y.set_size(&u_emlrtRTEI, &st, sizes_idx_1, y.size(1));
  for (i1 = 0; i1 < maxdimlen; i1++) {
    for (i3 = 0; i3 < sizes_idx_1; i3++) {
      b_y[i3 + b_y.size(0) * i1] = y[(i + i3) + y.size(0) * i1];
    }
  }
  b_st.site = &lb_emlrtRSI;
  coder::internal::blas::c_mtimes(&b_st, resid, b_y, Phi_1);
  st.site = &e_emlrtRSI;
  b_st.site = &mb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, X, Phi_1, X.size(1), Phi_1.size(0));
  b_st.site = &lb_emlrtRSI;
  coder::internal::blas::c_mtimes(&b_st, X, Phi_1, resid);
  result[0] = sizes_idx_1;
  result[1] = y.size(1);
  iv[0] = (*(int32_T(*)[2])resid.size())[0];
  iv[1] = (*(int32_T(*)[2])resid.size())[1];
  emlrtSizeEqCheckNDR2012b(&result[0], &iv[0], &b_emlrtECI, (emlrtCTX)sp);
  maxdimlen = y.size(1);
  resid.set_size(&v_emlrtRTEI, sp, sizes_idx_1, y.size(1));
  for (i1 = 0; i1 < maxdimlen; i1++) {
    for (i3 = 0; i3 < sizes_idx_1; i3++) {
      resid[i3 + resid.size(0) * i1] =
          y[(i + i3) + y.size(0) * i1] - resid[i3 + resid.size(0) * i1];
    }
  }
  st.site = &f_emlrtRSI;
  b_st.site = &mb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, resid, resid, resid.size(0), resid.size(0));
  b_st.site = &lb_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, resid, resid, Omega_hat);
  maxdimlen = Omega_hat.size(0) * Omega_hat.size(1);
  rc = static_cast<real_T>(y.size(0)) - p;
  for (i = 0; i < maxdimlen; i++) {
    Omega_hat[i] = Omega_hat[i] / rc;
  }
  st.site = &g_emlrtRSI;
  if ((Omega_hat.size(0) == 1) && (Omega_hat.size(1) == 1)) {
    sig.set_size(&x_emlrtRTEI, &st, 1);
    sig[0] = Omega_hat[0];
  } else {
    if ((Omega_hat.size(0) == 1) || (Omega_hat.size(1) == 1)) {
      emlrtErrorWithMessageIdR2018a(
          &st, &f_emlrtRTEI, "Coder:toolbox:diag_varsizedMatrixVector",
          "Coder:toolbox:diag_varsizedMatrixVector", 0);
    }
    maxdimlen = Omega_hat.size(0);
    result_idx_1 = Omega_hat.size(1);
    if (0 < Omega_hat.size(1)) {
      maxdimlen = muIntScalarMin_sint32(maxdimlen, result_idx_1);
    } else {
      maxdimlen = 0;
    }
    sig.set_size(&w_emlrtRTEI, &st, maxdimlen);
    i = maxdimlen - 1;
    for (maxdimlen = 0; maxdimlen <= i; maxdimlen++) {
      sig[maxdimlen] = Omega_hat[maxdimlen + Omega_hat.size(0) * maxdimlen];
    }
  }
  st.site = &g_emlrtRSI;
  coder::b_sqrt(&st, sig);
  //  Step 2: Calculate standard error
  if (!(p >= 0.0)) {
    emlrtNonNegativeCheckR2012b(p, &p_emlrtDCI, (emlrtCTX)sp);
  }
  if (p != static_cast<int32_T>(muDoubleScalarFloor(p))) {
    emlrtIntegerCheckR2012b(p, &o_emlrtDCI, (emlrtCTX)sp);
  }
  v.set_size(&y_emlrtRTEI, sp, i2, y.size(1), y.size(1));
  Phi_1.set_size(&ab_emlrtRTEI, sp, y.size(1), Phi_1.size(1));
  rc = static_cast<real_T>(y.size(1)) * p;
  if (!(rc >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &j_emlrtDCI, (emlrtCTX)sp);
  }
  if (rc != static_cast<int32_T>(muDoubleScalarFloor(rc))) {
    emlrtIntegerCheckR2012b(rc, &i_emlrtDCI, (emlrtCTX)sp);
  }
  Phi_1.set_size(&ab_emlrtRTEI, sp, Phi_1.size(0), static_cast<int32_T>(rc));
  rc = static_cast<real_T>(y.size(1)) * p;
  if (!(rc >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &r_emlrtDCI, (emlrtCTX)sp);
  }
  if (rc != static_cast<int32_T>(muDoubleScalarFloor(rc))) {
    emlrtIntegerCheckR2012b(rc, &q_emlrtDCI, (emlrtCTX)sp);
  }
  maxdimlen = y.size(1) * static_cast<int32_T>(rc);
  for (i = 0; i < maxdimlen; i++) {
    Phi_1[i] = 0.0;
  }
  emlrtForLoopVectorCheckR2021a(1.0, 1.0, p, mxDOUBLE_CLASS,
                                static_cast<int32_T>(p), &d_emlrtRTEI,
                                (emlrtCTX)sp);
  for (maxdimlen = 0; maxdimlen < i2; maxdimlen++) {
    for (b_i = 0; b_i < m; b_i++) {
      for (sizes_idx_1 = 0; sizes_idx_1 < m; sizes_idx_1++) {
        if (b_i == sizes_idx_1) {
          st.site = &h_emlrtRSI;
          b_st.site = &nc_emlrtRSI;
          c_st.site = &oc_emlrtRSI;
          if ((static_cast<int32_T>(maxdimlen + 1U) < 1) ||
              (static_cast<int32_T>(maxdimlen + 1U) > v.size(0))) {
            emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(maxdimlen + 1U),
                                          1, v.size(0), &j_emlrtBCI,
                                          (emlrtCTX)sp);
          }
          if (b_i + 1 > v.size(1)) {
            emlrtDynamicBoundsCheckR2012b(b_i + 1, 1, v.size(1), &j_emlrtBCI,
                                          (emlrtCTX)sp);
          }
          if (sizes_idx_1 + 1 > v.size(2)) {
            emlrtDynamicBoundsCheckR2012b(sizes_idx_1 + 1, 1, v.size(2),
                                          &j_emlrtBCI, (emlrtCTX)sp);
          }
          v[(maxdimlen + v.size(0) * b_i) +
            v.size(0) * v.size(1) * sizes_idx_1] =
              lambda_1 / ((static_cast<real_T>(maxdimlen) + 1.0) *
                          (static_cast<real_T>(maxdimlen) + 1.0));
        } else {
          st.site = &i_emlrtRSI;
          b_st.site = &nc_emlrtRSI;
          c_st.site = &oc_emlrtRSI;
          if (b_i + 1 > sig.size(0)) {
            emlrtDynamicBoundsCheckR2012b(b_i + 1, 1, sig.size(0), &g_emlrtBCI,
                                          (emlrtCTX)sp);
          }
          if (sizes_idx_1 + 1 > sig.size(0)) {
            emlrtDynamicBoundsCheckR2012b(sizes_idx_1 + 1, 1, sig.size(0),
                                          &h_emlrtBCI, (emlrtCTX)sp);
          }
          if ((static_cast<int32_T>(maxdimlen + 1U) < 1) ||
              (static_cast<int32_T>(maxdimlen + 1U) > v.size(0))) {
            emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(maxdimlen + 1U),
                                          1, v.size(0), &i_emlrtBCI,
                                          (emlrtCTX)sp);
          }
          if (b_i + 1 > v.size(1)) {
            emlrtDynamicBoundsCheckR2012b(b_i + 1, 1, v.size(1), &i_emlrtBCI,
                                          (emlrtCTX)sp);
          }
          if (sizes_idx_1 + 1 > v.size(2)) {
            emlrtDynamicBoundsCheckR2012b(sizes_idx_1 + 1, 1, v.size(2),
                                          &i_emlrtBCI, (emlrtCTX)sp);
          }
          v[(maxdimlen + v.size(0) * b_i) +
            v.size(0) * v.size(1) * sizes_idx_1] =
              lambda_1 * lambda_2 * (sig[b_i] / sig[sizes_idx_1]) /
              ((static_cast<real_T>(maxdimlen) + 1.0) *
               (static_cast<real_T>(maxdimlen) + 1.0));
        }
        if ((static_cast<int32_T>(maxdimlen + 1U) < 1) ||
            (static_cast<int32_T>(maxdimlen + 1U) > v.size(0))) {
          emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(maxdimlen + 1U), 1,
                                        v.size(0), &k_emlrtBCI, (emlrtCTX)sp);
        }
        if (b_i + 1 > v.size(1)) {
          emlrtDynamicBoundsCheckR2012b(b_i + 1, 1, v.size(1), &k_emlrtBCI,
                                        (emlrtCTX)sp);
        }
        if (sizes_idx_1 + 1 > v.size(2)) {
          emlrtDynamicBoundsCheckR2012b(sizes_idx_1 + 1, 1, v.size(2),
                                        &k_emlrtBCI, (emlrtCTX)sp);
        }
        if (b_i + 1 > Phi_1.size(0)) {
          emlrtDynamicBoundsCheckR2012b(b_i + 1, 1, Phi_1.size(0), &l_emlrtBCI,
                                        (emlrtCTX)sp);
        }
        i = static_cast<int32_T>(
            (static_cast<real_T>(sizes_idx_1) + 1.0) +
            static_cast<real_T>(m) *
                ((static_cast<real_T>(maxdimlen) + 1.0) - 1.0));
        if ((i < 1) || (i > Phi_1.size(1))) {
          emlrtDynamicBoundsCheckR2012b(i, 1, Phi_1.size(1), &l_emlrtBCI,
                                        (emlrtCTX)sp);
        }
        Phi_1[b_i + Phi_1.size(0) * (i - 1)] =
            v[(maxdimlen + v.size(0) * b_i) +
              v.size(0) * v.size(1) * sizes_idx_1];
        if (*emlrtBreakCheckR2012bFlagVar != 0) {
          emlrtBreakCheckR2012b((emlrtCTX)sp);
        }
      }
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b((emlrtCTX)sp);
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  resid.set_size(&bb_emlrtRTEI, sp, Phi_1.size(1), Phi_1.size(0));
  maxdimlen = Phi_1.size(0);
  for (i = 0; i < maxdimlen; i++) {
    sizes_idx_1 = Phi_1.size(1);
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      resid[i1 + resid.size(0) * i] = Phi_1[i + Phi_1.size(0) * i1];
    }
  }
  st.site = &j_emlrtRSI;
  b_st.site = &nc_emlrtRSI;
  c_st.site = &oc_emlrtRSI;
  st.site = &j_emlrtRSI;
  rc = static_cast<real_T>(y.size(1)) * static_cast<real_T>(y.size(1)) * p;
  sizes_idx_1 = resid.size(0) * resid.size(1);
  b_st.site = &kb_emlrtRSI;
  c_st.site = &jb_emlrtRSI;
  coder::internal::assertValidSizeArg(&c_st, rc);
  result_idx_1 = resid.size(0);
  if (resid.size(1) > resid.size(0)) {
    result_idx_1 = resid.size(1);
  }
  maxdimlen = muIntScalarMax_sint32(sizes_idx_1, result_idx_1);
  if (static_cast<int32_T>(rc) > maxdimlen) {
    emlrtErrorWithMessageIdR2018a(&st, &c_emlrtRTEI,
                                  "Coder:toolbox:reshape_emptyReshapeLimit",
                                  "Coder:toolbox:reshape_emptyReshapeLimit", 0);
  }
  if (1 > maxdimlen) {
    emlrtErrorWithMessageIdR2018a(&st, &c_emlrtRTEI,
                                  "Coder:toolbox:reshape_emptyReshapeLimit",
                                  "Coder:toolbox:reshape_emptyReshapeLimit", 0);
  }
  if (static_cast<int32_T>(rc) != sizes_idx_1) {
    emlrtErrorWithMessageIdR2018a(
        &st, &b_emlrtRTEI, "Coder:MATLAB:getReshapeDims_notSameNumel",
        "Coder:MATLAB:getReshapeDims_notSameNumel", 0);
  }
  maxdimlen = static_cast<int32_T>(rc);
  sig = resid.reshape(maxdimlen);
  st.site = &k_emlrtRSI;
  coder::diag(&st, sig, B_0);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (Minnesota_prior.cpp)
