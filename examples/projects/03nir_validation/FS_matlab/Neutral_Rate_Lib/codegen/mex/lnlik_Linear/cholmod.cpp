//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// cholmod.cpp
//
// Code generation for function 'cholmod'
//

// Include files
#include "cholmod.h"
#include "abs.h"
#include "diag.h"
#include "eml_int_forloop_overflow_check.h"
#include "eml_mtimes_helper.h"
#include "error.h"
#include "eye.h"
#include "lnlik_Linear_data.h"
#include "mtimes.h"
#include "rt_nonfinite.h"
#include "sum.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo o_emlrtRSI{
    29,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo p_emlrtRSI{
    31,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo q_emlrtRSI{
    37,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo r_emlrtRSI{
    38,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo s_emlrtRSI{
    39,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo t_emlrtRSI{
    40,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo u_emlrtRSI{
    41,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo v_emlrtRSI{
    50,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo w_emlrtRSI{
    59,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo x_emlrtRSI{
    61,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo y_emlrtRSI{
    62,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo ab_emlrtRSI{
    63,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo bb_emlrtRSI{
    66,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo cb_emlrtRSI{
    69,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo db_emlrtRSI{
    77,        // lineNo
    "cholmod", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pathName
};

static emlrtRSInfo vb_emlrtRSI{
    15,    // lineNo
    "max", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\max.m" // pathName
};

static emlrtRSInfo
    wb_emlrtRSI{
        44,         // lineNo
        "minOrMax", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\minOrMax."
        "m" // pathName
    };

static emlrtRSInfo
    xb_emlrtRSI{
        79,        // lineNo
        "maximum", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\minOrMax."
        "m" // pathName
    };

static emlrtRSInfo yb_emlrtRSI{
    175,             // lineNo
    "unaryMinOrMax", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pathName
};

static emlrtRSInfo ac_emlrtRSI{
    871,                    // lineNo
    "maxRealVectorOmitNaN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pathName
};

static emlrtRSInfo bc_emlrtRSI{
    62,                      // lineNo
    "vectorMinOrMaxInPlace", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\vectorMinOrMaxInPlace.m" // pathName
};

static emlrtRSInfo cc_emlrtRSI{
    54,                      // lineNo
    "vectorMinOrMaxInPlace", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\vectorMinOrMaxInPlace.m" // pathName
};

static emlrtRSInfo dc_emlrtRSI{
    103,         // lineNo
    "findFirst", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\vectorMinOrMaxInPlace.m" // pathName
};

static emlrtRSInfo ec_emlrtRSI{
    120,                        // lineNo
    "minOrMaxRealVectorKernel", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\vectorMinOrMaxInPlace.m" // pathName
};

static emlrtRSInfo gc_emlrtRSI{
    197,             // lineNo
    "unaryMinOrMax", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pathName
};

static emlrtRSInfo hc_emlrtRSI{
    288,                     // lineNo
    "unaryMinOrMaxDispatch", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pathName
};

static emlrtRSInfo ic_emlrtRSI{
    356,          // lineNo
    "minOrMax2D", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pathName
};

static emlrtRSInfo jc_emlrtRSI{
    438,                         // lineNo
    "minOrMax2DColumnMajorDim1", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pathName
};

static emlrtRSInfo kc_emlrtRSI{
    436,                         // lineNo
    "minOrMax2DColumnMajorDim1", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pathName
};

static emlrtRSInfo
    oc_emlrtRSI{
        28,      // lineNo
        "colon", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m" // pathName
    };

static emlrtRTEInfo h_emlrtRTEI{
    124,             // lineNo
    27,              // colNo
    "unaryMinOrMax", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pName
};

static emlrtRTEInfo i_emlrtRTEI{
    26,              // lineNo
    27,              // colNo
    "unaryMinOrMax", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pName
};

static emlrtECInfo f_emlrtECI{
    -1,        // nDims
    77,        // lineNo
    5,         // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtBCInfo n_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    77,        // lineNo
    9,         // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo o_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    77,        // lineNo
    28,        // colNo
    "d",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo p_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    71,        // lineNo
    10,        // colNo
    "d",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtECInfo g_emlrtECI{
    -1,        // nDims
    67,        // lineNo
    9,         // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtBCInfo q_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    67,        // lineNo
    13,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtECInfo h_emlrtECI{
    -1,        // nDims
    62,        // lineNo
    16,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtECInfo i_emlrtECI{
    -1,        // nDims
    62,        // lineNo
    33,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtBCInfo r_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    62,        // lineNo
    43,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo s_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    62,        // lineNo
    20,        // colNo
    "A",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtECInfo j_emlrtECI{
    -1,        // nDims
    59,        // lineNo
    31,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtBCInfo t_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    59,        // lineNo
    41,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo u_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    59,        // lineNo
    25,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtECInfo k_emlrtECI{
    2,         // nDims
    39,        // lineNo
    18,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtECInfo l_emlrtECI{
    2,         // nDims
    29,        // lineNo
    16,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtBCInfo v_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    59,        // lineNo
    33,        // colNo
    "d",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo w_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    59,        // lineNo
    43,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo x_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    59,        // lineNo
    27,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo y_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    59,        // lineNo
    14,        // colNo
    "A",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo ab_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    62,        // lineNo
    35,        // colNo
    "d",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo bb_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    69,        // lineNo
    9,         // colNo
    "d",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo cb_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    62,        // lineNo
    45,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtDCInfo c_emlrtDCI{
    62,        // lineNo
    18,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    1 // checkKind
};

static emlrtBCInfo db_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    62,        // lineNo
    18,        // colNo
    "A",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo eb_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    62,        // lineNo
    27,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo fb_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    62,        // lineNo
    29,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo gb_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    66,        // lineNo
    9,         // colNo
    "d",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo hb_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    67,        // lineNo
    11,        // colNo
    "L",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtBCInfo ib_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    67,        // lineNo
    23,        // colNo
    "d",       // aName
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m", // pName
    0 // checkKind
};

static emlrtRTEInfo y_emlrtRTEI{
    29,        // lineNo
    16,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo ab_emlrtRTEI{
    39,        // lineNo
    18,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo bb_emlrtRTEI{
    428,             // lineNo
    21,              // colNo
    "unaryMinOrMax", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pName
};

static emlrtRTEInfo cb_emlrtRTEI{
    175,             // lineNo
    38,              // colNo
    "unaryMinOrMax", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pName
};

static emlrtRTEInfo db_emlrtRTEI{
    46,        // lineNo
    1,         // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo eb_emlrtRTEI{
    57,        // lineNo
    5,         // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo fb_emlrtRTEI{
    59,        // lineNo
    33,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo gb_emlrtRTEI{
    59,        // lineNo
    39,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo hb_emlrtRTEI{
    59,        // lineNo
    23,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo ib_emlrtRTEI{
    59,        // lineNo
    31,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo jb_emlrtRTEI{
    61,        // lineNo
    9,         // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo kb_emlrtRTEI{
    62,        // lineNo
    41,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo lb_emlrtRTEI{
    62,        // lineNo
    18,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo mb_emlrtRTEI{
    62,        // lineNo
    25,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo nb_emlrtRTEI{
    62,        // lineNo
    33,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo ob_emlrtRTEI{
    62,        // lineNo
    9,         // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo pb_emlrtRTEI{
    67,        // lineNo
    11,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo qb_emlrtRTEI{
    77,        // lineNo
    14,        // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

static emlrtRTEInfo rb_emlrtRTEI{
    79,        // lineNo
    1,         // colNo
    "cholmod", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\cholmod.m" // pName
};

// Function Definitions
void cholmod(const emlrtStack *sp, const coder::array<real_T, 2U> &A,
             coder::array<real_T, 2U> &R, real_T *err)
{
  coder::array<real_T, 2U> K;
  coder::array<real_T, 2U> L;
  coder::array<real_T, 2U> varargin_1;
  coder::array<real_T, 1U> b_d;
  coder::array<real_T, 1U> b_varargin_1;
  coder::array<real_T, 1U> diagA;
  coder::array<real_T, 1U> r;
  coder::array<real_T, 1U> r1;
  coder::array<int32_T, 1U> r2;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack g_st;
  emlrtStack h_st;
  emlrtStack i_st;
  emlrtStack st;
  real_T c_varargin_1[3];
  real_T b_gamma;
  real_T d;
  real_T delta;
  real_T theta;
  real_T xi;
  int32_T b_A[2];
  int32_T iv[2];
  int32_T a;
  int32_T i;
  int32_T i1;
  int32_T idx;
  int32_T j;
  int32_T k;
  int32_T last;
  int32_T n;
  boolean_T exitg1;
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
  i_st.prev = &h_st;
  i_st.tls = h_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  function [R, indef, E, err] = cholmod(A)
  //  CHOLMOD Modified Cholesky factorization
  //   R = cholmod(A) returns the upper Cholesky factor of A (same as CHOL)
  //   if A is (sufficiently) positive definite, and otherwise returns a
  //   modified factor R with diagonal enries >= sqrt(delta) and
  //   offdiagonal entries <= beta in absolute value,
  //   where delta and beta are defined in terms of size of diagonal and
  //   offdiagonal entries of A and the machine precision; see below.
  //   The idea is to ensure that E = A - R'*R is reasonably small if A is
  //   not too far from being indefinite.  If A is sparse, so is R.
  //   The output parameter indef is set to 0 if A is sufficiently positive
  //   definite and to 1 if the factorization is modified.
  //
  //   The point of modified Cholesky is to avoid computing eigenvalues,
  //   but for dense matrices, MODCHOL takes longer than calling the built-in
  //   EIG, because of the cost of interpreting the code, even though it
  //   only has one loop and uses vector operations.
  //   reference: Nocedal and Wright, Algorithm 3.4 and subsequent discussion
  //   (not Algorithm 3.5, which is more complicated)
  //   original algorithm is due to Gill and Murray, 1974
  //   written by M. Overton, overton@cs.nyu.edu, last modified Feb 2005
  //   convenient to work with A = LDL' where D is diagonal, L is unit
  //   lower triangular, and so R = (LD^(1/2))'
  //
  b_A[0] = A.size(1);
  b_A[1] = A.size(0);
  iv[0] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&A)->size())[0];
  iv[1] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&A)->size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_A[0], &l_emlrtECI, (emlrtCTX)sp);
  L.set_size(&y_emlrtRTEI, sp, A.size(0), A.size(1));
  last = A.size(1);
  for (i = 0; i < last; i++) {
    idx = A.size(0);
    for (i1 = 0; i1 < idx; i1++) {
      L[i1 + L.size(0) * i] = A[i1 + A.size(0) * i] - A[i + A.size(0) * i1];
    }
  }
  st.site = &o_emlrtRSI;
  coder::b_abs(&st, L, varargin_1);
  st.site = &o_emlrtRSI;
  coder::sum(&st, varargin_1, K);
  st.site = &o_emlrtRSI;
  if (coder::sum(&st, K) > 0.0) {
    st.site = &p_emlrtRSI;
    coder::b_error(&st);
  }
  //  set parameters governing bounds on L and D (eps is machine epsilon)
  if ((A.size(0) == 0) || (A.size(1) == 0)) {
    n = 0;
  } else {
    i = A.size(0);
    i1 = A.size(1);
    n = muIntScalarMax_sint32(i, i1);
  }
  st.site = &q_emlrtRSI;
  coder::diag(&st, A, diagA);
  st.site = &r_emlrtRSI;
  b_st.site = &r_emlrtRSI;
  coder::b_abs(&b_st, diagA, b_varargin_1);
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  if (b_varargin_1.size(0) < 1) {
    emlrtErrorWithMessageIdR2018a(&d_st, &h_emlrtRTEI,
                                  "Coder:toolbox:eml_min_or_max_varDimZero",
                                  "Coder:toolbox:eml_min_or_max_varDimZero", 0);
  }
  e_st.site = &yb_emlrtRSI;
  f_st.site = &ac_emlrtRSI;
  last = b_varargin_1.size(0);
  if (b_varargin_1.size(0) <= 2) {
    if (b_varargin_1.size(0) == 1) {
      b_gamma = b_varargin_1[0];
    } else if ((b_varargin_1[0] < b_varargin_1[1]) ||
               (muDoubleScalarIsNaN(b_varargin_1[0]) &&
                (!muDoubleScalarIsNaN(b_varargin_1[1])))) {
      b_gamma = b_varargin_1[1];
    } else {
      b_gamma = b_varargin_1[0];
    }
  } else {
    g_st.site = &cc_emlrtRSI;
    if (!muDoubleScalarIsNaN(b_varargin_1[0])) {
      idx = 1;
    } else {
      idx = 0;
      h_st.site = &dc_emlrtRSI;
      if (b_varargin_1.size(0) > 2147483646) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= last)) {
        if (!muDoubleScalarIsNaN(b_varargin_1[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }
    if (idx == 0) {
      b_gamma = b_varargin_1[0];
    } else {
      g_st.site = &bc_emlrtRSI;
      b_gamma = b_varargin_1[idx - 1];
      a = idx + 1;
      h_st.site = &ec_emlrtRSI;
      if ((idx + 1 <= b_varargin_1.size(0)) &&
          (b_varargin_1.size(0) > 2147483646)) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      for (k = a; k <= last; k++) {
        d = b_varargin_1[k - 1];
        if (b_gamma < d) {
          b_gamma = d;
        }
      }
    }
  }
  //  max diagonal entry
  st.site = &s_emlrtRSI;
  coder::diag(&st, diagA, varargin_1);
  iv[0] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&A)->size())[0];
  iv[1] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&A)->size())[1];
  b_A[0] = (*(int32_T(*)[2])varargin_1.size())[0];
  b_A[1] = (*(int32_T(*)[2])varargin_1.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_A[0], &k_emlrtECI, (emlrtCTX)sp);
  st.site = &s_emlrtRSI;
  L.set_size(&ab_emlrtRTEI, &st, A.size(0), A.size(1));
  last = A.size(0) * A.size(1);
  for (i = 0; i < last; i++) {
    L[i] = A[i] - varargin_1[i];
  }
  b_st.site = &s_emlrtRSI;
  coder::b_abs(&b_st, L, varargin_1);
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  if ((varargin_1.size(0) == 1) && (varargin_1.size(1) != 1)) {
    emlrtErrorWithMessageIdR2018a(&d_st, &i_emlrtRTEI,
                                  "Coder:toolbox:autoDimIncompatibility",
                                  "Coder:toolbox:autoDimIncompatibility", 0);
  }
  if (varargin_1.size(0) < 1) {
    emlrtErrorWithMessageIdR2018a(&d_st, &h_emlrtRTEI,
                                  "Coder:toolbox:eml_min_or_max_varDimZero",
                                  "Coder:toolbox:eml_min_or_max_varDimZero", 0);
  }
  e_st.site = &gc_emlrtRSI;
  f_st.site = &hc_emlrtRSI;
  g_st.site = &ic_emlrtRSI;
  last = varargin_1.size(0);
  idx = varargin_1.size(1);
  K.set_size(&bb_emlrtRTEI, &g_st, 1, varargin_1.size(1));
  if (varargin_1.size(1) >= 1) {
    h_st.site = &kc_emlrtRSI;
    if (varargin_1.size(1) > 2147483646) {
      i_st.site = &j_emlrtRSI;
      coder::check_forloop_overflow_error(&i_st);
    }
    for (j = 0; j < idx; j++) {
      K[j] = varargin_1[varargin_1.size(0) * j];
      h_st.site = &jc_emlrtRSI;
      if ((2 <= last) && (last > 2147483646)) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      for (a = 2; a <= last; a++) {
        boolean_T p;
        xi = K[j];
        theta = varargin_1[(a + varargin_1.size(0) * j) - 1];
        if (muDoubleScalarIsNaN(theta)) {
          p = false;
        } else if (muDoubleScalarIsNaN(xi)) {
          p = true;
        } else {
          p = (xi < theta);
        }
        if (p) {
          K[j] = theta;
        }
      }
    }
  }
  st.site = &s_emlrtRSI;
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  if (K.size(1) < 1) {
    emlrtErrorWithMessageIdR2018a(&d_st, &h_emlrtRTEI,
                                  "Coder:toolbox:eml_min_or_max_varDimZero",
                                  "Coder:toolbox:eml_min_or_max_varDimZero", 0);
  }
  e_st.site = &yb_emlrtRSI;
  f_st.site = &ac_emlrtRSI;
  last = K.size(1);
  if (K.size(1) <= 2) {
    if (K.size(1) == 1) {
      xi = K[0];
    } else if ((K[0] < K[1]) ||
               (muDoubleScalarIsNaN(K[0]) && (!muDoubleScalarIsNaN(K[1])))) {
      xi = K[1];
    } else {
      xi = K[0];
    }
  } else {
    g_st.site = &cc_emlrtRSI;
    if (!muDoubleScalarIsNaN(K[0])) {
      idx = 1;
    } else {
      idx = 0;
      h_st.site = &dc_emlrtRSI;
      if (K.size(1) > 2147483646) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= last)) {
        if (!muDoubleScalarIsNaN(K[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }
    if (idx == 0) {
      xi = K[0];
    } else {
      g_st.site = &bc_emlrtRSI;
      xi = K[idx - 1];
      a = idx + 1;
      h_st.site = &ec_emlrtRSI;
      if ((idx + 1 <= K.size(1)) && (K.size(1) > 2147483646)) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      for (k = a; k <= last; k++) {
        d = K[k - 1];
        if (xi < d) {
          xi = d;
        }
      }
    }
  }
  //  max off-diagonal entry
  st.site = &t_emlrtRSI;
  theta = b_gamma + xi;
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  e_st.site = &yb_emlrtRSI;
  f_st.site = &ac_emlrtRSI;
  if ((theta < 1.0) || muDoubleScalarIsNaN(theta)) {
    theta = 1.0;
  }
  delta = 2.2204460492503131E-16 * theta;
  st.site = &u_emlrtRSI;
  c_varargin_1[0] = b_gamma;
  theta = xi / static_cast<real_T>(n);
  c_varargin_1[1] = theta;
  c_varargin_1[2] = 2.2204460492503131E-16;
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  e_st.site = &yb_emlrtRSI;
  K.set_size(&cb_emlrtRTEI, &e_st, 1, 3);
  K[0] = b_gamma;
  K[1] = theta;
  K[2] = 2.2204460492503131E-16;
  f_st.site = &ac_emlrtRSI;
  g_st.site = &cc_emlrtRSI;
  if (!muDoubleScalarIsNaN(K[0])) {
    idx = 1;
  } else {
    idx = 0;
    h_st.site = &dc_emlrtRSI;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k <= 3)) {
      if (!muDoubleScalarIsNaN(K[k - 1])) {
        idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (idx != 0) {
    g_st.site = &bc_emlrtRSI;
    b_gamma = c_varargin_1[idx - 1];
    a = idx + 1;
    h_st.site = &ec_emlrtRSI;
    for (k = a; k < 4; k++) {
      d = c_varargin_1[k - 1];
      if (b_gamma < d) {
        b_gamma = d;
      }
    }
  }
  st.site = &u_emlrtRSI;
  b_gamma = muDoubleScalarSqrt(b_gamma);
  //  initialize d and L
  b_d.set_size(&db_emlrtRTEI, sp, n);
  for (i = 0; i < n; i++) {
    b_d[i] = 0.0;
  }
  st.site = &v_emlrtRSI;
  coder::eye(&st, static_cast<real_T>(n), L);
  //  there are no inner for loops, everything implemented with
  //  vector operations for a reasonable level of efficiency
  for (j = 0; j < n; j++) {
    real_T djtemp;
    if (j < 1) {
      K.set_size(&eb_emlrtRTEI, sp, 1, 0);
    } else {
      K.set_size(&eb_emlrtRTEI, sp, 1, j);
      last = j - 1;
      for (i = 0; i <= last; i++) {
        K[i] = static_cast<real_T>(i) + 1.0;
      }
    }
    //  column index: all columns to left of diagonal
    //  d(K) doesn't work in case K is empty
    r.set_size(&fb_emlrtRTEI, sp, K.size(1));
    last = K.size(1);
    for (i = 0; i < last; i++) {
      r[i] = K[i];
    }
    last = r.size(0);
    for (i = 0; i < last; i++) {
      i1 = static_cast<int32_T>(r[i]);
      if (i1 > b_d.size(0)) {
        emlrtDynamicBoundsCheckR2012b(i1, 1, b_d.size(0), &v_emlrtBCI,
                                      (emlrtCTX)sp);
      }
    }
    if (j + 1 > L.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(0), &t_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    diagA.set_size(&gb_emlrtRTEI, sp, r.size(0));
    last = r.size(0);
    for (i = 0; i < last; i++) {
      i1 = static_cast<int32_T>(r[i]);
      if (i1 > L.size(1)) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r[i]), 1, L.size(1),
                                      &w_emlrtBCI, (emlrtCTX)sp);
      }
      diagA[i] = L[j + L.size(0) * (i1 - 1)];
    }
    if (r.size(0) != diagA.size(0)) {
      emlrtSizeEqCheck1DR2012b(r.size(0), diagA.size(0), &j_emlrtECI,
                               (emlrtCTX)sp);
    }
    st.site = &w_emlrtRSI;
    if (j + 1 > L.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(0), &u_emlrtBCI, &st);
    }
    K.set_size(&hb_emlrtRTEI, &st, 1, r.size(0));
    last = r.size(0);
    for (i = 0; i < last; i++) {
      i1 = static_cast<int32_T>(r[i]);
      if (i1 > L.size(1)) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r[i]), 1, L.size(1),
                                      &x_emlrtBCI, &st);
      }
      K[i] = L[j + L.size(0) * (i1 - 1)];
    }
    b_varargin_1.set_size(&ib_emlrtRTEI, &st, r.size(0));
    last = r.size(0);
    for (i = 0; i < last; i++) {
      b_varargin_1[i] = b_d[static_cast<int32_T>(r[i]) - 1] * diagA[i];
    }
    b_st.site = &h_emlrtRSI;
    coder::b_dynamic_size_checks(&b_st, K, b_varargin_1, r.size(0),
                                 b_varargin_1.size(0));
    if (j + 1 > A.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, A.size(0), &y_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (j + 1 > A.size(1)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, A.size(1), &y_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    djtemp =
        A[j + A.size(0) * j] - coder::internal::blas::mtimes(K, b_varargin_1);
    //  C(j,j) in book
    if (j + 1 < n) {
      st.site = &x_emlrtRSI;
      b_st.site = &oc_emlrtRSI;
      if (static_cast<uint32_T>(n) < j + 2U) {
        K.set_size(&jb_emlrtRTEI, &b_st, 1, 0);
      } else {
        i = n - j;
        K.set_size(&jb_emlrtRTEI, &b_st, 1, i - 1);
        last = i - 2;
        for (i = 0; i <= last; i++) {
          K[i] = (static_cast<uint32_T>(j) + i) + 2U;
        }
      }
      //  row index: all rows below diagonal
      last = r.size(0);
      for (i = 0; i < last; i++) {
        i1 = static_cast<int32_T>(r[i]);
        if (i1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(i1, 1, b_d.size(0), &ab_emlrtBCI,
                                        (emlrtCTX)sp);
        }
      }
      if (j + 1 > L.size(0)) {
        emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(0), &r_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      diagA.set_size(&kb_emlrtRTEI, sp, r.size(0));
      last = r.size(0);
      for (i = 0; i < last; i++) {
        i1 = static_cast<int32_T>(r[i]);
        if (i1 > L.size(1)) {
          emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r[i]), 1,
                                        L.size(1), &cb_emlrtBCI, (emlrtCTX)sp);
        }
        diagA[i] = L[j + L.size(0) * (i1 - 1)];
      }
      if (r.size(0) != diagA.size(0)) {
        emlrtSizeEqCheck1DR2012b(r.size(0), diagA.size(0), &i_emlrtECI,
                                 (emlrtCTX)sp);
      }
      r1.set_size(&lb_emlrtRTEI, sp, K.size(1));
      last = K.size(1);
      for (i = 0; i < last; i++) {
        r1[i] = K[i];
      }
      last = r1.size(0);
      for (i = 0; i < last; i++) {
        i1 = static_cast<int32_T>(r1[i]);
        if (r1[i] != i1) {
          emlrtIntegerCheckR2012b(r1[i], &c_emlrtDCI, (emlrtCTX)sp);
        }
        if ((i1 < 1) || (i1 > A.size(0))) {
          emlrtDynamicBoundsCheckR2012b(i1, 1, A.size(0), &db_emlrtBCI,
                                        (emlrtCTX)sp);
        }
      }
      if (j + 1 > A.size(1)) {
        emlrtDynamicBoundsCheckR2012b(j + 1, 1, A.size(1), &s_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      st.site = &y_emlrtRSI;
      varargin_1.set_size(&mb_emlrtRTEI, &st, r1.size(0), r.size(0));
      last = r.size(0);
      for (i = 0; i < last; i++) {
        idx = r1.size(0);
        for (i1 = 0; i1 < idx; i1++) {
          a = static_cast<int32_T>(r1[i1]);
          if ((a < 1) || (a > L.size(0))) {
            emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r1[i1]), 1,
                                          L.size(0), &eb_emlrtBCI, &st);
          }
          k = static_cast<int32_T>(r[i]);
          if (k > L.size(1)) {
            emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r[i]), 1,
                                          L.size(1), &fb_emlrtBCI, &st);
          }
          varargin_1[i1 + varargin_1.size(0) * i] =
              L[(a + L.size(0) * (k - 1)) - 1];
        }
      }
      b_varargin_1.set_size(&nb_emlrtRTEI, &st, r.size(0));
      last = r.size(0);
      for (i = 0; i < last; i++) {
        b_varargin_1[i] = b_d[static_cast<int32_T>(r[i]) - 1] * diagA[i];
      }
      b_st.site = &h_emlrtRSI;
      coder::dynamic_size_checks(&b_st, varargin_1, b_varargin_1, r.size(0),
                                 b_varargin_1.size(0));
      b_st.site = &g_emlrtRSI;
      coder::internal::blas::mtimes(&b_st, varargin_1, b_varargin_1, r);
      if (r1.size(0) != r.size(0)) {
        emlrtSizeEqCheck1DR2012b(r1.size(0), r.size(0), &h_emlrtECI,
                                 (emlrtCTX)sp);
      }
      diagA.set_size(&ob_emlrtRTEI, sp, r1.size(0));
      last = r1.size(0);
      for (i = 0; i < last; i++) {
        diagA[i] = A[(static_cast<int32_T>(r1[i]) + A.size(0) * j) - 1] - r[i];
      }
      //  C(I,j) in book
      st.site = &ab_emlrtRSI;
      b_st.site = &ab_emlrtRSI;
      coder::b_abs(&b_st, diagA, b_varargin_1);
      b_st.site = &vb_emlrtRSI;
      c_st.site = &wb_emlrtRSI;
      d_st.site = &xb_emlrtRSI;
      if (b_varargin_1.size(0) < 1) {
        emlrtErrorWithMessageIdR2018a(
            &d_st, &h_emlrtRTEI, "Coder:toolbox:eml_min_or_max_varDimZero",
            "Coder:toolbox:eml_min_or_max_varDimZero", 0);
      }
      e_st.site = &yb_emlrtRSI;
      f_st.site = &ac_emlrtRSI;
      last = b_varargin_1.size(0);
      if (b_varargin_1.size(0) <= 2) {
        if (b_varargin_1.size(0) == 1) {
          theta = b_varargin_1[0];
        } else if ((b_varargin_1[0] < b_varargin_1[1]) ||
                   (muDoubleScalarIsNaN(b_varargin_1[0]) &&
                    (!muDoubleScalarIsNaN(b_varargin_1[1])))) {
          theta = b_varargin_1[1];
        } else {
          theta = b_varargin_1[0];
        }
      } else {
        g_st.site = &cc_emlrtRSI;
        if (!muDoubleScalarIsNaN(b_varargin_1[0])) {
          idx = 1;
        } else {
          idx = 0;
          h_st.site = &dc_emlrtRSI;
          if (b_varargin_1.size(0) > 2147483646) {
            i_st.site = &j_emlrtRSI;
            coder::check_forloop_overflow_error(&i_st);
          }
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= last)) {
            if (!muDoubleScalarIsNaN(b_varargin_1[k - 1])) {
              idx = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }
        if (idx == 0) {
          theta = b_varargin_1[0];
        } else {
          g_st.site = &bc_emlrtRSI;
          theta = b_varargin_1[idx - 1];
          a = idx + 1;
          h_st.site = &ec_emlrtRSI;
          if ((idx + 1 <= b_varargin_1.size(0)) &&
              (b_varargin_1.size(0) > 2147483646)) {
            i_st.site = &j_emlrtRSI;
            coder::check_forloop_overflow_error(&i_st);
          }
          for (k = a; k <= last; k++) {
            d = b_varargin_1[k - 1];
            if (theta < d) {
              theta = d;
            }
          }
        }
      }
      //  guarantees d(j) not too small and L(I,j) not too big
      //  in sufficiently positive definite case, d(j) = djtemp
      st.site = &bb_emlrtRSI;
      xi = theta / b_gamma;
      b_st.site = &e_emlrtRSI;
      c_st.site = &f_emlrtRSI;
      st.site = &bb_emlrtRSI;
      c_varargin_1[0] = muDoubleScalarAbs(djtemp);
      theta = xi * xi;
      c_varargin_1[1] = theta;
      c_varargin_1[2] = delta;
      b_st.site = &vb_emlrtRSI;
      c_st.site = &wb_emlrtRSI;
      d_st.site = &xb_emlrtRSI;
      e_st.site = &yb_emlrtRSI;
      K.set_size(&cb_emlrtRTEI, &e_st, 1, 3);
      K[0] = muDoubleScalarAbs(djtemp);
      K[1] = theta;
      K[2] = delta;
      f_st.site = &ac_emlrtRSI;
      g_st.site = &cc_emlrtRSI;
      if (!muDoubleScalarIsNaN(K[0])) {
        idx = 1;
      } else {
        idx = 0;
        h_st.site = &dc_emlrtRSI;
        k = 2;
        exitg1 = false;
        while ((!exitg1) && (k <= 3)) {
          if (!muDoubleScalarIsNaN(K[k - 1])) {
            idx = k;
            exitg1 = true;
          } else {
            k++;
          }
        }
      }
      if (idx == 0) {
        if (j + 1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &gb_emlrtBCI,
                                        &f_st);
        }
        b_d[j] = muDoubleScalarAbs(djtemp);
      } else {
        g_st.site = &bc_emlrtRSI;
        theta = c_varargin_1[idx - 1];
        a = idx + 1;
        h_st.site = &ec_emlrtRSI;
        for (k = a; k < 4; k++) {
          d = c_varargin_1[k - 1];
          if (theta < d) {
            theta = d;
          }
        }
        if (j + 1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &gb_emlrtBCI,
                                        &f_st);
        }
        b_d[j] = theta;
      }
      r2.set_size(&pb_emlrtRTEI, sp, r1.size(0));
      last = r1.size(0);
      for (i = 0; i < last; i++) {
        i1 = static_cast<int32_T>(r1[i]);
        if ((i1 < 1) || (i1 > L.size(0))) {
          emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r1[i]), 1,
                                        L.size(0), &hb_emlrtBCI, (emlrtCTX)sp);
        }
        r2[i] = i1 - 1;
      }
      if (j + 1 > L.size(1)) {
        emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(1), &q_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      if (j + 1 > b_d.size(0)) {
        emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &ib_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      last = diagA.size(0);
      for (i = 0; i < last; i++) {
        diagA[i] = diagA[i] / b_d[j];
      }
      emlrtSubAssignSizeCheckR2012b(r2.size(), 1, diagA.size(), 1, &g_emlrtECI,
                                    (emlrtCTX)sp);
      last = diagA.size(0);
      for (i = 0; i < last; i++) {
        L[r2[i] + L.size(0) * j] = diagA[i];
      }
    } else {
      st.site = &cb_emlrtRSI;
      d = muDoubleScalarAbs(djtemp);
      b_st.site = &vb_emlrtRSI;
      c_st.site = &wb_emlrtRSI;
      d_st.site = &xb_emlrtRSI;
      e_st.site = &yb_emlrtRSI;
      f_st.site = &ac_emlrtRSI;
      if ((d < delta) ||
          (muDoubleScalarIsNaN(d) && (!muDoubleScalarIsNaN(delta)))) {
        if (j + 1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &bb_emlrtBCI,
                                        &f_st);
        }
        b_d[j] = delta;
      } else {
        if (j + 1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &bb_emlrtBCI,
                                        &f_st);
        }
        b_d[j] = d;
      }
    }
    if (j + 1 > b_d.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &p_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  //  convert to usual output format: replace L by L*sqrt(D) and transpose
  for (j = 0; j < n; j++) {
    if (j + 1 > L.size(1)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(1), &n_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    st.site = &db_emlrtRSI;
    if (j + 1 > b_d.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &o_emlrtBCI, &st);
    }
    theta = muDoubleScalarSqrt(b_d[j]);
    last = L.size(0);
    r.set_size(&qb_emlrtRTEI, sp, L.size(0));
    for (i = 0; i < last; i++) {
      r[i] = L[i + L.size(0) * j] * theta;
    }
    emlrtSubAssignSizeCheckR2012b(L.size(), 1, r.size(), 1, &f_emlrtECI,
                                  (emlrtCTX)sp);
    last = r.size(0);
    for (i = 0; i < last; i++) {
      L[i + L.size(0) * j] = r[i];
    }
    //  L = L*diag(sqrt(d)) bad in sparse case
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  R.set_size(&rb_emlrtRTEI, sp, L.size(1), L.size(0));
  last = L.size(0);
  for (i = 0; i < last; i++) {
    idx = L.size(1);
    for (i1 = 0; i1 < idx; i1++) {
      R[i1 + R.size(0) * i] = L[i + L.size(0) * i1];
    }
  }
  *err = 0.0;
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

void cholmod(const emlrtStack *sp, const coder::array<real_T, 2U> &A,
             coder::array<real_T, 2U> &R)
{
  coder::array<real_T, 2U> K;
  coder::array<real_T, 2U> L;
  coder::array<real_T, 2U> varargin_1;
  coder::array<real_T, 1U> b_d;
  coder::array<real_T, 1U> b_varargin_1;
  coder::array<real_T, 1U> diagA;
  coder::array<real_T, 1U> r;
  coder::array<real_T, 1U> r1;
  coder::array<int32_T, 1U> r2;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack g_st;
  emlrtStack h_st;
  emlrtStack i_st;
  emlrtStack st;
  real_T c_varargin_1[3];
  real_T b_gamma;
  real_T d;
  real_T delta;
  real_T theta;
  real_T xi;
  int32_T b_A[2];
  int32_T iv[2];
  int32_T a;
  int32_T i;
  int32_T i1;
  int32_T idx;
  int32_T j;
  int32_T k;
  int32_T last;
  int32_T n;
  boolean_T exitg1;
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
  i_st.prev = &h_st;
  i_st.tls = h_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  function [R, indef, E, err] = cholmod(A)
  //  CHOLMOD Modified Cholesky factorization
  //   R = cholmod(A) returns the upper Cholesky factor of A (same as CHOL)
  //   if A is (sufficiently) positive definite, and otherwise returns a
  //   modified factor R with diagonal enries >= sqrt(delta) and
  //   offdiagonal entries <= beta in absolute value,
  //   where delta and beta are defined in terms of size of diagonal and
  //   offdiagonal entries of A and the machine precision; see below.
  //   The idea is to ensure that E = A - R'*R is reasonably small if A is
  //   not too far from being indefinite.  If A is sparse, so is R.
  //   The output parameter indef is set to 0 if A is sufficiently positive
  //   definite and to 1 if the factorization is modified.
  //
  //   The point of modified Cholesky is to avoid computing eigenvalues,
  //   but for dense matrices, MODCHOL takes longer than calling the built-in
  //   EIG, because of the cost of interpreting the code, even though it
  //   only has one loop and uses vector operations.
  //   reference: Nocedal and Wright, Algorithm 3.4 and subsequent discussion
  //   (not Algorithm 3.5, which is more complicated)
  //   original algorithm is due to Gill and Murray, 1974
  //   written by M. Overton, overton@cs.nyu.edu, last modified Feb 2005
  //   convenient to work with A = LDL' where D is diagonal, L is unit
  //   lower triangular, and so R = (LD^(1/2))'
  //
  b_A[0] = A.size(1);
  b_A[1] = A.size(0);
  iv[0] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&A)->size())[0];
  iv[1] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&A)->size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_A[0], &l_emlrtECI, (emlrtCTX)sp);
  L.set_size(&y_emlrtRTEI, sp, A.size(0), A.size(1));
  last = A.size(1);
  for (i = 0; i < last; i++) {
    idx = A.size(0);
    for (i1 = 0; i1 < idx; i1++) {
      L[i1 + L.size(0) * i] = A[i1 + A.size(0) * i] - A[i + A.size(0) * i1];
    }
  }
  st.site = &o_emlrtRSI;
  coder::b_abs(&st, L, varargin_1);
  st.site = &o_emlrtRSI;
  coder::sum(&st, varargin_1, K);
  st.site = &o_emlrtRSI;
  if (coder::sum(&st, K) > 0.0) {
    st.site = &p_emlrtRSI;
    coder::b_error(&st);
  }
  //  set parameters governing bounds on L and D (eps is machine epsilon)
  if ((A.size(0) == 0) || (A.size(1) == 0)) {
    n = 0;
  } else {
    i = A.size(0);
    i1 = A.size(1);
    n = muIntScalarMax_sint32(i, i1);
  }
  st.site = &q_emlrtRSI;
  coder::diag(&st, A, diagA);
  st.site = &r_emlrtRSI;
  b_st.site = &r_emlrtRSI;
  coder::b_abs(&b_st, diagA, b_varargin_1);
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  if (b_varargin_1.size(0) < 1) {
    emlrtErrorWithMessageIdR2018a(&d_st, &h_emlrtRTEI,
                                  "Coder:toolbox:eml_min_or_max_varDimZero",
                                  "Coder:toolbox:eml_min_or_max_varDimZero", 0);
  }
  e_st.site = &yb_emlrtRSI;
  f_st.site = &ac_emlrtRSI;
  last = b_varargin_1.size(0);
  if (b_varargin_1.size(0) <= 2) {
    if (b_varargin_1.size(0) == 1) {
      b_gamma = b_varargin_1[0];
    } else if ((b_varargin_1[0] < b_varargin_1[1]) ||
               (muDoubleScalarIsNaN(b_varargin_1[0]) &&
                (!muDoubleScalarIsNaN(b_varargin_1[1])))) {
      b_gamma = b_varargin_1[1];
    } else {
      b_gamma = b_varargin_1[0];
    }
  } else {
    g_st.site = &cc_emlrtRSI;
    if (!muDoubleScalarIsNaN(b_varargin_1[0])) {
      idx = 1;
    } else {
      idx = 0;
      h_st.site = &dc_emlrtRSI;
      if (b_varargin_1.size(0) > 2147483646) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= last)) {
        if (!muDoubleScalarIsNaN(b_varargin_1[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }
    if (idx == 0) {
      b_gamma = b_varargin_1[0];
    } else {
      g_st.site = &bc_emlrtRSI;
      b_gamma = b_varargin_1[idx - 1];
      a = idx + 1;
      h_st.site = &ec_emlrtRSI;
      if ((idx + 1 <= b_varargin_1.size(0)) &&
          (b_varargin_1.size(0) > 2147483646)) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      for (k = a; k <= last; k++) {
        d = b_varargin_1[k - 1];
        if (b_gamma < d) {
          b_gamma = d;
        }
      }
    }
  }
  //  max diagonal entry
  st.site = &s_emlrtRSI;
  coder::diag(&st, diagA, varargin_1);
  iv[0] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&A)->size())[0];
  iv[1] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&A)->size())[1];
  b_A[0] = (*(int32_T(*)[2])varargin_1.size())[0];
  b_A[1] = (*(int32_T(*)[2])varargin_1.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_A[0], &k_emlrtECI, (emlrtCTX)sp);
  st.site = &s_emlrtRSI;
  L.set_size(&ab_emlrtRTEI, &st, A.size(0), A.size(1));
  last = A.size(0) * A.size(1);
  for (i = 0; i < last; i++) {
    L[i] = A[i] - varargin_1[i];
  }
  b_st.site = &s_emlrtRSI;
  coder::b_abs(&b_st, L, varargin_1);
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  if ((varargin_1.size(0) == 1) && (varargin_1.size(1) != 1)) {
    emlrtErrorWithMessageIdR2018a(&d_st, &i_emlrtRTEI,
                                  "Coder:toolbox:autoDimIncompatibility",
                                  "Coder:toolbox:autoDimIncompatibility", 0);
  }
  if (varargin_1.size(0) < 1) {
    emlrtErrorWithMessageIdR2018a(&d_st, &h_emlrtRTEI,
                                  "Coder:toolbox:eml_min_or_max_varDimZero",
                                  "Coder:toolbox:eml_min_or_max_varDimZero", 0);
  }
  e_st.site = &gc_emlrtRSI;
  f_st.site = &hc_emlrtRSI;
  g_st.site = &ic_emlrtRSI;
  last = varargin_1.size(0);
  idx = varargin_1.size(1);
  K.set_size(&bb_emlrtRTEI, &g_st, 1, varargin_1.size(1));
  if (varargin_1.size(1) >= 1) {
    h_st.site = &kc_emlrtRSI;
    if (varargin_1.size(1) > 2147483646) {
      i_st.site = &j_emlrtRSI;
      coder::check_forloop_overflow_error(&i_st);
    }
    for (j = 0; j < idx; j++) {
      K[j] = varargin_1[varargin_1.size(0) * j];
      h_st.site = &jc_emlrtRSI;
      if ((2 <= last) && (last > 2147483646)) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      for (a = 2; a <= last; a++) {
        boolean_T p;
        xi = K[j];
        theta = varargin_1[(a + varargin_1.size(0) * j) - 1];
        if (muDoubleScalarIsNaN(theta)) {
          p = false;
        } else if (muDoubleScalarIsNaN(xi)) {
          p = true;
        } else {
          p = (xi < theta);
        }
        if (p) {
          K[j] = theta;
        }
      }
    }
  }
  st.site = &s_emlrtRSI;
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  if (K.size(1) < 1) {
    emlrtErrorWithMessageIdR2018a(&d_st, &h_emlrtRTEI,
                                  "Coder:toolbox:eml_min_or_max_varDimZero",
                                  "Coder:toolbox:eml_min_or_max_varDimZero", 0);
  }
  e_st.site = &yb_emlrtRSI;
  f_st.site = &ac_emlrtRSI;
  last = K.size(1);
  if (K.size(1) <= 2) {
    if (K.size(1) == 1) {
      xi = K[0];
    } else if ((K[0] < K[1]) ||
               (muDoubleScalarIsNaN(K[0]) && (!muDoubleScalarIsNaN(K[1])))) {
      xi = K[1];
    } else {
      xi = K[0];
    }
  } else {
    g_st.site = &cc_emlrtRSI;
    if (!muDoubleScalarIsNaN(K[0])) {
      idx = 1;
    } else {
      idx = 0;
      h_st.site = &dc_emlrtRSI;
      if (K.size(1) > 2147483646) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= last)) {
        if (!muDoubleScalarIsNaN(K[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }
    if (idx == 0) {
      xi = K[0];
    } else {
      g_st.site = &bc_emlrtRSI;
      xi = K[idx - 1];
      a = idx + 1;
      h_st.site = &ec_emlrtRSI;
      if ((idx + 1 <= K.size(1)) && (K.size(1) > 2147483646)) {
        i_st.site = &j_emlrtRSI;
        coder::check_forloop_overflow_error(&i_st);
      }
      for (k = a; k <= last; k++) {
        d = K[k - 1];
        if (xi < d) {
          xi = d;
        }
      }
    }
  }
  //  max off-diagonal entry
  st.site = &t_emlrtRSI;
  theta = b_gamma + xi;
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  e_st.site = &yb_emlrtRSI;
  f_st.site = &ac_emlrtRSI;
  if ((theta < 1.0) || muDoubleScalarIsNaN(theta)) {
    theta = 1.0;
  }
  delta = 2.2204460492503131E-16 * theta;
  st.site = &u_emlrtRSI;
  c_varargin_1[0] = b_gamma;
  theta = xi / static_cast<real_T>(n);
  c_varargin_1[1] = theta;
  c_varargin_1[2] = 2.2204460492503131E-16;
  b_st.site = &vb_emlrtRSI;
  c_st.site = &wb_emlrtRSI;
  d_st.site = &xb_emlrtRSI;
  e_st.site = &yb_emlrtRSI;
  K.set_size(&cb_emlrtRTEI, &e_st, 1, 3);
  K[0] = b_gamma;
  K[1] = theta;
  K[2] = 2.2204460492503131E-16;
  f_st.site = &ac_emlrtRSI;
  g_st.site = &cc_emlrtRSI;
  if (!muDoubleScalarIsNaN(K[0])) {
    idx = 1;
  } else {
    idx = 0;
    h_st.site = &dc_emlrtRSI;
    k = 2;
    exitg1 = false;
    while ((!exitg1) && (k <= 3)) {
      if (!muDoubleScalarIsNaN(K[k - 1])) {
        idx = k;
        exitg1 = true;
      } else {
        k++;
      }
    }
  }
  if (idx != 0) {
    g_st.site = &bc_emlrtRSI;
    b_gamma = c_varargin_1[idx - 1];
    a = idx + 1;
    h_st.site = &ec_emlrtRSI;
    for (k = a; k < 4; k++) {
      d = c_varargin_1[k - 1];
      if (b_gamma < d) {
        b_gamma = d;
      }
    }
  }
  st.site = &u_emlrtRSI;
  b_gamma = muDoubleScalarSqrt(b_gamma);
  //  initialize d and L
  b_d.set_size(&db_emlrtRTEI, sp, n);
  for (i = 0; i < n; i++) {
    b_d[i] = 0.0;
  }
  st.site = &v_emlrtRSI;
  coder::eye(&st, static_cast<real_T>(n), L);
  //  there are no inner for loops, everything implemented with
  //  vector operations for a reasonable level of efficiency
  for (j = 0; j < n; j++) {
    real_T djtemp;
    if (j < 1) {
      K.set_size(&eb_emlrtRTEI, sp, 1, 0);
    } else {
      K.set_size(&eb_emlrtRTEI, sp, 1, j);
      last = j - 1;
      for (i = 0; i <= last; i++) {
        K[i] = static_cast<real_T>(i) + 1.0;
      }
    }
    //  column index: all columns to left of diagonal
    //  d(K) doesn't work in case K is empty
    r.set_size(&fb_emlrtRTEI, sp, K.size(1));
    last = K.size(1);
    for (i = 0; i < last; i++) {
      r[i] = K[i];
    }
    last = r.size(0);
    for (i = 0; i < last; i++) {
      i1 = static_cast<int32_T>(r[i]);
      if (i1 > b_d.size(0)) {
        emlrtDynamicBoundsCheckR2012b(i1, 1, b_d.size(0), &v_emlrtBCI,
                                      (emlrtCTX)sp);
      }
    }
    if (j + 1 > L.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(0), &t_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    diagA.set_size(&gb_emlrtRTEI, sp, r.size(0));
    last = r.size(0);
    for (i = 0; i < last; i++) {
      i1 = static_cast<int32_T>(r[i]);
      if (i1 > L.size(1)) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r[i]), 1, L.size(1),
                                      &w_emlrtBCI, (emlrtCTX)sp);
      }
      diagA[i] = L[j + L.size(0) * (i1 - 1)];
    }
    if (r.size(0) != diagA.size(0)) {
      emlrtSizeEqCheck1DR2012b(r.size(0), diagA.size(0), &j_emlrtECI,
                               (emlrtCTX)sp);
    }
    st.site = &w_emlrtRSI;
    if (j + 1 > L.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(0), &u_emlrtBCI, &st);
    }
    K.set_size(&hb_emlrtRTEI, &st, 1, r.size(0));
    last = r.size(0);
    for (i = 0; i < last; i++) {
      i1 = static_cast<int32_T>(r[i]);
      if (i1 > L.size(1)) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r[i]), 1, L.size(1),
                                      &x_emlrtBCI, &st);
      }
      K[i] = L[j + L.size(0) * (i1 - 1)];
    }
    b_varargin_1.set_size(&ib_emlrtRTEI, &st, r.size(0));
    last = r.size(0);
    for (i = 0; i < last; i++) {
      b_varargin_1[i] = b_d[static_cast<int32_T>(r[i]) - 1] * diagA[i];
    }
    b_st.site = &h_emlrtRSI;
    coder::b_dynamic_size_checks(&b_st, K, b_varargin_1, r.size(0),
                                 b_varargin_1.size(0));
    if (j + 1 > A.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, A.size(0), &y_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (j + 1 > A.size(1)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, A.size(1), &y_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    djtemp =
        A[j + A.size(0) * j] - coder::internal::blas::mtimes(K, b_varargin_1);
    //  C(j,j) in book
    if (j + 1 < n) {
      st.site = &x_emlrtRSI;
      b_st.site = &oc_emlrtRSI;
      if (static_cast<uint32_T>(n) < j + 2U) {
        K.set_size(&jb_emlrtRTEI, &b_st, 1, 0);
      } else {
        i = n - j;
        K.set_size(&jb_emlrtRTEI, &b_st, 1, i - 1);
        last = i - 2;
        for (i = 0; i <= last; i++) {
          K[i] = (static_cast<uint32_T>(j) + i) + 2U;
        }
      }
      //  row index: all rows below diagonal
      last = r.size(0);
      for (i = 0; i < last; i++) {
        i1 = static_cast<int32_T>(r[i]);
        if (i1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(i1, 1, b_d.size(0), &ab_emlrtBCI,
                                        (emlrtCTX)sp);
        }
      }
      if (j + 1 > L.size(0)) {
        emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(0), &r_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      diagA.set_size(&kb_emlrtRTEI, sp, r.size(0));
      last = r.size(0);
      for (i = 0; i < last; i++) {
        i1 = static_cast<int32_T>(r[i]);
        if (i1 > L.size(1)) {
          emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r[i]), 1,
                                        L.size(1), &cb_emlrtBCI, (emlrtCTX)sp);
        }
        diagA[i] = L[j + L.size(0) * (i1 - 1)];
      }
      if (r.size(0) != diagA.size(0)) {
        emlrtSizeEqCheck1DR2012b(r.size(0), diagA.size(0), &i_emlrtECI,
                                 (emlrtCTX)sp);
      }
      r1.set_size(&lb_emlrtRTEI, sp, K.size(1));
      last = K.size(1);
      for (i = 0; i < last; i++) {
        r1[i] = K[i];
      }
      last = r1.size(0);
      for (i = 0; i < last; i++) {
        i1 = static_cast<int32_T>(r1[i]);
        if (r1[i] != i1) {
          emlrtIntegerCheckR2012b(r1[i], &c_emlrtDCI, (emlrtCTX)sp);
        }
        if ((i1 < 1) || (i1 > A.size(0))) {
          emlrtDynamicBoundsCheckR2012b(i1, 1, A.size(0), &db_emlrtBCI,
                                        (emlrtCTX)sp);
        }
      }
      if (j + 1 > A.size(1)) {
        emlrtDynamicBoundsCheckR2012b(j + 1, 1, A.size(1), &s_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      st.site = &y_emlrtRSI;
      varargin_1.set_size(&mb_emlrtRTEI, &st, r1.size(0), r.size(0));
      last = r.size(0);
      for (i = 0; i < last; i++) {
        idx = r1.size(0);
        for (i1 = 0; i1 < idx; i1++) {
          a = static_cast<int32_T>(r1[i1]);
          if ((a < 1) || (a > L.size(0))) {
            emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r1[i1]), 1,
                                          L.size(0), &eb_emlrtBCI, &st);
          }
          k = static_cast<int32_T>(r[i]);
          if (k > L.size(1)) {
            emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r[i]), 1,
                                          L.size(1), &fb_emlrtBCI, &st);
          }
          varargin_1[i1 + varargin_1.size(0) * i] =
              L[(a + L.size(0) * (k - 1)) - 1];
        }
      }
      b_varargin_1.set_size(&nb_emlrtRTEI, &st, r.size(0));
      last = r.size(0);
      for (i = 0; i < last; i++) {
        b_varargin_1[i] = b_d[static_cast<int32_T>(r[i]) - 1] * diagA[i];
      }
      b_st.site = &h_emlrtRSI;
      coder::dynamic_size_checks(&b_st, varargin_1, b_varargin_1, r.size(0),
                                 b_varargin_1.size(0));
      b_st.site = &g_emlrtRSI;
      coder::internal::blas::mtimes(&b_st, varargin_1, b_varargin_1, r);
      if (r1.size(0) != r.size(0)) {
        emlrtSizeEqCheck1DR2012b(r1.size(0), r.size(0), &h_emlrtECI,
                                 (emlrtCTX)sp);
      }
      diagA.set_size(&ob_emlrtRTEI, sp, r1.size(0));
      last = r1.size(0);
      for (i = 0; i < last; i++) {
        diagA[i] = A[(static_cast<int32_T>(r1[i]) + A.size(0) * j) - 1] - r[i];
      }
      //  C(I,j) in book
      st.site = &ab_emlrtRSI;
      b_st.site = &ab_emlrtRSI;
      coder::b_abs(&b_st, diagA, b_varargin_1);
      b_st.site = &vb_emlrtRSI;
      c_st.site = &wb_emlrtRSI;
      d_st.site = &xb_emlrtRSI;
      if (b_varargin_1.size(0) < 1) {
        emlrtErrorWithMessageIdR2018a(
            &d_st, &h_emlrtRTEI, "Coder:toolbox:eml_min_or_max_varDimZero",
            "Coder:toolbox:eml_min_or_max_varDimZero", 0);
      }
      e_st.site = &yb_emlrtRSI;
      f_st.site = &ac_emlrtRSI;
      last = b_varargin_1.size(0);
      if (b_varargin_1.size(0) <= 2) {
        if (b_varargin_1.size(0) == 1) {
          theta = b_varargin_1[0];
        } else if ((b_varargin_1[0] < b_varargin_1[1]) ||
                   (muDoubleScalarIsNaN(b_varargin_1[0]) &&
                    (!muDoubleScalarIsNaN(b_varargin_1[1])))) {
          theta = b_varargin_1[1];
        } else {
          theta = b_varargin_1[0];
        }
      } else {
        g_st.site = &cc_emlrtRSI;
        if (!muDoubleScalarIsNaN(b_varargin_1[0])) {
          idx = 1;
        } else {
          idx = 0;
          h_st.site = &dc_emlrtRSI;
          if (b_varargin_1.size(0) > 2147483646) {
            i_st.site = &j_emlrtRSI;
            coder::check_forloop_overflow_error(&i_st);
          }
          k = 2;
          exitg1 = false;
          while ((!exitg1) && (k <= last)) {
            if (!muDoubleScalarIsNaN(b_varargin_1[k - 1])) {
              idx = k;
              exitg1 = true;
            } else {
              k++;
            }
          }
        }
        if (idx == 0) {
          theta = b_varargin_1[0];
        } else {
          g_st.site = &bc_emlrtRSI;
          theta = b_varargin_1[idx - 1];
          a = idx + 1;
          h_st.site = &ec_emlrtRSI;
          if ((idx + 1 <= b_varargin_1.size(0)) &&
              (b_varargin_1.size(0) > 2147483646)) {
            i_st.site = &j_emlrtRSI;
            coder::check_forloop_overflow_error(&i_st);
          }
          for (k = a; k <= last; k++) {
            d = b_varargin_1[k - 1];
            if (theta < d) {
              theta = d;
            }
          }
        }
      }
      //  guarantees d(j) not too small and L(I,j) not too big
      //  in sufficiently positive definite case, d(j) = djtemp
      st.site = &bb_emlrtRSI;
      xi = theta / b_gamma;
      b_st.site = &e_emlrtRSI;
      c_st.site = &f_emlrtRSI;
      st.site = &bb_emlrtRSI;
      c_varargin_1[0] = muDoubleScalarAbs(djtemp);
      theta = xi * xi;
      c_varargin_1[1] = theta;
      c_varargin_1[2] = delta;
      b_st.site = &vb_emlrtRSI;
      c_st.site = &wb_emlrtRSI;
      d_st.site = &xb_emlrtRSI;
      e_st.site = &yb_emlrtRSI;
      K.set_size(&cb_emlrtRTEI, &e_st, 1, 3);
      K[0] = muDoubleScalarAbs(djtemp);
      K[1] = theta;
      K[2] = delta;
      f_st.site = &ac_emlrtRSI;
      g_st.site = &cc_emlrtRSI;
      if (!muDoubleScalarIsNaN(K[0])) {
        idx = 1;
      } else {
        idx = 0;
        h_st.site = &dc_emlrtRSI;
        k = 2;
        exitg1 = false;
        while ((!exitg1) && (k <= 3)) {
          if (!muDoubleScalarIsNaN(K[k - 1])) {
            idx = k;
            exitg1 = true;
          } else {
            k++;
          }
        }
      }
      if (idx == 0) {
        if (j + 1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &gb_emlrtBCI,
                                        &f_st);
        }
        b_d[j] = muDoubleScalarAbs(djtemp);
      } else {
        g_st.site = &bc_emlrtRSI;
        theta = c_varargin_1[idx - 1];
        a = idx + 1;
        h_st.site = &ec_emlrtRSI;
        for (k = a; k < 4; k++) {
          d = c_varargin_1[k - 1];
          if (theta < d) {
            theta = d;
          }
        }
        if (j + 1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &gb_emlrtBCI,
                                        &f_st);
        }
        b_d[j] = theta;
      }
      r2.set_size(&pb_emlrtRTEI, sp, r1.size(0));
      last = r1.size(0);
      for (i = 0; i < last; i++) {
        i1 = static_cast<int32_T>(r1[i]);
        if ((i1 < 1) || (i1 > L.size(0))) {
          emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(r1[i]), 1,
                                        L.size(0), &hb_emlrtBCI, (emlrtCTX)sp);
        }
        r2[i] = i1 - 1;
      }
      if (j + 1 > L.size(1)) {
        emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(1), &q_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      if (j + 1 > b_d.size(0)) {
        emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &ib_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      last = diagA.size(0);
      for (i = 0; i < last; i++) {
        diagA[i] = diagA[i] / b_d[j];
      }
      emlrtSubAssignSizeCheckR2012b(r2.size(), 1, diagA.size(), 1, &g_emlrtECI,
                                    (emlrtCTX)sp);
      last = diagA.size(0);
      for (i = 0; i < last; i++) {
        L[r2[i] + L.size(0) * j] = diagA[i];
      }
    } else {
      st.site = &cb_emlrtRSI;
      d = muDoubleScalarAbs(djtemp);
      b_st.site = &vb_emlrtRSI;
      c_st.site = &wb_emlrtRSI;
      d_st.site = &xb_emlrtRSI;
      e_st.site = &yb_emlrtRSI;
      f_st.site = &ac_emlrtRSI;
      if ((d < delta) ||
          (muDoubleScalarIsNaN(d) && (!muDoubleScalarIsNaN(delta)))) {
        if (j + 1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &bb_emlrtBCI,
                                        &f_st);
        }
        b_d[j] = delta;
      } else {
        if (j + 1 > b_d.size(0)) {
          emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &bb_emlrtBCI,
                                        &f_st);
        }
        b_d[j] = d;
      }
    }
    if (j + 1 > b_d.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &p_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  //  convert to usual output format: replace L by L*sqrt(D) and transpose
  for (j = 0; j < n; j++) {
    if (j + 1 > L.size(1)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, L.size(1), &n_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    st.site = &db_emlrtRSI;
    if (j + 1 > b_d.size(0)) {
      emlrtDynamicBoundsCheckR2012b(j + 1, 1, b_d.size(0), &o_emlrtBCI, &st);
    }
    theta = muDoubleScalarSqrt(b_d[j]);
    last = L.size(0);
    r.set_size(&qb_emlrtRTEI, sp, L.size(0));
    for (i = 0; i < last; i++) {
      r[i] = L[i + L.size(0) * j] * theta;
    }
    emlrtSubAssignSizeCheckR2012b(L.size(), 1, r.size(), 1, &f_emlrtECI,
                                  (emlrtCTX)sp);
    last = r.size(0);
    for (i = 0; i < last; i++) {
      L[i + L.size(0) * j] = r[i];
    }
    //  L = L*diag(sqrt(d)) bad in sparse case
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  R.set_size(&rb_emlrtRTEI, sp, L.size(1), L.size(0));
  last = L.size(0);
  for (i = 0; i < last; i++) {
    idx = L.size(1);
    for (i1 = 0; i1 < idx; i1++) {
      R[i1 + R.size(0) * i] = L[i + L.size(0) * i1];
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (cholmod.cpp)
