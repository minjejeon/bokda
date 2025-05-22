//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnlik_NonLinear.cpp
//
// Code generation for function 'lnlik_NonLinear'
//

// Include files
#include "lnlik_NonLinear.h"
#include "diag.h"
#include "eml_mtimes_helper.h"
#include "eye.h"
#include "invpd.h"
#include "lnlik_data.h"
#include "lnpdfmvn1.h"
#include "mtimes.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo c_emlrtRSI{
    5,                 // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo d_emlrtRSI{
    22,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo e_emlrtRSI{
    24,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo f_emlrtRSI{
    25,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo g_emlrtRSI{
    27,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo h_emlrtRSI{
    28,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo i_emlrtRSI{
    30,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo j_emlrtRSI{
    34,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo k_emlrtRSI{
    46,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo l_emlrtRSI{
    47,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo m_emlrtRSI{
    48,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo n_emlrtRSI{
    49,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo o_emlrtRSI{
    51,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo p_emlrtRSI{
    54,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo q_emlrtRSI{
    56,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo r_emlrtRSI{
    57,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo s_emlrtRSI{
    58,                // lineNo
    "lnlik_NonLinear", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pathName
};

static emlrtRSInfo ab_emlrtRSI{
    24,    // lineNo
    "cat", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRSInfo bb_emlrtRSI{
    96,         // lineNo
    "cat_impl", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRTEInfo d_emlrtRTEI{
    271,                   // lineNo
    27,                    // colNo
    "check_non_axis_size", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pName
};

static emlrtECInfo c_emlrtECI{
    -1,                // nDims
    61,                // lineNo
    5,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtBCInfo c_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    61,                // lineNo
    17,                // colNo
    "P_ttm",           // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtECInfo d_emlrtECI{
    -1,                // nDims
    60,                // lineNo
    5,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtBCInfo d_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    60,                // lineNo
    11,                // colNo
    "G_ttm",           // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo c_emlrtDCI{
    60,                // lineNo
    11,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtECInfo e_emlrtECI{
    2,                 // nDims
    58,                // lineNo
    12,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo f_emlrtECI{
    -1,                // nDims
    57,                // lineNo
    12,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo g_emlrtECI{
    -1,                // nDims
    57,                // lineNo
    32,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo h_emlrtECI{
    2,                 // nDims
    52,                // lineNo
    20,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo i_emlrtECI{
    2,                 // nDims
    50,                // lineNo
    17,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo j_emlrtECI{
    2,                 // nDims
    49,                // lineNo
    12,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo k_emlrtECI{
    -1,                // nDims
    48,                // lineNo
    12,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo l_emlrtECI{
    2,                 // nDims
    47,                // lineNo
    12,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo m_emlrtECI{
    -1,                // nDims
    19,                // lineNo
    5,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtBCInfo e_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    19,                // lineNo
    20,                // colNo
    "G_LL",            // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo d_emlrtDCI{
    19,                // lineNo
    20,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtBCInfo f_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    19,                // lineNo
    10,                // colNo
    "G_LL",            // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo e_emlrtDCI{
    19,                // lineNo
    10,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtBCInfo g_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    45,                // lineNo
    21,                // colNo
    "YLm",             // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo f_emlrtDCI{
    45,                // lineNo
    21,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtBCInfo h_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    44,                // lineNo
    14,                // colNo
    "Y0",              // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo g_emlrtDCI{
    44,                // lineNo
    14,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtRTEInfo e_emlrtRTEI{
    42,                // lineNo
    9,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo n_emlrtECI{
    2,                 // nDims
    34,                // lineNo
    9,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtBCInfo i_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    34,                // lineNo
    47,                // colNo
    "X_mat",           // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtBCInfo j_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    34,                // lineNo
    41,                // colNo
    "X_mat",           // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtBCInfo k_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    34,                // lineNo
    30,                // colNo
    "Phi",             // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtBCInfo l_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    34,                // lineNo
    20,                // colNo
    "Phi",             // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtRTEInfo f_emlrtRTEI{
    33,                // lineNo
    9,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtBCInfo m_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    32,                // lineNo
    13,                // colNo
    "X_mat",           // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtBCInfo n_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    32,                // lineNo
    11,                // colNo
    "X_mat",           // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo h_emlrtDCI{
    25,                // lineNo
    20,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo i_emlrtDCI{
    25,                // lineNo
    20,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    4              // checkKind
};

static emlrtDCInfo j_emlrtDCI{
    24,                // lineNo
    44,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo k_emlrtDCI{
    24,                // lineNo
    44,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    4              // checkKind
};

static emlrtDCInfo l_emlrtDCI{
    24,                // lineNo
    22,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo m_emlrtDCI{
    24,                // lineNo
    22,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    4              // checkKind
};

static emlrtBCInfo o_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    19,                // lineNo
    30,                // colNo
    "Mu",              // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo n_emlrtDCI{
    19,                // lineNo
    30,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtRTEInfo g_emlrtRTEI{
    18,                // lineNo
    9,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtECInfo o_emlrtECI{
    -1,                // nDims
    12,                // lineNo
    9,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtBCInfo p_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    12,                // lineNo
    15,                // colNo
    "Mu",              // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo o_emlrtDCI{
    12,                // lineNo
    26,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo p_emlrtDCI{
    12,                // lineNo
    26,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    4              // checkKind
};

static emlrtDCInfo q_emlrtDCI{
    37,                // lineNo
    15,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo r_emlrtDCI{
    37,                // lineNo
    15,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    4              // checkKind
};

static emlrtDCInfo s_emlrtDCI{
    37,                // lineNo
    18,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo t_emlrtDCI{
    38,                // lineNo
    15,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo u_emlrtDCI{
    38,                // lineNo
    20,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo v_emlrtDCI{
    38,                // lineNo
    25,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtBCInfo q_emlrtBCI{
    -1,                // iFirst
    -1,                // iLast
    11,                // lineNo
    8,                 // colNo
    "gamma",           // aName
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    0              // checkKind
};

static emlrtDCInfo w_emlrtDCI{
    17,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo x_emlrtDCI{
    17,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    4              // checkKind
};

static emlrtDCInfo y_emlrtDCI{
    37,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtDCInfo ab_emlrtDCI{
    38,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m", // pName
    1              // checkKind
};

static emlrtRTEInfo w_emlrtRTEI{
    17,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo x_emlrtRTEI{
    24,    // lineNo
    5,     // colNo
    "cat", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pName
};

static emlrtRTEInfo y_emlrtRTEI{
    24,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo ab_emlrtRTEI{
    25,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo bb_emlrtRTEI{
    32,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo cb_emlrtRTEI{
    34,                // lineNo
    13,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo db_emlrtRTEI{
    34,                // lineNo
    35,                // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo eb_emlrtRTEI{
    37,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo fb_emlrtRTEI{
    38,                // lineNo
    1,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo gb_emlrtRTEI{
    44,                // lineNo
    5,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo hb_emlrtRTEI{
    45,                // lineNo
    5,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo ib_emlrtRTEI{
    52,                // lineNo
    5,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

static emlrtRTEInfo jb_emlrtRTEI{
    57,                // lineNo
    5,                 // colNo
    "lnlik_NonLinear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
    "NonLinear.m" // pName
};

// Function Definitions
real_T lnlik_NonLinear(const emlrtStack *sp, coder::array<real_T, 2U> &Mu,
                       const coder::array<real_T, 2U> &Y0,
                       const coder::array<real_T, 3U> &YLm,
                       const coder::array<real_T, 1U> &beta,
                       const coder::array<real_T, 2U> &Phi,
                       const coder::array<real_T, 2U> &Omega,
                       const coder::array<real_T, 1U> &diag_Sigma,
                       const coder::array<real_T, 1U> &b_gamma)
{
  coder::array<real_T, 3U> P_ttm;
  coder::array<real_T, 2U> F;
  coder::array<real_T, 2U> Kalman_gain;
  coder::array<real_T, 2U> P_tL;
  coder::array<real_T, 2U> SIGMA;
  coder::array<real_T, 2U> Sigma;
  coder::array<real_T, 2U> W;
  coder::array<real_T, 2U> X;
  coder::array<real_T, 2U> y;
  coder::array<real_T, 1U> G_tL;
  coder::array<real_T, 1U> G_tt;
  coder::array<real_T, 1U> y_t;
  coder::array<real_T, 1U> y_tL;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T MP1;
  real_T P;
  real_T T;
  real_T d;
  real_T d1;
  real_T lnL;
  int32_T b_result[2];
  int32_T iv[2];
  int32_T result[2];
  int32_T M;
  int32_T b_loop_ub;
  int32_T c_loop_ub;
  int32_T d_loop_ub;
  int32_T e_loop_ub;
  int32_T i;
  int32_T i1;
  int32_T i2;
  int32_T i3;
  int32_T i7;
  int32_T i8;
  int32_T i9;
  int32_T input_sizes_idx_1;
  int32_T loop_ub;
  int32_T p;
  int32_T sizes_idx_1;
  boolean_T empty_non_axis_sizes;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  우도함수 계산하기
  M = YLm.size(0);
  st.site = &c_emlrtRSI;
  b_st.site = &t_emlrtRSI;
  P = static_cast<real_T>(YLm.size(1)) /
      (static_cast<real_T>(YLm.size(0)) * static_cast<real_T>(YLm.size(0)));
  T = static_cast<real_T>(YLm.size(2)) + P;
  MP1 = static_cast<real_T>(YLm.size(0)) * (P + 1.0);
  i = YLm.size(0);
  for (sizes_idx_1 = 0; sizes_idx_1 < i; sizes_idx_1++) {
    if (sizes_idx_1 + 1 > b_gamma.size(0)) {
      emlrtDynamicBoundsCheckR2012b(sizes_idx_1 + 1, 1, b_gamma.size(0),
                                    &q_emlrtBCI, (emlrtCTX)sp);
    }
    if (b_gamma[sizes_idx_1] == 0.0) {
      if (sizes_idx_1 + 1 > Mu.size(1)) {
        emlrtDynamicBoundsCheckR2012b(sizes_idx_1 + 1, 1, Mu.size(1),
                                      &p_emlrtBCI, (emlrtCTX)sp);
      }
      if (!(T >= 0.0)) {
        emlrtNonNegativeCheckR2012b(rtNaN, &p_emlrtDCI, (emlrtCTX)sp);
      }
      if (T != static_cast<int32_T>(muDoubleScalarFloor(T))) {
        emlrtIntegerCheckR2012b(T, &o_emlrtDCI, (emlrtCTX)sp);
      }
      input_sizes_idx_1 = static_cast<int32_T>(T);
      emlrtSubAssignSizeCheckR2012b(Mu.size(), 1, &input_sizes_idx_1, 1,
                                    &o_emlrtECI, (emlrtCTX)sp);
      for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
        Mu[i1 + Mu.size(0) * sizes_idx_1] = 0.0;
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  //  상태변수의 초기값은 주어진 것으로 처리
  if (!(MP1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &x_emlrtDCI, (emlrtCTX)sp);
  }
  i = static_cast<int32_T>(muDoubleScalarFloor(MP1));
  if (MP1 != i) {
    emlrtIntegerCheckR2012b(MP1, &w_emlrtDCI, (emlrtCTX)sp);
  }
  G_tt.set_size(&w_emlrtRTEI, sp, static_cast<int32_T>(MP1));
  if (MP1 != i) {
    emlrtIntegerCheckR2012b(MP1, &w_emlrtDCI, (emlrtCTX)sp);
  }
  loop_ub = static_cast<int32_T>(MP1);
  for (i1 = 0; i1 < loop_ub; i1++) {
    G_tt[i1] = 0.0;
  }
  i1 = static_cast<int32_T>(((-1.0 - (P + 1.0)) + 1.0) / -1.0);
  emlrtForLoopVectorCheckR2021a(P + 1.0, -1.0, 1.0, mxDOUBLE_CLASS, i1,
                                &g_emlrtRTEI, (emlrtCTX)sp);
  for (p = 0; p < i1; p++) {
    real_T b_p;
    b_p = (P + 1.0) + -static_cast<real_T>(p);
    d = static_cast<real_T>(M) * (b_p - 1.0) + 1.0;
    d1 = static_cast<real_T>(M) * b_p;
    if (d > d1) {
      i2 = -1;
      i3 = -1;
    } else {
      if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
        emlrtIntegerCheckR2012b(d, &e_emlrtDCI, (emlrtCTX)sp);
      }
      if ((static_cast<int32_T>(d) < 1) ||
          (static_cast<int32_T>(d) > G_tt.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, G_tt.size(0),
                                      &f_emlrtBCI, (emlrtCTX)sp);
      }
      i2 = static_cast<int32_T>(d) - 2;
      if (d1 != static_cast<int32_T>(muDoubleScalarFloor(d1))) {
        emlrtIntegerCheckR2012b(d1, &d_emlrtDCI, (emlrtCTX)sp);
      }
      if ((static_cast<int32_T>(d1) < 1) ||
          (static_cast<int32_T>(d1) > G_tt.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d1), 1, G_tt.size(0),
                                      &e_emlrtBCI, (emlrtCTX)sp);
      }
      i3 = static_cast<int32_T>(d1) - 1;
    }
    if (b_p != static_cast<int32_T>(muDoubleScalarFloor(b_p))) {
      emlrtIntegerCheckR2012b(b_p, &n_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(b_p) < 1) ||
        (static_cast<int32_T>(b_p) > Mu.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, Mu.size(0),
                                    &o_emlrtBCI, (emlrtCTX)sp);
    }
    loop_ub = i3 - i2;
    if (loop_ub != Mu.size(1)) {
      emlrtSubAssignSizeCheck1dR2017a(loop_ub, Mu.size(1), &m_emlrtECI,
                                      (emlrtCTX)sp);
    }
    for (i3 = 0; i3 < loop_ub; i3++) {
      G_tt[(i2 + i3) + 1] =
          Mu[(static_cast<int32_T>(b_p) + Mu.size(0) * i3) - 1];
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  st.site = &d_emlrtRSI;
  coder::eye(&st, MP1, P_tL);
  loop_ub = P_tL.size(0) * P_tL.size(1);
  for (i1 = 0; i1 < loop_ub; i1++) {
    P_tL[i1] = 1.0E-5 * P_tL[i1];
  }
  st.site = &e_emlrtRSI;
  b_st.site = &e_emlrtRSI;
  coder::eye(&b_st, static_cast<real_T>(YLm.size(0)), Kalman_gain);
  d = static_cast<real_T>(YLm.size(0)) * P;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &m_emlrtDCI, &st);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &l_emlrtDCI, &st);
  }
  b_st.site = &ab_emlrtRSI;
  if ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0)) {
    loop_ub = Kalman_gain.size(0);
  } else if ((YLm.size(0) != 0) &&
             (static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) !=
              0)) {
    loop_ub = YLm.size(0);
  } else {
    loop_ub = Kalman_gain.size(0);
    if (YLm.size(0) > Kalman_gain.size(0)) {
      loop_ub = YLm.size(0);
    }
  }
  c_st.site = &bb_emlrtRSI;
  if ((Kalman_gain.size(0) != loop_ub) &&
      ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &d_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((YLm.size(0) != loop_ub) &&
      ((YLm.size(0) != 0) &&
       (static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &d_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (loop_ub == 0);
  if (empty_non_axis_sizes ||
      ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0))) {
    input_sizes_idx_1 = Kalman_gain.size(1);
  } else {
    input_sizes_idx_1 = 0;
  }
  if (empty_non_axis_sizes ||
      ((YLm.size(0) != 0) &&
       (static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0))) {
    sizes_idx_1 = static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P);
  } else {
    sizes_idx_1 = 0;
  }
  Sigma.set_size(&x_emlrtRTEI, &b_st, loop_ub, input_sizes_idx_1 + sizes_idx_1);
  for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      Sigma[i2 + Sigma.size(0) * i1] = Kalman_gain[i2 + loop_ub * i1];
    }
  }
  for (i1 = 0; i1 < sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      Sigma[i2 + Sigma.size(0) * (i1 + input_sizes_idx_1)] = 0.0;
    }
  }
  st.site = &e_emlrtRSI;
  b_st.site = &e_emlrtRSI;
  coder::eye(&b_st, static_cast<real_T>(YLm.size(0)) * P, Kalman_gain);
  d = static_cast<real_T>(YLm.size(0)) * P;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &k_emlrtDCI, &st);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &j_emlrtDCI, &st);
  }
  b_st.site = &ab_emlrtRSI;
  if ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0)) {
    loop_ub = Kalman_gain.size(0);
  } else if ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) !=
              0) &&
             (YLm.size(0) != 0)) {
    loop_ub = static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P);
  } else {
    loop_ub = Kalman_gain.size(0);
    if (static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) >
        Kalman_gain.size(0)) {
      loop_ub = static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P);
    }
  }
  c_st.site = &bb_emlrtRSI;
  if ((Kalman_gain.size(0) != loop_ub) &&
      ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &d_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != loop_ub) &&
      ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0) &&
       (YLm.size(0) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &d_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (loop_ub == 0);
  if (empty_non_axis_sizes ||
      ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0))) {
    input_sizes_idx_1 = Kalman_gain.size(1);
  } else {
    input_sizes_idx_1 = 0;
  }
  if (empty_non_axis_sizes ||
      ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0) &&
       (YLm.size(0) != 0))) {
    sizes_idx_1 = YLm.size(0);
  } else {
    sizes_idx_1 = 0;
  }
  result[0] = loop_ub;
  W.set_size(&x_emlrtRTEI, &b_st, loop_ub, input_sizes_idx_1 + sizes_idx_1);
  for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      W[i2 + W.size(0) * i1] = Kalman_gain[i2 + loop_ub * i1];
    }
  }
  for (i1 = 0; i1 < sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      W[i2 + W.size(0) * (i1 + input_sizes_idx_1)] = 0.0;
    }
  }
  st.site = &e_emlrtRSI;
  b_st.site = &ab_emlrtRSI;
  if ((Sigma.size(0) != 0) && (Sigma.size(1) != 0)) {
    loop_ub = Sigma.size(1);
  } else if ((W.size(0) != 0) && (W.size(1) != 0)) {
    loop_ub = W.size(1);
  } else {
    loop_ub = Sigma.size(1);
    if (W.size(1) > Sigma.size(1)) {
      loop_ub = W.size(1);
    }
  }
  c_st.site = &bb_emlrtRSI;
  if ((Sigma.size(1) != loop_ub) &&
      ((Sigma.size(0) != 0) && (Sigma.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &d_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((W.size(1) != loop_ub) && ((W.size(0) != 0) && (W.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &d_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (loop_ub == 0);
  if (empty_non_axis_sizes || ((Sigma.size(0) != 0) && (Sigma.size(1) != 0))) {
    input_sizes_idx_1 = Sigma.size(0);
  } else {
    input_sizes_idx_1 = 0;
  }
  if (empty_non_axis_sizes || ((W.size(0) != 0) && (W.size(1) != 0))) {
    sizes_idx_1 = W.size(0);
  } else {
    sizes_idx_1 = 0;
  }
  F.set_size(&y_emlrtRTEI, &b_st, input_sizes_idx_1 + sizes_idx_1, loop_ub);
  for (i1 = 0; i1 < loop_ub; i1++) {
    for (i2 = 0; i2 < input_sizes_idx_1; i2++) {
      F[i2 + F.size(0) * i1] = Sigma[i2 + input_sizes_idx_1 * i1];
    }
  }
  for (i1 = 0; i1 < loop_ub; i1++) {
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      F[(i2 + input_sizes_idx_1) + F.size(0) * i1] = W[i2 + sizes_idx_1 * i1];
    }
  }
  st.site = &f_emlrtRSI;
  b_st.site = &f_emlrtRSI;
  coder::eye(&b_st, static_cast<real_T>(YLm.size(0)), Kalman_gain);
  d = static_cast<real_T>(YLm.size(0)) * P;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &i_emlrtDCI, &st);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &h_emlrtDCI, &st);
  }
  b_st.site = &ab_emlrtRSI;
  if ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0)) {
    loop_ub = Kalman_gain.size(1);
  } else if ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) !=
              0) &&
             (YLm.size(0) != 0)) {
    loop_ub = YLm.size(0);
  } else {
    loop_ub = Kalman_gain.size(1);
    if (YLm.size(0) > Kalman_gain.size(1)) {
      loop_ub = YLm.size(0);
    }
  }
  c_st.site = &bb_emlrtRSI;
  if ((Kalman_gain.size(1) != loop_ub) &&
      ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &d_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((YLm.size(0) != loop_ub) &&
      ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0) &&
       (YLm.size(0) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &d_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (loop_ub == 0);
  if (empty_non_axis_sizes ||
      ((Kalman_gain.size(0) != 0) && (Kalman_gain.size(1) != 0))) {
    input_sizes_idx_1 = Kalman_gain.size(0);
  } else {
    input_sizes_idx_1 = 0;
  }
  if (empty_non_axis_sizes ||
      ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0) &&
       (YLm.size(0) != 0))) {
    sizes_idx_1 = static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P);
  } else {
    sizes_idx_1 = 0;
  }
  W.set_size(&ab_emlrtRTEI, &b_st, input_sizes_idx_1 + sizes_idx_1, loop_ub);
  for (i1 = 0; i1 < loop_ub; i1++) {
    for (i2 = 0; i2 < input_sizes_idx_1; i2++) {
      W[i2 + W.size(0) * i1] = Kalman_gain[i2 + input_sizes_idx_1 * i1];
    }
  }
  for (i1 = 0; i1 < loop_ub; i1++) {
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      W[(i2 + input_sizes_idx_1) + W.size(0) * i1] = 0.0;
    }
  }
  st.site = &g_emlrtRSI;
  coder::diag(&st, diag_Sigma, Sigma);
  //  M by M
  st.site = &h_emlrtRSI;
  b_st.site = &eb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, W, Sigma, W.size(1), Sigma.size(0));
  b_st.site = &db_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, W, Sigma, y);
  st.site = &h_emlrtRSI;
  b_st.site = &eb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, y, W, y.size(1), W.size(1));
  b_st.site = &db_emlrtRSI;
  coder::internal::blas::b_mtimes(&b_st, y, W, SIGMA);
  st.site = &i_emlrtRSI;
  coder::eye(&st, MP1, W);
  if (1 > YLm.size(0)) {
    loop_ub = 0;
  } else {
    if (1 > W.size(0)) {
      emlrtDynamicBoundsCheckR2012b(1, 1, W.size(0), &n_emlrtBCI, (emlrtCTX)sp);
    }
    if (YLm.size(0) > W.size(0)) {
      emlrtDynamicBoundsCheckR2012b(YLm.size(0), 1, W.size(0), &m_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    loop_ub = YLm.size(0);
  }
  b_loop_ub = W.size(1);
  X.set_size(&bb_emlrtRTEI, sp, loop_ub, W.size(1));
  for (i1 = 0; i1 < b_loop_ub; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      X[i2 + X.size(0) * i1] = W[i2 + W.size(0) * i1];
    }
  }
  i1 = static_cast<int32_T>(P);
  emlrtForLoopVectorCheckR2021a(1.0, 1.0, P, mxDOUBLE_CLASS,
                                static_cast<int32_T>(P), &f_emlrtRTEI,
                                (emlrtCTX)sp);
  for (p = 0; p < i1; p++) {
    int32_T i4;
    int32_T i5;
    int32_T i6;
    d = static_cast<real_T>(M) * ((static_cast<real_T>(p) + 1.0) - 1.0) + 1.0;
    d1 = static_cast<real_T>(M) * (static_cast<real_T>(p) + 1.0);
    if (d > d1) {
      i2 = 0;
      i3 = 0;
    } else {
      if ((static_cast<int32_T>(d) < 1) ||
          (static_cast<int32_T>(d) > Phi.size(1))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, Phi.size(1),
                                      &l_emlrtBCI, (emlrtCTX)sp);
      }
      i2 = static_cast<int32_T>(d) - 1;
      if ((static_cast<int32_T>(d1) < 1) ||
          (static_cast<int32_T>(d1) > Phi.size(1))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d1), 1, Phi.size(1),
                                      &k_emlrtBCI, (emlrtCTX)sp);
      }
      i3 = static_cast<int32_T>(d1);
    }
    d = static_cast<real_T>(M) * ((static_cast<real_T>(p) + 1.0) + 1.0);
    if (d1 + 1.0 > d) {
      input_sizes_idx_1 = 0;
      i4 = 0;
    } else {
      if ((static_cast<int32_T>(d1 + 1.0) < 1) ||
          (static_cast<int32_T>(d1 + 1.0) > W.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d1 + 1.0), 1,
                                      W.size(0), &j_emlrtBCI, (emlrtCTX)sp);
      }
      input_sizes_idx_1 = static_cast<int32_T>(d1 + 1.0) - 1;
      if ((static_cast<int32_T>(d) < 1) ||
          (static_cast<int32_T>(d) > W.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, W.size(0),
                                      &i_emlrtBCI, (emlrtCTX)sp);
      }
      i4 = static_cast<int32_T>(d);
    }
    st.site = &j_emlrtRSI;
    loop_ub = Phi.size(0);
    b_loop_ub = i3 - i2;
    Kalman_gain.set_size(&cb_emlrtRTEI, &st, Phi.size(0), b_loop_ub);
    for (i5 = 0; i5 < b_loop_ub; i5++) {
      for (i6 = 0; i6 < loop_ub; i6++) {
        Kalman_gain[i6 + Kalman_gain.size(0) * i5] =
            Phi[i6 + Phi.size(0) * (i2 + i5)];
      }
    }
    loop_ub = W.size(1);
    sizes_idx_1 = i4 - input_sizes_idx_1;
    y.set_size(&db_emlrtRTEI, &st, sizes_idx_1, W.size(1));
    for (i5 = 0; i5 < loop_ub; i5++) {
      for (i6 = 0; i6 < sizes_idx_1; i6++) {
        y[i6 + y.size(0) * i5] = W[(input_sizes_idx_1 + i6) + W.size(0) * i5];
      }
    }
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, Kalman_gain, y, i3 - i2,
                               i4 - input_sizes_idx_1);
    loop_ub = Phi.size(0);
    Kalman_gain.set_size(&cb_emlrtRTEI, &st, Phi.size(0), b_loop_ub);
    for (i3 = 0; i3 < b_loop_ub; i3++) {
      for (i4 = 0; i4 < loop_ub; i4++) {
        Kalman_gain[i4 + Kalman_gain.size(0) * i3] =
            Phi[i4 + Phi.size(0) * (i2 + i3)];
      }
    }
    loop_ub = W.size(1);
    y.set_size(&db_emlrtRTEI, &st, sizes_idx_1, W.size(1));
    for (i2 = 0; i2 < loop_ub; i2++) {
      for (i3 = 0; i3 < sizes_idx_1; i3++) {
        y[i3 + y.size(0) * i2] = W[(input_sizes_idx_1 + i3) + W.size(0) * i2];
      }
    }
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, Kalman_gain, y, Sigma);
    iv[0] = (*(int32_T(*)[2])X.size())[0];
    iv[1] = (*(int32_T(*)[2])X.size())[1];
    b_result[0] = (*(int32_T(*)[2])Sigma.size())[0];
    b_result[1] = (*(int32_T(*)[2])Sigma.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_result[0], &n_emlrtECI, (emlrtCTX)sp);
    loop_ub = X.size(0) * X.size(1);
    for (i2 = 0; i2 < loop_ub; i2++) {
      X[i2] = X[i2] - Sigma[i2];
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  if (!(T >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &r_emlrtDCI, (emlrtCTX)sp);
  }
  d = static_cast<int32_T>(muDoubleScalarFloor(T));
  if (T != d) {
    emlrtIntegerCheckR2012b(T, &q_emlrtDCI, (emlrtCTX)sp);
  }
  W.set_size(&eb_emlrtRTEI, sp, static_cast<int32_T>(T), W.size(1));
  if (MP1 != i) {
    emlrtIntegerCheckR2012b(MP1, &s_emlrtDCI, (emlrtCTX)sp);
  }
  i1 = static_cast<int32_T>(MP1);
  W.set_size(&eb_emlrtRTEI, sp, W.size(0), i1);
  if (T != d) {
    emlrtIntegerCheckR2012b(T, &y_emlrtDCI, (emlrtCTX)sp);
  }
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &y_emlrtDCI, (emlrtCTX)sp);
  }
  loop_ub = static_cast<int32_T>(T) * i1;
  for (i2 = 0; i2 < loop_ub; i2++) {
    W[i2] = 0.0;
  }
  //  filtered values
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &t_emlrtDCI, (emlrtCTX)sp);
  }
  P_ttm.set_size(&fb_emlrtRTEI, sp, i1, P_ttm.size(1), P_ttm.size(2));
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &u_emlrtDCI, (emlrtCTX)sp);
  }
  P_ttm.set_size(&fb_emlrtRTEI, sp, P_ttm.size(0), i1, P_ttm.size(2));
  i2 = static_cast<int32_T>(muDoubleScalarFloor(T));
  if (T != i2) {
    emlrtIntegerCheckR2012b(T, &v_emlrtDCI, (emlrtCTX)sp);
  }
  i3 = static_cast<int32_T>(T);
  P_ttm.set_size(&fb_emlrtRTEI, sp, P_ttm.size(0), P_ttm.size(1), i3);
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &ab_emlrtDCI, (emlrtCTX)sp);
  }
  if (i3 != i2) {
    emlrtIntegerCheckR2012b(T, &ab_emlrtDCI, (emlrtCTX)sp);
  }
  loop_ub = i1 * i1 * i3;
  for (i = 0; i < loop_ub; i++) {
    P_ttm[i] = 0.0;
  }
  lnL = 0.0;
  i = static_cast<int32_T>(T + (1.0 - (P + 1.0)));
  emlrtForLoopVectorCheckR2021a(P + 1.0, 1.0, T, mxDOUBLE_CLASS, i,
                                &e_emlrtRTEI, (emlrtCTX)sp);
  if (0 <= i - 1) {
    i7 = Y0.size(1);
    c_loop_ub = Y0.size(1);
    d_loop_ub = YLm.size(0);
    i8 = YLm.size(1);
    e_loop_ub = YLm.size(1);
    i9 = YLm.size(1);
    result[0] = 1;
  }
  for (input_sizes_idx_1 = 0; input_sizes_idx_1 < i; input_sizes_idx_1++) {
    MP1 = (P + 1.0) + static_cast<real_T>(input_sizes_idx_1);
    d = MP1 - P;
    d1 = static_cast<int32_T>(muDoubleScalarFloor(d));
    if (d != d1) {
      emlrtIntegerCheckR2012b(d, &g_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(d) < 1) ||
        (static_cast<int32_T>(d) > Y0.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, Y0.size(0),
                                    &h_emlrtBCI, (emlrtCTX)sp);
    }
    y_t.set_size(&gb_emlrtRTEI, sp, i7);
    for (i1 = 0; i1 < c_loop_ub; i1++) {
      y_t[i1] = Y0[(static_cast<int32_T>(d) + Y0.size(0) * i1) - 1];
    }
    if (d != d1) {
      emlrtIntegerCheckR2012b(d, &f_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(d) < 1) ||
        (static_cast<int32_T>(d) > YLm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, YLm.size(2),
                                    &g_emlrtBCI, (emlrtCTX)sp);
    }
    Sigma.set_size(&hb_emlrtRTEI, sp, d_loop_ub, i8);
    for (i1 = 0; i1 < e_loop_ub; i1++) {
      for (i2 = 0; i2 < d_loop_ub; i2++) {
        Sigma[i2 + Sigma.size(0) * i1] =
            YLm[(i2 + YLm.size(0) * i1) +
                YLm.size(0) * YLm.size(1) * (static_cast<int32_T>(d) - 1)];
      }
    }
    st.site = &k_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, F, G_tt, F.size(1), G_tt.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, F, G_tt, G_tL);
    st.site = &l_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, F, P_tL, F.size(1), P_tL.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, F, P_tL, y);
    st.site = &l_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, F, y.size(1), F.size(1));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, y, F, P_tL);
    iv[0] = (*(int32_T(*)[2])P_tL.size())[0];
    iv[1] = (*(int32_T(*)[2])P_tL.size())[1];
    b_result[0] = (*(int32_T(*)[2])SIGMA.size())[0];
    b_result[1] = (*(int32_T(*)[2])SIGMA.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_result[0], &l_emlrtECI, (emlrtCTX)sp);
    loop_ub = P_tL.size(0) * P_tL.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      P_tL[i1] = P_tL[i1] + SIGMA[i1];
    }
    st.site = &m_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, Sigma, beta, i9, beta.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, Sigma, beta, y_tL);
    st.site = &m_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, X, G_tL, X.size(1), G_tL.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, X, G_tL, G_tt);
    if (y_tL.size(0) != G_tt.size(0)) {
      emlrtSizeEqCheck1DR2012b(y_tL.size(0), G_tt.size(0), &k_emlrtECI,
                               (emlrtCTX)sp);
    }
    loop_ub = y_tL.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      y_tL[i1] = y_tL[i1] + G_tt[i1];
    }
    st.site = &n_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, X, P_tL, X.size(1), P_tL.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, X, P_tL, y);
    st.site = &n_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, X, y.size(1), X.size(1));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, y, X, Sigma);
    iv[0] = (*(int32_T(*)[2])Sigma.size())[0];
    iv[1] = (*(int32_T(*)[2])Sigma.size())[1];
    b_result[0] =
        (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&Omega)->size())[0];
    b_result[1] =
        (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&Omega)->size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_result[0], &j_emlrtECI, (emlrtCTX)sp);
    loop_ub = Sigma.size(0) * Sigma.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      Sigma[i1] = Sigma[i1] + Omega[i1];
    }
    b_result[0] = Sigma.size(1);
    b_result[1] = Sigma.size(0);
    iv[0] = (*(int32_T(*)[2])Sigma.size())[0];
    iv[1] = (*(int32_T(*)[2])Sigma.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_result[0], &i_emlrtECI, (emlrtCTX)sp);
    Kalman_gain.set_size(&u_emlrtRTEI, sp, Sigma.size(0), Sigma.size(1));
    loop_ub = Sigma.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_loop_ub = Sigma.size(0);
      for (i2 = 0; i2 < b_loop_ub; i2++) {
        Kalman_gain[i2 + Kalman_gain.size(0) * i1] =
            0.5 *
            (Sigma[i2 + Sigma.size(0) * i1] + Sigma[i1 + Sigma.size(0) * i2]);
      }
    }
    st.site = &o_emlrtRSI;
    invpd(&st, Kalman_gain, Sigma);
    b_result[0] = Sigma.size(1);
    b_result[1] = Sigma.size(0);
    iv[0] = (*(int32_T(*)[2])Sigma.size())[0];
    iv[1] = (*(int32_T(*)[2])Sigma.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_result[0], &h_emlrtECI, (emlrtCTX)sp);
    Kalman_gain.set_size(&u_emlrtRTEI, sp, Sigma.size(0), Sigma.size(1));
    loop_ub = Sigma.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_loop_ub = Sigma.size(0);
      for (i2 = 0; i2 < b_loop_ub; i2++) {
        Kalman_gain[i2 + Kalman_gain.size(0) * i1] =
            0.5 *
            (Sigma[i2 + Sigma.size(0) * i1] + Sigma[i1 + Sigma.size(0) * i2]);
      }
    }
    Sigma.set_size(&ib_emlrtRTEI, sp, Kalman_gain.size(0), Kalman_gain.size(1));
    loop_ub = Kalman_gain.size(0) * Kalman_gain.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      Sigma[i1] = Kalman_gain[i1];
    }
    st.site = &p_emlrtRSI;
    lnL += lnpdfmvn1(&st, y_t, y_tL, Sigma);
    st.site = &q_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, P_tL, X, P_tL.size(1), X.size(1));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, P_tL, X, y);
    st.site = &q_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, Sigma, y.size(1), Sigma.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, Sigma, Kalman_gain);
    if (y_t.size(0) != y_tL.size(0)) {
      emlrtSizeEqCheck1DR2012b(y_t.size(0), y_tL.size(0), &g_emlrtECI,
                               (emlrtCTX)sp);
    }
    st.site = &r_emlrtRSI;
    loop_ub = y_t.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      y_t[i1] = y_t[i1] - y_tL[i1];
    }
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, Kalman_gain, y_t, Kalman_gain.size(1),
                               y_t.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, Kalman_gain, y_t, G_tt);
    if (G_tL.size(0) != G_tt.size(0)) {
      emlrtSizeEqCheck1DR2012b(G_tL.size(0), G_tt.size(0), &f_emlrtECI,
                               (emlrtCTX)sp);
    }
    G_tt.set_size(&jb_emlrtRTEI, sp, G_tL.size(0));
    loop_ub = G_tL.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      G_tt[i1] = G_tL[i1] + G_tt[i1];
    }
    st.site = &s_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, Kalman_gain, X, Kalman_gain.size(1),
                               X.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, Kalman_gain, X, y);
    st.site = &s_emlrtRSI;
    b_st.site = &eb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, P_tL, y.size(1), P_tL.size(0));
    b_st.site = &db_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, P_tL, Sigma);
    iv[0] = (*(int32_T(*)[2])P_tL.size())[0];
    iv[1] = (*(int32_T(*)[2])P_tL.size())[1];
    b_result[0] = (*(int32_T(*)[2])Sigma.size())[0];
    b_result[1] = (*(int32_T(*)[2])Sigma.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_result[0], &e_emlrtECI, (emlrtCTX)sp);
    loop_ub = P_tL.size(0) * P_tL.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      P_tL[i1] = P_tL[i1] - Sigma[i1];
    }
    if (MP1 != static_cast<int32_T>(muDoubleScalarFloor(MP1))) {
      emlrtIntegerCheckR2012b(MP1, &c_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(MP1) < 1) ||
        (static_cast<int32_T>(MP1) > W.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(MP1), 1, W.size(0),
                                    &d_emlrtBCI, (emlrtCTX)sp);
    }
    result[1] = W.size(1);
    b_result[0] = 1;
    b_result[1] = G_tt.size(0);
    emlrtSubAssignSizeCheckR2012b(&result[0], 2, &b_result[0], 2, &d_emlrtECI,
                                  (emlrtCTX)sp);
    loop_ub = G_tt.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      W[(static_cast<int32_T>(MP1) + W.size(0) * i1) - 1] = G_tt[i1];
    }
    if ((static_cast<int32_T>(MP1) < 1) ||
        (static_cast<int32_T>(MP1) > P_ttm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(MP1), 1, P_ttm.size(2),
                                    &c_emlrtBCI, (emlrtCTX)sp);
    }
    b_result[0] = P_ttm.size(0);
    b_result[1] = P_ttm.size(1);
    emlrtSubAssignSizeCheckR2012b(&b_result[0], 2, P_tL.size(), 2, &c_emlrtECI,
                                  (emlrtCTX)sp);
    loop_ub = P_tL.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_loop_ub = P_tL.size(0);
      for (i2 = 0; i2 < b_loop_ub; i2++) {
        P_ttm[(i2 + P_ttm.size(0) * i1) +
              P_ttm.size(0) * P_ttm.size(1) * (static_cast<int32_T>(MP1) - 1)] =
            P_tL[i2 + P_tL.size(0) * i1];
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
  return lnL;
}

// End of code generation (lnlik_NonLinear.cpp)
