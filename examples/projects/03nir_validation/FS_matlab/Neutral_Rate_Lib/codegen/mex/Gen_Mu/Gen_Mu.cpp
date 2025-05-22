//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Mu.cpp
//
// Code generation for function 'Gen_Mu'
//

// Include files
#include "Gen_Mu.h"
#include "Gen_Mu_data.h"
#include "chol.h"
#include "cholmod.h"
#include "diag.h"
#include "eml_mtimes_helper.h"
#include "eye.h"
#include "indexShapeCheck.h"
#include "inv.h"
#include "mtimes.h"
#include "randn.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo
    emlrtRSI{
        5,        // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    b_emlrtRSI{
        23,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    c_emlrtRSI{
        25,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    d_emlrtRSI{
        26,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    e_emlrtRSI{
        28,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    f_emlrtRSI{
        29,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    g_emlrtRSI{
        31,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    h_emlrtRSI{
        35,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    i_emlrtRSI{
        53,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    j_emlrtRSI{
        54,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    k_emlrtRSI{
        55,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    l_emlrtRSI{
        56,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    m_emlrtRSI{
        57,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    n_emlrtRSI{
        58,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    o_emlrtRSI{
        59,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    p_emlrtRSI{
        78,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    q_emlrtRSI{
        79,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    r_emlrtRSI{
        86,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    s_emlrtRSI{
        87,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    t_emlrtRSI{
        88,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    u_emlrtRSI{
        91,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo
    v_emlrtRSI{
        93,       // lineNo
        "Gen_Mu", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pathName
    };

static emlrtRSInfo db_emlrtRSI{
    24,    // lineNo
    "cat", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRSInfo eb_emlrtRSI{
    96,         // lineNo
    "cat_impl", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRSInfo gc_emlrtRSI{
    34,     // lineNo
    "chol", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\chol.m" // pathName
};

static emlrtRTEInfo o_emlrtRTEI{
    271,                   // lineNo
    27,                    // colNo
    "check_non_axis_size", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pName
};

static emlrtECInfo
    h_emlrtECI{
        -1,       // nDims
        93,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    v_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        93,       // lineNo
        12,       // colNo
        "Mu",     // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    w_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        93,       // lineNo
        25,       // colNo
        "G_t",    // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    x_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        93,       // lineNo
        23,       // colNo
        "G_t",    // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    i_emlrtECI{
        -1,       // nDims
        91,       // lineNo
        15,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    j_emlrtECI{
        2,        // nDims
        90,       // lineNo
        18,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    k_emlrtECI{
        2,        // nDims
        88,       // lineNo
        17,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    l_emlrtECI{
        -1,       // nDims
        87,       // lineNo
        17,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    m_emlrtECI{
        -1,       // nDims
        87,       // lineNo
        48,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    y_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        87,       // lineNo
        23,       // colNo
        "G_ttm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    ab_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        85,       // lineNo
        22,       // colNo
        "G_tLm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    n_emlrtECI{
        -1,       // nDims
        61,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    bb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        61,       // lineNo
        15,       // colNo
        "G_tLm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    c_emlrtDCI{
        61,       // lineNo
        15,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtBCInfo
    cb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        84,       // lineNo
        28,       // colNo
        "P_tLm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    o_emlrtECI{
        -1,       // nDims
        62,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    db_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        62,       // lineNo
        20,       // colNo
        "P_tLm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    eb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        83,       // lineNo
        28,       // colNo
        "P_ttm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtRTEInfo
    p_emlrtRTEI{
        81,       // lineNo
        13,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    p_emlrtECI{
        -1,       // nDims
        79,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    fb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        79,       // lineNo
        8,        // colNo
        "Mu",     // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    gb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        79,       // lineNo
        23,       // colNo
        "G_t1",   // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    hb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        79,       // lineNo
        21,       // colNo
        "G_t1",   // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    q_emlrtECI{
        -1,       // nDims
        78,       // lineNo
        12,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    ib_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        76,       // lineNo
        24,       // colNo
        "P_ttm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    r_emlrtECI{
        -1,       // nDims
        65,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    jb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        65,       // lineNo
        21,       // colNo
        "P_ttm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    kb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        75,       // lineNo
        18,       // colNo
        "G_ttm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    s_emlrtECI{
        -1,       // nDims
        64,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    lb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        64,       // lineNo
        15,       // colNo
        "G_ttm",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    t_emlrtECI{
        2,        // nDims
        59,       // lineNo
        16,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    u_emlrtECI{
        -1,       // nDims
        58,       // lineNo
        16,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    v_emlrtECI{
        -1,       // nDims
        58,       // lineNo
        46,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    w_emlrtECI{
        2,        // nDims
        56,       // lineNo
        16,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    x_emlrtECI{
        -1,       // nDims
        55,       // lineNo
        16,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    y_emlrtECI{
        2,        // nDims
        54,       // lineNo
        16,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    ab_emlrtECI{
        -1,       // nDims
        20,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    mb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        20,       // lineNo
        20,       // colNo
        "G_LL",   // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    d_emlrtDCI{
        20,       // lineNo
        20,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtBCInfo
    nb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        20,       // lineNo
        10,       // colNo
        "G_LL",   // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    e_emlrtDCI{
        20,       // lineNo
        10,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtBCInfo
    ob_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        52,       // lineNo
        25,       // colNo
        "YLm",    // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    f_emlrtDCI{
        52,       // lineNo
        25,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtBCInfo
    pb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        51,       // lineNo
        18,       // colNo
        "Y0",     // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    g_emlrtDCI{
        51,       // lineNo
        18,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtRTEInfo
    q_emlrtRTEI{
        50,       // lineNo
        13,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    bb_emlrtECI{
        2,        // nDims
        35,       // lineNo
        13,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    qb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        35,       // lineNo
        51,       // colNo
        "X_mat",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    rb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        35,       // lineNo
        45,       // colNo
        "X_mat",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    sb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        35,       // lineNo
        34,       // colNo
        "Phi",    // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    tb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        35,       // lineNo
        24,       // colNo
        "Phi",    // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtRTEInfo
    r_emlrtRTEI{
        34,       // lineNo
        13,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    ub_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        33,       // lineNo
        17,       // colNo
        "X_mat",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    vb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        33,       // lineNo
        15,       // colNo
        "X_mat",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    h_emlrtDCI{
        26,       // lineNo
        20,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    i_emlrtDCI{
        26,       // lineNo
        20,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        4       // checkKind
    };

static emlrtDCInfo
    j_emlrtDCI{
        25,       // lineNo
        44,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    k_emlrtDCI{
        25,       // lineNo
        44,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        4       // checkKind
    };

static emlrtDCInfo
    l_emlrtDCI{
        25,       // lineNo
        22,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    m_emlrtDCI{
        25,       // lineNo
        22,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        4       // checkKind
    };

static emlrtBCInfo
    wb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        20,       // lineNo
        30,       // colNo
        "Mu",     // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    n_emlrtDCI{
        20,       // lineNo
        30,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtRTEInfo
    s_emlrtRTEI{
        19,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtECInfo
    cb_emlrtECI{
        -1,       // nDims
        12,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtBCInfo
    xb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        12,       // lineNo
        15,       // colNo
        "Mu",     // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    o_emlrtDCI{
        12,       // lineNo
        26,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    p_emlrtDCI{
        12,       // lineNo
        26,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        4       // checkKind
    };

static emlrtDCInfo
    q_emlrtDCI{
        38,       // lineNo
        19,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    r_emlrtDCI{
        38,       // lineNo
        19,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        4       // checkKind
    };

static emlrtDCInfo
    s_emlrtDCI{
        38,       // lineNo
        22,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    t_emlrtDCI{
        44,       // lineNo
        19,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    u_emlrtDCI{
        44,       // lineNo
        24,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    v_emlrtDCI{
        44,       // lineNo
        29,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    w_emlrtDCI{
        46,       // lineNo
        19,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    x_emlrtDCI{
        46,       // lineNo
        22,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    y_emlrtDCI{
        47,       // lineNo
        19,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    ab_emlrtDCI{
        47,       // lineNo
        24,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    bb_emlrtDCI{
        47,       // lineNo
        29,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtBCInfo
    yb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        11,       // lineNo
        8,        // colNo
        "gamma",  // aName
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        0       // checkKind
    };

static emlrtDCInfo
    cb_emlrtDCI{
        18,       // lineNo
        1,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    db_emlrtDCI{
        18,       // lineNo
        1,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        4       // checkKind
    };

static emlrtDCInfo
    eb_emlrtDCI{
        38,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    fb_emlrtDCI{
        44,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    gb_emlrtDCI{
        46,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtDCInfo
    hb_emlrtDCI{
        47,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m", // pName
        1       // checkKind
    };

static emlrtRTEInfo
    gc_emlrtRTEI{
        18,       // lineNo
        1,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo hc_emlrtRTEI{
    24,    // lineNo
    5,     // colNo
    "cat", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pName
};

static emlrtRTEInfo
    ic_emlrtRTEI{
        25,       // lineNo
        1,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    jc_emlrtRTEI{
        26,       // lineNo
        1,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    kc_emlrtRTEI{
        33,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    lc_emlrtRTEI{
        35,       // lineNo
        17,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    mc_emlrtRTEI{
        35,       // lineNo
        39,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    nc_emlrtRTEI{
        38,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    oc_emlrtRTEI{
        44,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    pc_emlrtRTEI{
        46,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    qc_emlrtRTEI{
        47,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    rc_emlrtRTEI{
        51,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    sc_emlrtRTEI{
        52,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    tc_emlrtRTEI{
        56,       // lineNo
        16,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    uc_emlrtRTEI{
        58,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    vc_emlrtRTEI{
        59,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    wc_emlrtRTEI{
        75,       // lineNo
        5,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo xc_emlrtRTEI{
    34,     // lineNo
    5,      // colNo
    "chol", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\chol.m" // pName
};

static emlrtRTEInfo
    yc_emlrtRTEI{
        85,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    ad_emlrtRTEI{
        84,       // lineNo
        16,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    bd_emlrtRTEI{
        87,       // lineNo
        17,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    cd_emlrtRTEI{
        83,       // lineNo
        16,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    dd_emlrtRTEI{
        88,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    ed_emlrtRTEI{
        90,       // lineNo
        17,       // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    fd_emlrtRTEI{
        90,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

static emlrtRTEInfo
    gd_emlrtRTEI{
        91,       // lineNo
        9,        // colNo
        "Gen_Mu", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Mu.m" // pName
    };

// Function Definitions
void Gen_Mu(const emlrtStack *sp, coder::array<real_T, 2U> &Mu,
            const coder::array<real_T, 2U> &Y0,
            const coder::array<real_T, 3U> &YLm,
            const coder::array<real_T, 1U> &beta,
            const coder::array<real_T, 2U> &Phi,
            const coder::array<real_T, 2U> &Omega,
            const coder::array<real_T, 1U> &diag_Sigma,
            const coder::array<real_T, 1U> &b_gamma,
            coder::array<real_T, 2U> &G_ttm)
{
  coder::array<real_T, 3U> P_tLm;
  coder::array<real_T, 3U> P_ttm;
  coder::array<real_T, 2U> F;
  coder::array<real_T, 2U> G_tLm;
  coder::array<real_T, 2U> P_tt;
  coder::array<real_T, 2U> SIGMA;
  coder::array<real_T, 2U> Sigma;
  coder::array<real_T, 2U> W;
  coder::array<real_T, 2U> X;
  coder::array<real_T, 2U> b_Phi;
  coder::array<real_T, 2U> varargin_1;
  coder::array<real_T, 2U> y;
  coder::array<real_T, 1U> G_tL;
  coder::array<real_T, 1U> G_tt;
  coder::array<real_T, 1U> G_tt1;
  coder::array<real_T, 1U> r;
  coder::array<real_T, 1U> y_t;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T MP1;
  real_T P;
  real_T T;
  real_T b_p;
  real_T d;
  real_T d1;
  int32_T b_result[2];
  int32_T iv[2];
  int32_T result[2];
  int32_T M;
  int32_T i;
  int32_T i1;
  int32_T i2;
  int32_T i3;
  int32_T i4;
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
  M = YLm.size(0);
  st.site = &emlrtRSI;
  b_st.site = &w_emlrtRSI;
  P = static_cast<real_T>(YLm.size(1)) /
      (static_cast<real_T>(YLm.size(0)) * static_cast<real_T>(YLm.size(0)));
  T = static_cast<real_T>(YLm.size(2)) + P;
  MP1 = static_cast<real_T>(YLm.size(0)) * (P + 1.0);
  i = YLm.size(0);
  for (sizes_idx_1 = 0; sizes_idx_1 < i; sizes_idx_1++) {
    if (sizes_idx_1 + 1 > b_gamma.size(0)) {
      emlrtDynamicBoundsCheckR2012b(sizes_idx_1 + 1, 1, b_gamma.size(0),
                                    &yb_emlrtBCI, (emlrtCTX)sp);
    }
    if (b_gamma[sizes_idx_1] == 0.0) {
      i1 = Mu.size(1);
      if (sizes_idx_1 + 1 > i1) {
        emlrtDynamicBoundsCheckR2012b(sizes_idx_1 + 1, 1, i1, &xb_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      if (!(T >= 0.0)) {
        emlrtNonNegativeCheckR2012b(rtNaN, &p_emlrtDCI, (emlrtCTX)sp);
      }
      if (T != static_cast<int32_T>(muDoubleScalarFloor(T))) {
        emlrtIntegerCheckR2012b(T, &o_emlrtDCI, (emlrtCTX)sp);
      }
      input_sizes_idx_1 = static_cast<int32_T>(T);
      emlrtSubAssignSizeCheckR2012b(Mu.size(), 1, &input_sizes_idx_1, 1,
                                    &cb_emlrtECI, (emlrtCTX)sp);
      for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
        Mu[i1 + Mu.size(0) * sizes_idx_1] = 0.0;
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  //  Mu(1:P, :) = zeros(P, M);
  //  상태변수의 초기값은 주어진 것으로 처리
  if (!(MP1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &db_emlrtDCI, (emlrtCTX)sp);
  }
  i = static_cast<int32_T>(muDoubleScalarFloor(MP1));
  if (MP1 != i) {
    emlrtIntegerCheckR2012b(MP1, &cb_emlrtDCI, (emlrtCTX)sp);
  }
  G_tt.set_size(&gc_emlrtRTEI, sp, static_cast<int32_T>(MP1));
  if (MP1 != i) {
    emlrtIntegerCheckR2012b(MP1, &cb_emlrtDCI, (emlrtCTX)sp);
  }
  loop_ub = static_cast<int32_T>(MP1);
  for (i1 = 0; i1 < loop_ub; i1++) {
    G_tt[i1] = 0.0;
  }
  i1 = static_cast<int32_T>(((-1.0 - (P + 1.0)) + 1.0) / -1.0);
  emlrtForLoopVectorCheckR2021a(P + 1.0, -1.0, 1.0, mxDOUBLE_CLASS, i1,
                                &s_emlrtRTEI, (emlrtCTX)sp);
  for (p = 0; p < i1; p++) {
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
                                      &nb_emlrtBCI, (emlrtCTX)sp);
      }
      i2 = static_cast<int32_T>(d) - 2;
      if (d1 != static_cast<int32_T>(muDoubleScalarFloor(d1))) {
        emlrtIntegerCheckR2012b(d1, &d_emlrtDCI, (emlrtCTX)sp);
      }
      if ((static_cast<int32_T>(d1) < 1) ||
          (static_cast<int32_T>(d1) > G_tt.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d1), 1, G_tt.size(0),
                                      &mb_emlrtBCI, (emlrtCTX)sp);
      }
      i3 = static_cast<int32_T>(d1) - 1;
    }
    i4 = Mu.size(0);
    if (b_p != static_cast<int32_T>(muDoubleScalarFloor(b_p))) {
      emlrtIntegerCheckR2012b(b_p, &n_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(b_p) < 1) || (static_cast<int32_T>(b_p) > i4)) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, i4,
                                    &wb_emlrtBCI, (emlrtCTX)sp);
    }
    loop_ub = i3 - i2;
    i3 = Mu.size(1);
    if (loop_ub != i3) {
      emlrtSubAssignSizeCheck1dR2017a(loop_ub, i3, &ab_emlrtECI, (emlrtCTX)sp);
    }
    for (i3 = 0; i3 < loop_ub; i3++) {
      G_tt[(i2 + i3) + 1] =
          Mu[(static_cast<int32_T>(b_p) + Mu.size(0) * i3) - 1];
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  st.site = &b_emlrtRSI;
  coder::eye(&st, MP1, P_tt);
  loop_ub = P_tt.size(0) * P_tt.size(1);
  for (i1 = 0; i1 < loop_ub; i1++) {
    P_tt[i1] = 0.1 * P_tt[i1];
  }
  st.site = &c_emlrtRSI;
  b_st.site = &c_emlrtRSI;
  coder::eye(&b_st, static_cast<real_T>(YLm.size(0)), varargin_1);
  d = static_cast<real_T>(YLm.size(0)) * P;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &m_emlrtDCI, &st);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &l_emlrtDCI, &st);
  }
  b_st.site = &db_emlrtRSI;
  if ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0)) {
    loop_ub = varargin_1.size(0);
  } else if ((YLm.size(0) != 0) &&
             (static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) !=
              0)) {
    loop_ub = YLm.size(0);
  } else {
    loop_ub = varargin_1.size(0);
    if (YLm.size(0) > varargin_1.size(0)) {
      loop_ub = YLm.size(0);
    }
  }
  c_st.site = &eb_emlrtRSI;
  if ((varargin_1.size(0) != loop_ub) &&
      ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &o_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((YLm.size(0) != loop_ub) &&
      ((YLm.size(0) != 0) &&
       (static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &o_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (loop_ub == 0);
  if (empty_non_axis_sizes ||
      ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0))) {
    input_sizes_idx_1 = varargin_1.size(1);
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
  Sigma.set_size(&hc_emlrtRTEI, &b_st, loop_ub,
                 input_sizes_idx_1 + sizes_idx_1);
  for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      Sigma[i2 + Sigma.size(0) * i1] = varargin_1[i2 + loop_ub * i1];
    }
  }
  for (i1 = 0; i1 < sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      Sigma[i2 + Sigma.size(0) * (i1 + input_sizes_idx_1)] = 0.0;
    }
  }
  st.site = &c_emlrtRSI;
  b_st.site = &c_emlrtRSI;
  coder::eye(&b_st, static_cast<real_T>(YLm.size(0)) * P, varargin_1);
  d = static_cast<real_T>(YLm.size(0)) * P;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &k_emlrtDCI, &st);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &j_emlrtDCI, &st);
  }
  b_st.site = &db_emlrtRSI;
  if ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0)) {
    loop_ub = varargin_1.size(0);
  } else if ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) !=
              0) &&
             (YLm.size(0) != 0)) {
    loop_ub = static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P);
  } else {
    loop_ub = varargin_1.size(0);
    if (static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) >
        varargin_1.size(0)) {
      loop_ub = static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P);
    }
  }
  c_st.site = &eb_emlrtRSI;
  if ((varargin_1.size(0) != loop_ub) &&
      ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &o_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != loop_ub) &&
      ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0) &&
       (YLm.size(0) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &o_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (loop_ub == 0);
  if (empty_non_axis_sizes ||
      ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0))) {
    input_sizes_idx_1 = varargin_1.size(1);
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
  W.set_size(&hc_emlrtRTEI, &b_st, loop_ub, input_sizes_idx_1 + sizes_idx_1);
  for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      W[i2 + W.size(0) * i1] = varargin_1[i2 + loop_ub * i1];
    }
  }
  for (i1 = 0; i1 < sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      W[i2 + W.size(0) * (i1 + input_sizes_idx_1)] = 0.0;
    }
  }
  st.site = &c_emlrtRSI;
  b_st.site = &db_emlrtRSI;
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
  c_st.site = &eb_emlrtRSI;
  if ((Sigma.size(1) != loop_ub) &&
      ((Sigma.size(0) != 0) && (Sigma.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &o_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((W.size(1) != loop_ub) && ((W.size(0) != 0) && (W.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &o_emlrtRTEI,
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
  F.set_size(&ic_emlrtRTEI, &b_st, input_sizes_idx_1 + sizes_idx_1, loop_ub);
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
  st.site = &d_emlrtRSI;
  b_st.site = &d_emlrtRSI;
  coder::eye(&b_st, static_cast<real_T>(YLm.size(0)), varargin_1);
  d = static_cast<real_T>(YLm.size(0)) * P;
  if (!(d >= 0.0)) {
    emlrtNonNegativeCheckR2012b(rtNaN, &i_emlrtDCI, &st);
  }
  if (d != static_cast<int32_T>(muDoubleScalarFloor(d))) {
    emlrtIntegerCheckR2012b(d, &h_emlrtDCI, &st);
  }
  b_st.site = &db_emlrtRSI;
  if ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0)) {
    loop_ub = varargin_1.size(1);
  } else if ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) !=
              0) &&
             (YLm.size(0) != 0)) {
    loop_ub = YLm.size(0);
  } else {
    loop_ub = varargin_1.size(1);
    if (YLm.size(0) > varargin_1.size(1)) {
      loop_ub = YLm.size(0);
    }
  }
  c_st.site = &eb_emlrtRSI;
  if ((varargin_1.size(1) != loop_ub) &&
      ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &o_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((YLm.size(0) != loop_ub) &&
      ((static_cast<int32_T>(static_cast<real_T>(YLm.size(0)) * P) != 0) &&
       (YLm.size(0) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &o_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (loop_ub == 0);
  if (empty_non_axis_sizes ||
      ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0))) {
    input_sizes_idx_1 = varargin_1.size(0);
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
  W.set_size(&jc_emlrtRTEI, &b_st, input_sizes_idx_1 + sizes_idx_1, loop_ub);
  for (i1 = 0; i1 < loop_ub; i1++) {
    for (i2 = 0; i2 < input_sizes_idx_1; i2++) {
      W[i2 + W.size(0) * i1] = varargin_1[i2 + input_sizes_idx_1 * i1];
    }
  }
  for (i1 = 0; i1 < loop_ub; i1++) {
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      W[(i2 + input_sizes_idx_1) + W.size(0) * i1] = 0.0;
    }
  }
  st.site = &e_emlrtRSI;
  coder::diag(&st, diag_Sigma, Sigma);
  //  M by M
  st.site = &f_emlrtRSI;
  b_st.site = &hb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, W, Sigma, W.size(1), Sigma.size(0));
  b_st.site = &gb_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, W, Sigma, y);
  st.site = &f_emlrtRSI;
  b_st.site = &hb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, y, W, y.size(1), W.size(1));
  b_st.site = &gb_emlrtRSI;
  coder::internal::blas::b_mtimes(&b_st, y, W, SIGMA);
  st.site = &g_emlrtRSI;
  coder::eye(&st, MP1, W);
  if (1 > YLm.size(0)) {
    loop_ub = 0;
  } else {
    if (1 > W.size(0)) {
      emlrtDynamicBoundsCheckR2012b(1, 1, W.size(0), &vb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (YLm.size(0) > W.size(0)) {
      emlrtDynamicBoundsCheckR2012b(YLm.size(0), 1, W.size(0), &ub_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    loop_ub = YLm.size(0);
  }
  sizes_idx_1 = W.size(1);
  X.set_size(&kc_emlrtRTEI, sp, loop_ub, W.size(1));
  for (i1 = 0; i1 < sizes_idx_1; i1++) {
    for (i2 = 0; i2 < loop_ub; i2++) {
      X[i2 + X.size(0) * i1] = W[i2 + W.size(0) * i1];
    }
  }
  i1 = static_cast<int32_T>(P);
  emlrtForLoopVectorCheckR2021a(1.0, 1.0, P, mxDOUBLE_CLASS,
                                static_cast<int32_T>(P), &r_emlrtRTEI,
                                (emlrtCTX)sp);
  for (p = 0; p < i1; p++) {
    int32_T i5;
    int32_T i6;
    int32_T i7;
    d = static_cast<real_T>(M) * ((static_cast<real_T>(p) + 1.0) - 1.0) + 1.0;
    d1 = static_cast<real_T>(M) * (static_cast<real_T>(p) + 1.0);
    if (d > d1) {
      i2 = 0;
      i3 = 0;
    } else {
      if ((static_cast<int32_T>(d) < 1) ||
          (static_cast<int32_T>(d) > Phi.size(1))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, Phi.size(1),
                                      &tb_emlrtBCI, (emlrtCTX)sp);
      }
      i2 = static_cast<int32_T>(d) - 1;
      if ((static_cast<int32_T>(d1) < 1) ||
          (static_cast<int32_T>(d1) > Phi.size(1))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d1), 1, Phi.size(1),
                                      &sb_emlrtBCI, (emlrtCTX)sp);
      }
      i3 = static_cast<int32_T>(d1);
    }
    d = static_cast<real_T>(M) * ((static_cast<real_T>(p) + 1.0) + 1.0);
    if (d1 + 1.0 > d) {
      i4 = 0;
      i5 = 0;
    } else {
      if ((static_cast<int32_T>(d1 + 1.0) < 1) ||
          (static_cast<int32_T>(d1 + 1.0) > W.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d1 + 1.0), 1,
                                      W.size(0), &rb_emlrtBCI, (emlrtCTX)sp);
      }
      i4 = static_cast<int32_T>(d1 + 1.0) - 1;
      if ((static_cast<int32_T>(d) < 1) ||
          (static_cast<int32_T>(d) > W.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, W.size(0),
                                      &qb_emlrtBCI, (emlrtCTX)sp);
      }
      i5 = static_cast<int32_T>(d);
    }
    st.site = &h_emlrtRSI;
    loop_ub = Phi.size(0);
    sizes_idx_1 = i3 - i2;
    b_Phi.set_size(&lc_emlrtRTEI, &st, Phi.size(0), sizes_idx_1);
    for (i6 = 0; i6 < sizes_idx_1; i6++) {
      for (i7 = 0; i7 < loop_ub; i7++) {
        b_Phi[i7 + b_Phi.size(0) * i6] = Phi[i7 + Phi.size(0) * (i2 + i6)];
      }
    }
    loop_ub = W.size(1);
    input_sizes_idx_1 = i5 - i4;
    y.set_size(&mc_emlrtRTEI, &st, input_sizes_idx_1, W.size(1));
    for (i6 = 0; i6 < loop_ub; i6++) {
      for (i7 = 0; i7 < input_sizes_idx_1; i7++) {
        y[i7 + y.size(0) * i6] = W[(i4 + i7) + W.size(0) * i6];
      }
    }
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, b_Phi, y, i3 - i2, i5 - i4);
    loop_ub = Phi.size(0);
    b_Phi.set_size(&lc_emlrtRTEI, &st, Phi.size(0), sizes_idx_1);
    for (i3 = 0; i3 < sizes_idx_1; i3++) {
      for (i5 = 0; i5 < loop_ub; i5++) {
        b_Phi[i5 + b_Phi.size(0) * i3] = Phi[i5 + Phi.size(0) * (i2 + i3)];
      }
    }
    loop_ub = W.size(1);
    y.set_size(&mc_emlrtRTEI, &st, input_sizes_idx_1, W.size(1));
    for (i2 = 0; i2 < loop_ub; i2++) {
      for (i3 = 0; i3 < input_sizes_idx_1; i3++) {
        y[i3 + y.size(0) * i2] = W[(i4 + i3) + W.size(0) * i2];
      }
    }
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, b_Phi, y, varargin_1);
    iv[0] = (*(int32_T(*)[2])X.size())[0];
    iv[1] = (*(int32_T(*)[2])X.size())[1];
    result[0] = (*(int32_T(*)[2])varargin_1.size())[0];
    result[1] = (*(int32_T(*)[2])varargin_1.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &result[0], &bb_emlrtECI, (emlrtCTX)sp);
    loop_ub = X.size(0) * X.size(1);
    for (i2 = 0; i2 < loop_ub; i2++) {
      X[i2] = X[i2] - varargin_1[i2];
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
  G_ttm.set_size(&nc_emlrtRTEI, sp, static_cast<int32_T>(T), G_ttm.size(1));
  if (MP1 != i) {
    emlrtIntegerCheckR2012b(MP1, &s_emlrtDCI, (emlrtCTX)sp);
  }
  i1 = static_cast<int32_T>(MP1);
  G_ttm.set_size(&nc_emlrtRTEI, sp, G_ttm.size(0), i1);
  if (T != d) {
    emlrtIntegerCheckR2012b(T, &eb_emlrtDCI, (emlrtCTX)sp);
  }
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &eb_emlrtDCI, (emlrtCTX)sp);
  }
  loop_ub = static_cast<int32_T>(T) * i1;
  for (i2 = 0; i2 < loop_ub; i2++) {
    G_ttm[i2] = 0.0;
  }
  //  filtered values
  //      G_ttm(1:P, :) = kron(ones(P, P+1), Mu(1, :));
  //      for p = 1:(P+1)
  //          G_ttm(p:end, (p-1)*M+1:p*M) = Mu(1:end+1-p, :);
  //      end
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &t_emlrtDCI, (emlrtCTX)sp);
  }
  P_ttm.set_size(&oc_emlrtRTEI, sp, i1, P_ttm.size(1), P_ttm.size(2));
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &u_emlrtDCI, (emlrtCTX)sp);
  }
  P_ttm.set_size(&oc_emlrtRTEI, sp, P_ttm.size(0), i1, P_ttm.size(2));
  i2 = static_cast<int32_T>(muDoubleScalarFloor(T));
  if (T != i2) {
    emlrtIntegerCheckR2012b(T, &v_emlrtDCI, (emlrtCTX)sp);
  }
  i3 = static_cast<int32_T>(T);
  P_ttm.set_size(&oc_emlrtRTEI, sp, P_ttm.size(0), P_ttm.size(1), i3);
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &fb_emlrtDCI, (emlrtCTX)sp);
  }
  if (i3 != i2) {
    emlrtIntegerCheckR2012b(T, &fb_emlrtDCI, (emlrtCTX)sp);
  }
  input_sizes_idx_1 = i1 * i1 * i3;
  for (i4 = 0; i4 < input_sizes_idx_1; i4++) {
    P_ttm[i4] = 0.0;
  }
  if (i3 != i2) {
    emlrtIntegerCheckR2012b(T, &w_emlrtDCI, (emlrtCTX)sp);
  }
  G_tLm.set_size(&pc_emlrtRTEI, sp, i3, G_tLm.size(1));
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &x_emlrtDCI, (emlrtCTX)sp);
  }
  G_tLm.set_size(&pc_emlrtRTEI, sp, G_tLm.size(0), i1);
  if (i3 != i2) {
    emlrtIntegerCheckR2012b(T, &gb_emlrtDCI, (emlrtCTX)sp);
  }
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &gb_emlrtDCI, (emlrtCTX)sp);
  }
  loop_ub = static_cast<int32_T>(T) * static_cast<int32_T>(MP1);
  for (i4 = 0; i4 < loop_ub; i4++) {
    G_tLm[i4] = 0.0;
  }
  //  predictive values
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &y_emlrtDCI, (emlrtCTX)sp);
  }
  P_tLm.set_size(&qc_emlrtRTEI, sp, i1, P_tLm.size(1), P_tLm.size(2));
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &ab_emlrtDCI, (emlrtCTX)sp);
  }
  P_tLm.set_size(&qc_emlrtRTEI, sp, P_tLm.size(0), i1, P_tLm.size(2));
  if (i3 != i2) {
    emlrtIntegerCheckR2012b(T, &bb_emlrtDCI, (emlrtCTX)sp);
  }
  P_tLm.set_size(&qc_emlrtRTEI, sp, P_tLm.size(0), P_tLm.size(1), i3);
  if (i1 != i) {
    emlrtIntegerCheckR2012b(MP1, &hb_emlrtDCI, (emlrtCTX)sp);
  }
  if (i3 != i2) {
    emlrtIntegerCheckR2012b(T, &hb_emlrtDCI, (emlrtCTX)sp);
  }
  for (i = 0; i < input_sizes_idx_1; i++) {
    P_tLm[i] = 0.0;
  }
  i = static_cast<int32_T>(T + (1.0 - (P + 1.0)));
  emlrtForLoopVectorCheckR2021a(P + 1.0, 1.0, T, mxDOUBLE_CLASS, i,
                                &q_emlrtRTEI, (emlrtCTX)sp);
  for (input_sizes_idx_1 = 0; input_sizes_idx_1 < i; input_sizes_idx_1++) {
    b_p = (P + 1.0) + static_cast<real_T>(input_sizes_idx_1);
    d = b_p - P;
    d1 = static_cast<int32_T>(muDoubleScalarFloor(d));
    if (d != d1) {
      emlrtIntegerCheckR2012b(d, &g_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(d) < 1) ||
        (static_cast<int32_T>(d) > Y0.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, Y0.size(0),
                                    &pb_emlrtBCI, (emlrtCTX)sp);
    }
    loop_ub = Y0.size(1);
    y_t.set_size(&rc_emlrtRTEI, sp, Y0.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      y_t[i1] = Y0[(static_cast<int32_T>(d) + Y0.size(0) * i1) - 1];
    }
    if (d != d1) {
      emlrtIntegerCheckR2012b(d, &f_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(d) < 1) ||
        (static_cast<int32_T>(d) > YLm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(d), 1, YLm.size(2),
                                    &ob_emlrtBCI, (emlrtCTX)sp);
    }
    loop_ub = YLm.size(0);
    sizes_idx_1 = YLm.size(1);
    Sigma.set_size(&sc_emlrtRTEI, sp, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < sizes_idx_1; i1++) {
      for (i2 = 0; i2 < loop_ub; i2++) {
        Sigma[i2 + Sigma.size(0) * i1] =
            YLm[(i2 + YLm.size(0) * i1) +
                YLm.size(0) * YLm.size(1) * (static_cast<int32_T>(d) - 1)];
      }
    }
    st.site = &i_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, F, G_tt, F.size(1), G_tt.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, F, G_tt, G_tL);
    st.site = &j_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, F, P_tt, F.size(1), P_tt.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, F, P_tt, y);
    st.site = &j_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, F, y.size(1), F.size(1));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, y, F, W);
    iv[0] = (*(int32_T(*)[2])W.size())[0];
    iv[1] = (*(int32_T(*)[2])W.size())[1];
    result[0] = (*(int32_T(*)[2])SIGMA.size())[0];
    result[1] = (*(int32_T(*)[2])SIGMA.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &result[0], &y_emlrtECI, (emlrtCTX)sp);
    loop_ub = W.size(0) * W.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      W[i1] = W[i1] + SIGMA[i1];
    }
    st.site = &k_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, Sigma, beta, YLm.size(1), beta.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, Sigma, beta, G_tt);
    st.site = &k_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, X, G_tL, X.size(1), G_tL.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, X, G_tL, r);
    if (G_tt.size(0) != r.size(0)) {
      emlrtSizeEqCheck1DR2012b(G_tt.size(0), r.size(0), &x_emlrtECI,
                               (emlrtCTX)sp);
    }
    loop_ub = G_tt.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      G_tt[i1] = G_tt[i1] + r[i1];
    }
    st.site = &l_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, X, W, X.size(1), W.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, X, W, y);
    st.site = &l_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, X, y.size(1), X.size(1));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, y, X, varargin_1);
    iv[0] = (*(int32_T(*)[2])varargin_1.size())[0];
    iv[1] = (*(int32_T(*)[2])varargin_1.size())[1];
    result[0] =
        (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&Omega)->size())[0];
    result[1] =
        (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&Omega)->size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &result[0], &w_emlrtECI, (emlrtCTX)sp);
    b_Phi.set_size(&tc_emlrtRTEI, sp, varargin_1.size(0), varargin_1.size(1));
    loop_ub = varargin_1.size(0) * varargin_1.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_Phi[i1] = varargin_1[i1] + Omega[i1];
    }
    st.site = &m_emlrtRSI;
    coder::inv(&st, b_Phi, Sigma);
    if (y_t.size(0) != G_tt.size(0)) {
      emlrtSizeEqCheck1DR2012b(y_t.size(0), G_tt.size(0), &v_emlrtECI,
                               (emlrtCTX)sp);
    }
    st.site = &n_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, W, X, W.size(1), X.size(1));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, W, X, y);
    st.site = &n_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, Sigma, y.size(1), Sigma.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, Sigma, varargin_1);
    st.site = &n_emlrtRSI;
    loop_ub = y_t.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      y_t[i1] = y_t[i1] - G_tt[i1];
    }
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, varargin_1, y_t, varargin_1.size(1),
                               y_t.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, varargin_1, y_t, G_tt);
    if (G_tL.size(0) != G_tt.size(0)) {
      emlrtSizeEqCheck1DR2012b(G_tL.size(0), G_tt.size(0), &u_emlrtECI,
                               (emlrtCTX)sp);
    }
    G_tt.set_size(&uc_emlrtRTEI, sp, G_tL.size(0));
    loop_ub = G_tL.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      G_tt[i1] = G_tL[i1] + G_tt[i1];
    }
    st.site = &o_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, W, X, W.size(1), X.size(1));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, W, X, y);
    st.site = &o_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, Sigma, y.size(1), Sigma.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, Sigma, varargin_1);
    st.site = &o_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, varargin_1, X, varargin_1.size(1),
                               X.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, varargin_1, X, y);
    st.site = &o_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, W, y.size(1), W.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, W, P_tt);
    iv[0] = (*(int32_T(*)[2])W.size())[0];
    iv[1] = (*(int32_T(*)[2])W.size())[1];
    result[0] = (*(int32_T(*)[2])P_tt.size())[0];
    result[1] = (*(int32_T(*)[2])P_tt.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &result[0], &t_emlrtECI, (emlrtCTX)sp);
    loop_ub = W.size(0) * W.size(1);
    P_tt.set_size(&vc_emlrtRTEI, sp, W.size(0), W.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      P_tt[i1] = W[i1] - P_tt[i1];
    }
    if (b_p != static_cast<int32_T>(muDoubleScalarFloor(b_p))) {
      emlrtIntegerCheckR2012b(b_p, &c_emlrtDCI, (emlrtCTX)sp);
    }
    if ((static_cast<int32_T>(b_p) < 1) ||
        (static_cast<int32_T>(b_p) > G_tLm.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, G_tLm.size(0),
                                    &bb_emlrtBCI, (emlrtCTX)sp);
    }
    result[0] = 1;
    result[1] = G_tLm.size(1);
    b_result[0] = 1;
    b_result[1] = G_tL.size(0);
    emlrtSubAssignSizeCheckR2012b(&result[0], 2, &b_result[0], 2, &n_emlrtECI,
                                  (emlrtCTX)sp);
    loop_ub = G_tL.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      G_tLm[(static_cast<int32_T>(b_p) + G_tLm.size(0) * i1) - 1] = G_tL[i1];
    }
    //  save for use in backward recursion
    if ((static_cast<int32_T>(b_p) < 1) ||
        (static_cast<int32_T>(b_p) > P_tLm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, P_tLm.size(2),
                                    &db_emlrtBCI, (emlrtCTX)sp);
    }
    b_result[0] = P_tLm.size(0);
    b_result[1] = P_tLm.size(1);
    emlrtSubAssignSizeCheckR2012b(&b_result[0], 2, W.size(), 2, &o_emlrtECI,
                                  (emlrtCTX)sp);
    loop_ub = W.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      sizes_idx_1 = W.size(0);
      for (i2 = 0; i2 < sizes_idx_1; i2++) {
        P_tLm[(i2 + P_tLm.size(0) * i1) +
              P_tLm.size(0) * P_tLm.size(1) * (static_cast<int32_T>(b_p) - 1)] =
            W[i2 + W.size(0) * i1];
      }
    }
    if ((static_cast<int32_T>(b_p) < 1) ||
        (static_cast<int32_T>(b_p) > G_ttm.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, G_ttm.size(0),
                                    &lb_emlrtBCI, (emlrtCTX)sp);
    }
    result[0] = 1;
    result[1] = G_ttm.size(1);
    b_result[0] = 1;
    b_result[1] = G_tt.size(0);
    emlrtSubAssignSizeCheckR2012b(&result[0], 2, &b_result[0], 2, &s_emlrtECI,
                                  (emlrtCTX)sp);
    loop_ub = G_tt.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      G_ttm[(static_cast<int32_T>(b_p) + G_ttm.size(0) * i1) - 1] = G_tt[i1];
    }
    if ((static_cast<int32_T>(b_p) < 1) ||
        (static_cast<int32_T>(b_p) > P_ttm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, P_ttm.size(2),
                                    &jb_emlrtBCI, (emlrtCTX)sp);
    }
    b_result[0] = P_ttm.size(0);
    b_result[1] = P_ttm.size(1);
    emlrtSubAssignSizeCheckR2012b(&b_result[0], 2, P_tt.size(), 2, &r_emlrtECI,
                                  (emlrtCTX)sp);
    loop_ub = P_tt.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      sizes_idx_1 = P_tt.size(0);
      for (i2 = 0; i2 < sizes_idx_1; i2++) {
        P_ttm[(i2 + P_ttm.size(0) * i1) +
              P_ttm.size(0) * P_ttm.size(1) * (static_cast<int32_T>(b_p) - 1)] =
            P_tt[i2 + P_tt.size(0) * i1];
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  //
  //  backward recursion
  if ((T < 1.0) || (i3 > G_ttm.size(0))) {
    emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(T), 1, G_ttm.size(0),
                                  &kb_emlrtBCI, (emlrtCTX)sp);
  }
  loop_ub = G_ttm.size(1);
  y_t.set_size(&wc_emlrtRTEI, sp, G_ttm.size(1));
  for (i = 0; i < loop_ub; i++) {
    y_t[i] = G_ttm[(static_cast<int32_T>(T) + G_ttm.size(0) * i) - 1];
  }
  //  M*(P+1) by 1
  if ((T < 1.0) || (i3 > P_ttm.size(2))) {
    emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(T), 1, P_ttm.size(2),
                                  &ib_emlrtBCI, (emlrtCTX)sp);
  }
  //  M*(P+1) by M*(P+1)
  st.site = &p_emlrtRSI;
  loop_ub = P_ttm.size(0);
  sizes_idx_1 = P_ttm.size(1);
  Sigma.set_size(&xc_emlrtRTEI, &st, P_ttm.size(0), P_ttm.size(1));
  for (i = 0; i < sizes_idx_1; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      Sigma[i1 + Sigma.size(0) * i] =
          P_ttm[(i1 + P_ttm.size(0) * i) +
                P_ttm.size(0) * P_ttm.size(1) * (static_cast<int32_T>(T) - 1)];
    }
  }
  b_st.site = &gc_emlrtRSI;
  coder::cholesky(&b_st, Sigma);
  st.site = &p_emlrtRSI;
  b_st.site = &p_emlrtRSI;
  coder::randn(&b_st, MP1, G_tt);
  b_st.site = &hb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, Sigma, G_tt, Sigma.size(0), G_tt.size(0));
  b_st.site = &gb_emlrtRSI;
  coder::internal::blas::b_mtimes(&b_st, Sigma, G_tt, r);
  if (y_t.size(0) != r.size(0)) {
    emlrtSizeEqCheck1DR2012b(y_t.size(0), r.size(0), &q_emlrtECI, (emlrtCTX)sp);
  }
  loop_ub = y_t.size(0);
  for (i = 0; i < loop_ub; i++) {
    y_t[i] = y_t[i] + r[i];
  }
  //  G(t+1)
  if (1 > YLm.size(0)) {
    loop_ub = 0;
  } else {
    if (1 > y_t.size(0)) {
      emlrtDynamicBoundsCheckR2012b(1, 1, y_t.size(0), &hb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (YLm.size(0) > y_t.size(0)) {
      emlrtDynamicBoundsCheckR2012b(YLm.size(0), 1, y_t.size(0), &gb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    loop_ub = YLm.size(0);
  }
  result[0] = 1;
  result[1] = loop_ub;
  st.site = &q_emlrtRSI;
  coder::internal::indexShapeCheck(&st, y_t.size(0), result);
  i = Mu.size(0);
  if ((T < 1.0) || (i3 > i)) {
    emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(T), 1, i, &fb_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  result[0] = 1;
  result[1] = Mu.size(1);
  b_result[0] = 1;
  b_result[1] = loop_ub;
  emlrtSubAssignSizeCheckR2012b(&result[0], 2, &b_result[0], 2, &p_emlrtECI,
                                (emlrtCTX)sp);
  for (i = 0; i < loop_ub; i++) {
    Mu[(static_cast<int32_T>(T) + Mu.size(0) * i) - 1] = y_t[i];
  }
  i = static_cast<int32_T>(((P + 1.0) + (-1.0 - (T - 1.0))) / -1.0);
  emlrtForLoopVectorCheckR2021a(T - 1.0, -1.0, P + 1.0, mxDOUBLE_CLASS, i,
                                &p_emlrtRTEI, (emlrtCTX)sp);
  for (input_sizes_idx_1 = 0; input_sizes_idx_1 < i; input_sizes_idx_1++) {
    b_p = (T - 1.0) + -static_cast<real_T>(input_sizes_idx_1);
    i1 = static_cast<int32_T>(b_p);
    if ((b_p < 1.0) || (i1 > P_ttm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, P_ttm.size(2),
                                    &eb_emlrtBCI, (emlrtCTX)sp);
    }
    if ((b_p + 1.0 < 1.0) || (i1 + 1 > P_tLm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p) + 1, 1,
                                    P_tLm.size(2), &cb_emlrtBCI, (emlrtCTX)sp);
    }
    if ((b_p + 1.0 < 1.0) || (static_cast<int32_T>(b_p) + 1 > G_tLm.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p) + 1, 1,
                                    G_tLm.size(0), &ab_emlrtBCI, (emlrtCTX)sp);
    }
    loop_ub = G_tLm.size(1);
    G_tL.set_size(&yc_emlrtRTEI, sp, G_tLm.size(1));
    for (i2 = 0; i2 < loop_ub; i2++) {
      G_tL[i2] = G_tLm[static_cast<int32_T>(b_p) + G_tLm.size(0) * i2];
    }
    loop_ub = P_tLm.size(0);
    sizes_idx_1 = P_tLm.size(1);
    b_Phi.set_size(&ad_emlrtRTEI, sp, P_tLm.size(0), P_tLm.size(1));
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      for (i3 = 0; i3 < loop_ub; i3++) {
        b_Phi[i3 + b_Phi.size(0) * i2] =
            P_tLm[(i3 + P_tLm.size(0) * i2) +
                  P_tLm.size(0) * P_tLm.size(1) * static_cast<int32_T>(b_p)];
      }
    }
    st.site = &r_emlrtRSI;
    coder::inv(&st, b_Phi, Sigma);
    if (y_t.size(0) != G_tL.size(0)) {
      emlrtSizeEqCheck1DR2012b(y_t.size(0), G_tL.size(0), &m_emlrtECI,
                               (emlrtCTX)sp);
    }
    if ((b_p < 1.0) || (i1 > G_ttm.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, G_ttm.size(0),
                                    &y_emlrtBCI, (emlrtCTX)sp);
    }
    loop_ub = G_ttm.size(1);
    G_tt1.set_size(&bd_emlrtRTEI, sp, G_ttm.size(1));
    for (i2 = 0; i2 < loop_ub; i2++) {
      G_tt1[i2] = G_ttm[(static_cast<int32_T>(b_p) + G_ttm.size(0) * i2) - 1];
    }
    st.site = &s_emlrtRSI;
    loop_ub = P_ttm.size(1);
    sizes_idx_1 = P_ttm.size(0);
    b_Phi.set_size(&cd_emlrtRTEI, &st, P_ttm.size(0), P_ttm.size(1));
    for (i2 = 0; i2 < loop_ub; i2++) {
      for (i3 = 0; i3 < sizes_idx_1; i3++) {
        b_Phi[i3 + b_Phi.size(0) * i2] =
            P_ttm[(i3 + P_ttm.size(0) * i2) +
                  P_ttm.size(0) * P_ttm.size(1) *
                      (static_cast<int32_T>(b_p) - 1)];
      }
    }
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, b_Phi, F, P_ttm.size(1), F.size(1));
    loop_ub = P_ttm.size(0);
    sizes_idx_1 = P_ttm.size(1);
    b_Phi.set_size(&cd_emlrtRTEI, &st, P_ttm.size(0), P_ttm.size(1));
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      for (i3 = 0; i3 < loop_ub; i3++) {
        b_Phi[i3 + b_Phi.size(0) * i2] =
            P_ttm[(i3 + P_ttm.size(0) * i2) +
                  P_ttm.size(0) * P_ttm.size(1) *
                      (static_cast<int32_T>(b_p) - 1)];
      }
    }
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, b_Phi, F, y);
    st.site = &s_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, Sigma, y.size(1), Sigma.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, Sigma, varargin_1);
    st.site = &s_emlrtRSI;
    loop_ub = y_t.size(0);
    for (i2 = 0; i2 < loop_ub; i2++) {
      y_t[i2] = y_t[i2] - G_tL[i2];
    }
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, varargin_1, y_t, varargin_1.size(1),
                               y_t.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, varargin_1, y_t, r);
    if (G_tt1.size(0) != r.size(0)) {
      emlrtSizeEqCheck1DR2012b(G_tt1.size(0), r.size(0), &l_emlrtECI,
                               (emlrtCTX)sp);
    }
    loop_ub = G_tt1.size(0);
    for (i2 = 0; i2 < loop_ub; i2++) {
      G_tt1[i2] = G_tt1[i2] + r[i2];
    }
    st.site = &t_emlrtRSI;
    loop_ub = P_ttm.size(1);
    sizes_idx_1 = P_ttm.size(0);
    b_Phi.set_size(&cd_emlrtRTEI, &st, P_ttm.size(0), P_ttm.size(1));
    for (i2 = 0; i2 < loop_ub; i2++) {
      for (i3 = 0; i3 < sizes_idx_1; i3++) {
        b_Phi[i3 + b_Phi.size(0) * i2] =
            P_ttm[(i3 + P_ttm.size(0) * i2) +
                  P_ttm.size(0) * P_ttm.size(1) *
                      (static_cast<int32_T>(b_p) - 1)];
      }
    }
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, b_Phi, F, P_ttm.size(1), F.size(1));
    loop_ub = P_ttm.size(0);
    sizes_idx_1 = P_ttm.size(1);
    b_Phi.set_size(&cd_emlrtRTEI, &st, P_ttm.size(0), P_ttm.size(1));
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      for (i3 = 0; i3 < loop_ub; i3++) {
        b_Phi[i3 + b_Phi.size(0) * i2] =
            P_ttm[(i3 + P_ttm.size(0) * i2) +
                  P_ttm.size(0) * P_ttm.size(1) *
                      (static_cast<int32_T>(b_p) - 1)];
      }
    }
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, b_Phi, F, y);
    st.site = &t_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, Sigma, y.size(1), Sigma.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, Sigma, varargin_1);
    st.site = &t_emlrtRSI;
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, varargin_1, F, varargin_1.size(1),
                               F.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, varargin_1, F, y);
    st.site = &t_emlrtRSI;
    loop_ub = P_ttm.size(0);
    sizes_idx_1 = P_ttm.size(1);
    b_Phi.set_size(&cd_emlrtRTEI, &st, P_ttm.size(0), P_ttm.size(1));
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      for (i3 = 0; i3 < loop_ub; i3++) {
        b_Phi[i3 + b_Phi.size(0) * i2] =
            P_ttm[(i3 + P_ttm.size(0) * i2) +
                  P_ttm.size(0) * P_ttm.size(1) *
                      (static_cast<int32_T>(b_p) - 1)];
      }
    }
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, b_Phi, y.size(1), P_ttm.size(0));
    loop_ub = P_ttm.size(0);
    sizes_idx_1 = P_ttm.size(1);
    b_Phi.set_size(&cd_emlrtRTEI, &st, P_ttm.size(0), P_ttm.size(1));
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      for (i3 = 0; i3 < loop_ub; i3++) {
        b_Phi[i3 + b_Phi.size(0) * i2] =
            P_ttm[(i3 + P_ttm.size(0) * i2) +
                  P_ttm.size(0) * P_ttm.size(1) *
                      (static_cast<int32_T>(b_p) - 1)];
      }
    }
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, b_Phi, Sigma);
    loop_ub = P_ttm.size(0);
    sizes_idx_1 = P_ttm.size(1);
    result[0] = P_ttm.size(0);
    result[1] = P_ttm.size(1);
    iv[0] = (*(int32_T(*)[2])Sigma.size())[0];
    iv[1] = (*(int32_T(*)[2])Sigma.size())[1];
    emlrtSizeEqCheckNDR2012b(&result[0], &iv[0], &k_emlrtECI, (emlrtCTX)sp);
    Sigma.set_size(&dd_emlrtRTEI, sp, P_ttm.size(0), P_ttm.size(1));
    for (i2 = 0; i2 < sizes_idx_1; i2++) {
      for (i3 = 0; i3 < loop_ub; i3++) {
        Sigma[i3 + Sigma.size(0) * i2] =
            P_ttm[(i3 + P_ttm.size(0) * i2) +
                  P_ttm.size(0) * P_ttm.size(1) *
                      (static_cast<int32_T>(b_p) - 1)] -
            Sigma[i3 + Sigma.size(0) * i2];
      }
    }
    b_result[0] = Sigma.size(1);
    b_result[1] = Sigma.size(0);
    iv[0] = (*(int32_T(*)[2])Sigma.size())[0];
    iv[1] = (*(int32_T(*)[2])Sigma.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_result[0], &j_emlrtECI, (emlrtCTX)sp);
    b_Phi.set_size(&ed_emlrtRTEI, sp, Sigma.size(0), Sigma.size(1));
    loop_ub = Sigma.size(1);
    for (i2 = 0; i2 < loop_ub; i2++) {
      sizes_idx_1 = Sigma.size(0);
      for (i3 = 0; i3 < sizes_idx_1; i3++) {
        b_Phi[i3 + b_Phi.size(0) * i2] =
            (Sigma[i3 + Sigma.size(0) * i2] + Sigma[i2 + Sigma.size(0) * i3]) /
            2.0;
      }
    }
    Sigma.set_size(&fd_emlrtRTEI, sp, b_Phi.size(0), b_Phi.size(1));
    loop_ub = b_Phi.size(0) * b_Phi.size(1);
    for (i2 = 0; i2 < loop_ub; i2++) {
      Sigma[i2] = b_Phi[i2];
    }
    st.site = &u_emlrtRSI;
    cholmod(&st, Sigma, varargin_1);
    st.site = &u_emlrtRSI;
    b_st.site = &u_emlrtRSI;
    coder::randn(&b_st, MP1, G_tt);
    b_st.site = &hb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, varargin_1, G_tt, varargin_1.size(0),
                               G_tt.size(0));
    b_st.site = &gb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, varargin_1, G_tt, y_t);
    if (G_tt1.size(0) != y_t.size(0)) {
      emlrtSizeEqCheck1DR2012b(G_tt1.size(0), y_t.size(0), &i_emlrtECI,
                               (emlrtCTX)sp);
    }
    y_t.set_size(&gd_emlrtRTEI, sp, G_tt1.size(0));
    loop_ub = G_tt1.size(0);
    for (i2 = 0; i2 < loop_ub; i2++) {
      y_t[i2] = G_tt1[i2] + y_t[i2];
    }
    if (1 > M) {
      loop_ub = 0;
    } else {
      if (1 > y_t.size(0)) {
        emlrtDynamicBoundsCheckR2012b(1, 1, y_t.size(0), &x_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      if (M > y_t.size(0)) {
        emlrtDynamicBoundsCheckR2012b(M, 1, y_t.size(0), &w_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      loop_ub = M;
    }
    result[0] = 1;
    result[1] = loop_ub;
    st.site = &v_emlrtRSI;
    coder::internal::indexShapeCheck(&st, y_t.size(0), result);
    i2 = Mu.size(0);
    if ((b_p < 1.0) || (i1 > i2)) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(b_p), 1, i2,
                                    &v_emlrtBCI, (emlrtCTX)sp);
    }
    result[0] = 1;
    result[1] = Mu.size(1);
    b_result[0] = 1;
    b_result[1] = loop_ub;
    emlrtSubAssignSizeCheckR2012b(&result[0], 2, &b_result[0], 2, &h_emlrtECI,
                                  (emlrtCTX)sp);
    for (i1 = 0; i1 < loop_ub; i1++) {
      Mu[(static_cast<int32_T>(b_p) + Mu.size(0) * i1) - 1] = y_t[i1];
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (Gen_Mu.cpp)
