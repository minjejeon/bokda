//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Um.cpp
//
// Code generation for function 'Gen_Um'
//

// Include files
#include "Gen_Um.h"
#include "Gen_Um_data.h"
#include "cholmod.h"
#include "eml_mtimes_helper.h"
#include "eye.h"
#include "invpd.h"
#include "mtimes.h"
#include "randn.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Variable Definitions
static emlrtRSInfo
    emlrtRSI{
        8,        // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    b_emlrtRSI{
        12,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    c_emlrtRSI{
        16,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    d_emlrtRSI{
        23,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    e_emlrtRSI{
        26,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    f_emlrtRSI{
        27,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    g_emlrtRSI{
        28,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    h_emlrtRSI{
        30,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    i_emlrtRSI{
        32,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    j_emlrtRSI{
        33,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    k_emlrtRSI{
        34,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    l_emlrtRSI{
        35,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    m_emlrtRSI{
        36,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    n_emlrtRSI{
        51,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    o_emlrtRSI{
        54,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    p_emlrtRSI{
        62,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    q_emlrtRSI{
        64,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    r_emlrtRSI{
        66,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    s_emlrtRSI{
        68,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    t_emlrtRSI{
        69,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    u_emlrtRSI{
        71,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    v_emlrtRSI{
        75,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo
    w_emlrtRSI{
        77,       // lineNo
        "Gen_Um", // fcnName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pathName
    };

static emlrtRSInfo cb_emlrtRSI{
    24,    // lineNo
    "cat", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtRSInfo db_emlrtRSI{
    96,         // lineNo
    "cat_impl", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pathName
};

static emlrtBCInfo
    hb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        9,        // lineNo
        3,        // colNo
        "H",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    i_emlrtECI{
        -1,       // nDims
        9,        // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    ib_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        22,       // lineNo
        11,       // colNo
        "Um",     // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    jb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        17,       // lineNo
        3,        // colNo
        "F",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    kb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        17,       // lineNo
        7,        // colNo
        "F",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    lb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        17,       // lineNo
        12,       // colNo
        "F",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    mb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        17,       // lineNo
        16,       // colNo
        "F",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    j_emlrtECI{
        -1,       // nDims
        17,       // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    nb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        13,       // lineNo
        3,        // colNo
        "Q",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    ob_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        13,       // lineNo
        5,        // colNo
        "Q",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    pb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        13,       // lineNo
        8,        // colNo
        "Q",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    qb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        13,       // lineNo
        10,       // colNo
        "Q",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    k_emlrtECI{
        -1,       // nDims
        13,       // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    rb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        14,       // lineNo
        3,        // colNo
        "Q",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    sb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        14,       // lineNo
        7,        // colNo
        "Q",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    tb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        14,       // lineNo
        12,       // colNo
        "Q",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    ub_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        14,       // lineNo
        16,       // colNo
        "Q",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    l_emlrtECI{
        -1,       // nDims
        14,       // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    m_emlrtECI{
        2,        // nDims
        27,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    n_emlrtECI{
        2,        // nDims
        29,       // lineNo
        18,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    vb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        32,       // lineNo
        14,       // colNo
        "Ym",     // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    o_emlrtECI{
        -1,       // nDims
        32,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    p_emlrtECI{
        -1,       // nDims
        34,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    q_emlrtECI{
        2,        // nDims
        35,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    wb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        39,       // lineNo
        14,       // colNo
        "P_ttm",  // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    r_emlrtECI{
        -1,       // nDims
        39,       // lineNo
        4,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    xb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        49,       // lineNo
        18,       // colNo
        "P_ttm",  // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    s_emlrtECI{
        2,        // nDims
        50,       // lineNo
        9,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    yb_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        38,       // lineNo
        13,       // colNo
        "f_ttm",  // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    t_emlrtECI{
        -1,       // nDims
        38,       // lineNo
        4,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    ac_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        53,       // lineNo
        17,       // colNo
        "f_ttm",  // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    u_emlrtECI{
        -1,       // nDims
        54,       // lineNo
        6,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    l_emlrtRTEI{
        57,       // lineNo
        9,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    bc_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        59,       // lineNo
        19,       // colNo
        "f_ttm",  // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtBCInfo
    cc_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        60,       // lineNo
        20,       // colNo
        "P_ttm",  // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    v_emlrtECI{
        2,        // nDims
        62,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    w_emlrtECI{
        2,        // nDims
        63,       // lineNo
        12,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    dc_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        55,       // lineNo
        4,        // colNo
        "Fm",     // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    x_emlrtECI{
        -1,       // nDims
        55,       // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    ec_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        66,       // lineNo
        13,       // colNo
        "Fm",     // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    y_emlrtECI{
        -1,       // nDims
        66,       // lineNo
        10,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    ab_emlrtECI{
        -1,       // nDims
        69,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    bb_emlrtECI{
        2,        // nDims
        72,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    cb_emlrtECI{
        2,        // nDims
        74,       // lineNo
        12,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtECInfo
    db_emlrtECI{
        -1,       // nDims
        77,       // lineNo
        8,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtBCInfo
    fc_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        78,       // lineNo
        6,        // colNo
        "Fm",     // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtECInfo
    eb_emlrtECI{
        -1,       // nDims
        78,       // lineNo
        3,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo m_emlrtRTEI{
    271,                   // lineNo
    27,                    // colNo
    "check_non_axis_size", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\cat.m" // pName
};

static emlrtBCInfo
    gc_emlrtBCI{
        -1,       // iFirst
        -1,       // iLast
        9,        // lineNo
        6,        // colNo
        "H",      // aName
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m", // pName
        0       // checkKind
    };

static emlrtRTEInfo
    wb_emlrtRTEI{
        8,        // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    xb_emlrtRTEI{
        9,        // lineNo
        13,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    yb_emlrtRTEI{
        20,       // lineNo
        9,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    ac_emlrtRTEI{
        21,       // lineNo
        9,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    bc_emlrtRTEI{
        22,       // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    cc_emlrtRTEI{
        77,                  // lineNo
        13,                  // colNo
        "eml_mtimes_helper", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pName
    };

static emlrtRTEInfo
    dc_emlrtRTEI{
        32,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    ec_emlrtRTEI{
        34,       // lineNo
        4,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    fc_emlrtRTEI{
        47,       // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    gc_emlrtRTEI{
        50,       // lineNo
        8,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    hc_emlrtRTEI{
        54,       // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    ic_emlrtRTEI{
        60,       // lineNo
        10,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    jc_emlrtRTEI{
        63,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    kc_emlrtRTEI{
        63,       // lineNo
        3,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    lc_emlrtRTEI{
        66,       // lineNo
        10,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    mc_emlrtRTEI{
        59,       // lineNo
        10,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    nc_emlrtRTEI{
        69,       // lineNo
        3,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    oc_emlrtRTEI{
        72,       // lineNo
        3,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    pc_emlrtRTEI{
        74,       // lineNo
        11,       // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    qc_emlrtRTEI{
        74,       // lineNo
        3,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

static emlrtRTEInfo
    rc_emlrtRTEI{
        83,       // lineNo
        1,        // colNo
        "Gen_Um", // fName
        "D:\\Dropbox\\Policy_Works\\KOR_US_NIR\\Codes2\\Neutral_Rate_Lib\\Gen_"
        "Um.m" // pName
    };

// Function Definitions
void Gen_Um(const emlrtStack *sp, const coder::array<real_T, 2U> &Ym,
            coder::array<real_T, 2U> &Um, const coder::array<real_T, 1U> &rho,
            const coder::array<real_T, 2U> &Phi,
            const coder::array<real_T, 2U> &Sigma,
            const coder::array<real_T, 2U> &Omega,
            coder::array<real_T, 2U> &Uttm)
{
  coder::array<real_T, 3U> P_ttm;
  coder::array<real_T, 2U> F;
  coder::array<real_T, 2U> H;
  coder::array<real_T, 2U> Kalgain;
  coder::array<real_T, 2U> P_tl;
  coder::array<real_T, 2U> P_tt;
  coder::array<real_T, 2U> Q;
  coder::array<real_T, 2U> b_P_ttm;
  coder::array<real_T, 2U> b_rho;
  coder::array<real_T, 2U> f_ttm;
  coder::array<real_T, 2U> var_tlinv;
  coder::array<real_T, 2U> y;
  coder::array<real_T, 1U> e_tl;
  coder::array<real_T, 1U> f_tl;
  coder::array<real_T, 1U> f_tt;
  coder::array<real_T, 1U> r;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  int32_T b_result[2];
  int32_T input_sizes[2];
  int32_T iv[2];
  int32_T b_loop_ub;
  int32_T c_loop_ub;
  int32_T d_loop_ub;
  int32_T e_loop_ub;
  int32_T f_loop_ub;
  int32_T g_loop_ub;
  int32_T h_loop_ub;
  int32_T i;
  int32_T i1;
  int32_T i10;
  int32_T i11;
  int32_T i12;
  int32_T i13;
  int32_T i14;
  int32_T i15;
  int32_T i16;
  int32_T i17;
  int32_T i18;
  int32_T i19;
  int32_T i2;
  int32_T i3;
  int32_T i4;
  int32_T i5;
  int32_T i6;
  int32_T i7;
  int32_T i8;
  int32_T i9;
  int32_T i_loop_ub;
  int32_T j_loop_ub;
  int32_T k_loop_ub;
  int32_T l_loop_ub;
  int32_T loop_ub;
  int32_T m_loop_ub;
  int32_T n_loop_ub;
  int32_T o_loop_ub;
  int32_T p_loop_ub;
  int32_T q_loop_ub;
  int32_T r_loop_ub;
  int32_T result;
  int32_T result_idx_1;
  int32_T sizes_idx_1;
  int32_T t;
  uint32_T u;
  boolean_T empty_non_axis_sizes;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //  Factor 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  st.site = &emlrtRSI;
  b_st.site = &emlrtRSI;
  coder::eye(&b_st, static_cast<real_T>(Ym.size(1)), P_tt);
  b_st.site = &emlrtRSI;
  coder::eye(&b_st, static_cast<real_T>(Ym.size(1)), var_tlinv);
  b_st.site = &cb_emlrtRSI;
  if ((P_tt.size(0) != 0) && (P_tt.size(1) != 0)) {
    result = P_tt.size(0);
  } else if ((var_tlinv.size(0) != 0) && (var_tlinv.size(1) != 0)) {
    result = var_tlinv.size(0);
  } else {
    result = P_tt.size(0);
    if (var_tlinv.size(0) > P_tt.size(0)) {
      result = var_tlinv.size(0);
    }
  }
  c_st.site = &db_emlrtRSI;
  if ((P_tt.size(0) != result) &&
      ((P_tt.size(0) != 0) && (P_tt.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &m_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  if ((var_tlinv.size(0) != result) &&
      ((var_tlinv.size(0) != 0) && (var_tlinv.size(1) != 0))) {
    emlrtErrorWithMessageIdR2018a(&c_st, &m_emlrtRTEI,
                                  "MATLAB:catenate:matrixDimensionMismatch",
                                  "MATLAB:catenate:matrixDimensionMismatch", 0);
  }
  empty_non_axis_sizes = (result == 0);
  if (empty_non_axis_sizes || ((P_tt.size(0) != 0) && (P_tt.size(1) != 0))) {
    input_sizes[1] = P_tt.size(1);
  } else {
    input_sizes[1] = 0;
  }
  if (empty_non_axis_sizes ||
      ((var_tlinv.size(0) != 0) && (var_tlinv.size(1) != 0))) {
    sizes_idx_1 = var_tlinv.size(1);
  } else {
    sizes_idx_1 = 0;
  }
  result_idx_1 = input_sizes[1];
  H.set_size(&wb_emlrtRTEI, &b_st, result, input_sizes[1] + sizes_idx_1);
  loop_ub = input_sizes[1];
  for (i = 0; i < loop_ub; i++) {
    for (i1 = 0; i1 < result; i1++) {
      H[i1 + H.size(0) * i] = P_tt[i1 + result * i];
    }
  }
  for (i = 0; i < sizes_idx_1; i++) {
    for (i1 = 0; i1 < result; i1++) {
      H[i1 + H.size(0) * (i + result_idx_1)] = var_tlinv[i1 + result * i];
    }
  }
  if (4 > H.size(0)) {
    emlrtDynamicBoundsCheckR2012b(4, 1, H.size(0), &hb_emlrtBCI, (emlrtCTX)sp);
  }
  if (1 > H.size(1)) {
    emlrtDynamicBoundsCheckR2012b(1, 1, H.size(1), &gc_emlrtBCI, (emlrtCTX)sp);
  }
  if (2 > H.size(1)) {
    emlrtDynamicBoundsCheckR2012b(2, 1, H.size(1), &gc_emlrtBCI, (emlrtCTX)sp);
  }
  input_sizes[0] = 1;
  input_sizes[1] = 2;
  b_result[0] = 1;
  b_result[1] = rho.size(0);
  emlrtSubAssignSizeCheckR2012b(&input_sizes[0], 2, &b_result[0], 2,
                                &i_emlrtECI, (emlrtCTX)sp);
  b_rho.set_size(&xb_emlrtRTEI, sp, 1, rho.size(0));
  loop_ub = rho.size(0);
  for (i = 0; i < loop_ub; i++) {
    b_rho[i] = rho[i];
  }
  H[3] = b_rho[0];
  H[H.size(0) + 3] = b_rho[1];
  //  number of columns
  sizes_idx_1 = H.size(1);
  st.site = &b_emlrtRSI;
  coder::eye(&st, 2.0 * static_cast<real_T>(Ym.size(1)), Q);
  if (1 > Ym.size(1)) {
    i = 0;
    i1 = 0;
  } else {
    if (1 > Q.size(0)) {
      emlrtDynamicBoundsCheckR2012b(1, 1, Q.size(0), &nb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (Ym.size(1) > Q.size(0)) {
      emlrtDynamicBoundsCheckR2012b(Ym.size(1), 1, Q.size(0), &ob_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    i = Ym.size(1);
    if (1 > Q.size(1)) {
      emlrtDynamicBoundsCheckR2012b(1, 1, Q.size(1), &pb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (Ym.size(1) > Q.size(1)) {
      emlrtDynamicBoundsCheckR2012b(Ym.size(1), 1, Q.size(1), &qb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    i1 = Ym.size(1);
  }
  input_sizes[0] = i;
  input_sizes[1] = i1;
  emlrtSubAssignSizeCheckR2012b(&input_sizes[0], 2,
                                ((coder::array<real_T, 2U> *)&Omega)->size(), 2,
                                &k_emlrtECI, (emlrtCTX)sp);
  loop_ub = Omega.size(1);
  for (i = 0; i < loop_ub; i++) {
    result_idx_1 = Omega.size(0);
    for (i1 = 0; i1 < result_idx_1; i1++) {
      Q[i1 + Q.size(0) * i] = Omega[i1 + Omega.size(0) * i];
    }
  }
  u = static_cast<uint32_T>(Ym.size(1)) << 1;
  if (Ym.size(1) + 1U > u) {
    i = 0;
    i1 = 0;
  } else {
    if ((static_cast<int32_T>(Ym.size(1) + 1U) < 1) ||
        (static_cast<int32_T>(Ym.size(1) + 1U) > Q.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(Ym.size(1) + 1U), 1,
                                    Q.size(0), &rb_emlrtBCI, (emlrtCTX)sp);
    }
    i = Ym.size(1);
    if ((static_cast<int32_T>(u) < 1) ||
        (static_cast<int32_T>(u) > Q.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(u), 1, Q.size(0),
                                    &sb_emlrtBCI, (emlrtCTX)sp);
    }
    i1 = static_cast<int32_T>(u);
  }
  u = static_cast<uint32_T>(Ym.size(1)) << 1;
  if (Ym.size(1) + 1U > u) {
    i2 = 0;
    result = 0;
  } else {
    if ((static_cast<int32_T>(Ym.size(1) + 1U) < 1) ||
        (static_cast<int32_T>(Ym.size(1) + 1U) > Q.size(1))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(Ym.size(1) + 1U), 1,
                                    Q.size(1), &tb_emlrtBCI, (emlrtCTX)sp);
    }
    i2 = Ym.size(1);
    if ((static_cast<int32_T>(u) < 1) ||
        (static_cast<int32_T>(u) > Q.size(1))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(u), 1, Q.size(1),
                                    &ub_emlrtBCI, (emlrtCTX)sp);
    }
    result = static_cast<int32_T>(u);
  }
  input_sizes[0] = i1 - i;
  input_sizes[1] = result - i2;
  emlrtSubAssignSizeCheckR2012b(&input_sizes[0], 2,
                                ((coder::array<real_T, 2U> *)&Sigma)->size(), 2,
                                &l_emlrtECI, (emlrtCTX)sp);
  loop_ub = Sigma.size(1);
  for (i1 = 0; i1 < loop_ub; i1++) {
    result_idx_1 = Sigma.size(0);
    for (result = 0; result < result_idx_1; result++) {
      Q[(i + result) + Q.size(0) * (i2 + i1)] =
          Sigma[result + Sigma.size(0) * i1];
    }
  }
  st.site = &c_emlrtRSI;
  coder::eye(&st, 2.0 * static_cast<real_T>(Ym.size(1)), F);
  u = static_cast<uint32_T>(Ym.size(1)) << 1;
  if (Ym.size(1) + 1U > u) {
    i = 0;
    i1 = 0;
  } else {
    if ((static_cast<int32_T>(Ym.size(1) + 1U) < 1) ||
        (static_cast<int32_T>(Ym.size(1) + 1U) > F.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(Ym.size(1) + 1U), 1,
                                    F.size(0), &jb_emlrtBCI, (emlrtCTX)sp);
    }
    i = Ym.size(1);
    if ((static_cast<int32_T>(u) < 1) ||
        (static_cast<int32_T>(u) > F.size(0))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(u), 1, F.size(0),
                                    &kb_emlrtBCI, (emlrtCTX)sp);
    }
    i1 = static_cast<int32_T>(u);
  }
  u = static_cast<uint32_T>(Ym.size(1)) << 1;
  if (Ym.size(1) + 1U > u) {
    i2 = 0;
    result = 0;
  } else {
    if ((static_cast<int32_T>(Ym.size(1) + 1U) < 1) ||
        (static_cast<int32_T>(Ym.size(1) + 1U) > F.size(1))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(Ym.size(1) + 1U), 1,
                                    F.size(1), &lb_emlrtBCI, (emlrtCTX)sp);
    }
    i2 = Ym.size(1);
    if ((static_cast<int32_T>(u) < 1) ||
        (static_cast<int32_T>(u) > F.size(1))) {
      emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(u), 1, F.size(1),
                                    &mb_emlrtBCI, (emlrtCTX)sp);
    }
    result = static_cast<int32_T>(u);
  }
  input_sizes[0] = i1 - i;
  input_sizes[1] = result - i2;
  emlrtSubAssignSizeCheckR2012b(&input_sizes[0], 2,
                                ((coder::array<real_T, 2U> *)&Phi)->size(), 2,
                                &j_emlrtECI, (emlrtCTX)sp);
  loop_ub = Phi.size(1);
  for (i1 = 0; i1 < loop_ub; i1++) {
    result_idx_1 = Phi.size(0);
    for (result = 0; result < result_idx_1; result++) {
      F[(i + result) + F.size(0) * (i2 + i1)] = Phi[result + Phi.size(0) * i1];
    }
  }
  // %%%% Kalman filtering step
  f_ttm.set_size(&yb_emlrtRTEI, sp, H.size(1), Ym.size(0));
  P_ttm.set_size(&ac_emlrtRTEI, sp, H.size(1), H.size(1), Ym.size(0));
  i = Um.size(0);
  if (1 > i) {
    emlrtDynamicBoundsCheckR2012b(1, 1, i, &ib_emlrtBCI, (emlrtCTX)sp);
  }
  loop_ub = Um.size(1);
  f_tt.set_size(&bc_emlrtRTEI, sp, loop_ub);
  for (i = 0; i < loop_ub; i++) {
    f_tt[i] = Um[Um.size(0) * i];
  }
  st.site = &d_emlrtRSI;
  coder::eye(&st, static_cast<real_T>(H.size(1)), P_tt);
  loop_ub = P_tt.size(0) * P_tt.size(1);
  for (i = 0; i < loop_ub; i++) {
    P_tt[i] = 0.1 * P_tt[i];
  }
  //  비조건부 분산-공분산 행렬
  i = Ym.size(0);
  for (t = 0; t < i; t++) {
    st.site = &e_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, F, f_tt, F.size(1), f_tt.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, F, f_tt, f_tl);
    st.site = &f_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, F, P_tt, F.size(1), P_tt.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, F, P_tt, y);
    st.site = &f_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, F, y.size(1), F.size(1));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, y, F, P_tl);
    iv[0] = (*(int32_T(*)[2])P_tl.size())[0];
    iv[1] = (*(int32_T(*)[2])P_tl.size())[1];
    input_sizes[0] = (*(int32_T(*)[2])Q.size())[0];
    input_sizes[1] = (*(int32_T(*)[2])Q.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &input_sizes[0], &m_emlrtECI,
                             (emlrtCTX)sp);
    loop_ub = P_tl.size(0) * P_tl.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      P_tl[i1] = P_tl[i1] + Q[i1];
    }
    st.site = &g_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, H, P_tl, H.size(1), P_tl.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, H, P_tl, y);
    st.site = &g_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, H, y.size(1), H.size(1));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, y, H, P_tt);
    input_sizes[0] = P_tt.size(1);
    input_sizes[1] = P_tt.size(0);
    iv[0] = (*(int32_T(*)[2])P_tt.size())[0];
    iv[1] = (*(int32_T(*)[2])P_tt.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &input_sizes[0], &n_emlrtECI,
                             (emlrtCTX)sp);
    b_P_ttm.set_size(&cc_emlrtRTEI, sp, P_tt.size(0), P_tt.size(1));
    loop_ub = P_tt.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      result_idx_1 = P_tt.size(0);
      for (i2 = 0; i2 < result_idx_1; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            0.5 * (P_tt[i2 + P_tt.size(0) * i1] + P_tt[i1 + P_tt.size(0) * i2]);
      }
    }
    st.site = &h_emlrtRSI;
    invpd(&st, b_P_ttm, var_tlinv);
    if (t + 1 > Ym.size(0)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, Ym.size(0), &vb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    loop_ub = Ym.size(1);
    e_tl.set_size(&dc_emlrtRTEI, sp, Ym.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      e_tl[i1] = Ym[t + Ym.size(0) * i1];
    }
    st.site = &i_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, H, f_tl, H.size(1), f_tl.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, H, f_tl, r);
    if (e_tl.size(0) != r.size(0)) {
      emlrtSizeEqCheck1DR2012b(e_tl.size(0), r.size(0), &o_emlrtECI,
                               (emlrtCTX)sp);
    }
    loop_ub = e_tl.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      e_tl[i1] = e_tl[i1] - r[i1];
    }
    st.site = &j_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, P_tl, H, P_tl.size(1), H.size(1));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, P_tl, H, y);
    st.site = &j_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, var_tlinv, y.size(1),
                               var_tlinv.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, var_tlinv, Kalgain);
    st.site = &k_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, Kalgain, e_tl, Kalgain.size(1),
                               e_tl.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, Kalgain, e_tl, f_tt);
    if (f_tl.size(0) != f_tt.size(0)) {
      emlrtSizeEqCheck1DR2012b(f_tl.size(0), f_tt.size(0), &p_emlrtECI,
                               (emlrtCTX)sp);
    }
    f_tt.set_size(&ec_emlrtRTEI, sp, f_tl.size(0));
    loop_ub = f_tl.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      f_tt[i1] = f_tl[i1] + f_tt[i1];
    }
    st.site = &l_emlrtRSI;
    coder::eye(&st, static_cast<real_T>(sizes_idx_1), var_tlinv);
    st.site = &l_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, Kalgain, H, Kalgain.size(1), H.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, Kalgain, H, P_tt);
    iv[0] = (*(int32_T(*)[2])var_tlinv.size())[0];
    iv[1] = (*(int32_T(*)[2])var_tlinv.size())[1];
    input_sizes[0] = (*(int32_T(*)[2])P_tt.size())[0];
    input_sizes[1] = (*(int32_T(*)[2])P_tt.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &input_sizes[0], &q_emlrtECI,
                             (emlrtCTX)sp);
    loop_ub = var_tlinv.size(0) * var_tlinv.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      var_tlinv[i1] = var_tlinv[i1] - P_tt[i1];
    }
    st.site = &m_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, var_tlinv, P_tl, var_tlinv.size(1),
                               P_tl.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, var_tlinv, P_tl, P_tt);
    if (t + 1 > f_ttm.size(1)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, f_ttm.size(1), &yb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    emlrtSubAssignSizeCheckR2012b(f_ttm.size(), 1, f_tt.size(), 1, &t_emlrtECI,
                                  (emlrtCTX)sp);
    loop_ub = f_tt.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      f_ttm[i1 + f_ttm.size(0) * t] = f_tt[i1];
    }
    if (t + 1 > P_ttm.size(2)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, P_ttm.size(2), &wb_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    input_sizes[0] = P_ttm.size(0);
    input_sizes[1] = P_ttm.size(1);
    emlrtSubAssignSizeCheckR2012b(&input_sizes[0], 2, P_tt.size(), 2,
                                  &r_emlrtECI, (emlrtCTX)sp);
    loop_ub = P_tt.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      result_idx_1 = P_tt.size(0);
      for (i2 = 0; i2 < result_idx_1; i2++) {
        P_ttm[(i2 + P_ttm.size(0) * i1) + P_ttm.size(0) * P_ttm.size(1) * t] =
            P_tt[i2 + P_tt.size(0) * i1];
      }
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  // %% Backward recursion
  Um.set_size(&fc_emlrtRTEI, sp, Ym.size(0), H.size(1));
  loop_ub = H.size(1);
  for (i = 0; i < loop_ub; i++) {
    result_idx_1 = Ym.size(0);
    for (i1 = 0; i1 < result_idx_1; i1++) {
      Um[i1 + Um.size(0) * i] = 0.0;
    }
  }
  //  T by k
  if ((Ym.size(0) < 1) || (Ym.size(0) > P_ttm.size(2))) {
    emlrtDynamicBoundsCheckR2012b(Ym.size(0), 1, P_ttm.size(2), &xb_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  //  k by k
  input_sizes[0] = P_ttm.size(0);
  input_sizes[1] = P_ttm.size(1);
  b_result[0] = P_ttm.size(1);
  b_result[1] = P_ttm.size(0);
  if (P_ttm.size(0) != P_ttm.size(1)) {
    emlrtSizeEqCheckNDR2012b(&input_sizes[0], &b_result[0], &s_emlrtECI,
                             (emlrtCTX)sp);
  }
  loop_ub = P_ttm.size(0);
  result_idx_1 = P_ttm.size(1);
  b_P_ttm.set_size(&gc_emlrtRTEI, sp, P_ttm.size(0), P_ttm.size(1));
  for (i = 0; i < result_idx_1; i++) {
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_P_ttm[i1 + b_P_ttm.size(0) * i] =
          (P_ttm[(i1 + P_ttm.size(0) * i) +
                 P_ttm.size(0) * P_ttm.size(1) * (Ym.size(0) - 1)] +
           P_ttm[(i + P_ttm.size(0) * i1) +
                 P_ttm.size(0) * P_ttm.size(1) * (Ym.size(0) - 1)]) /
          2.0;
    }
  }
  st.site = &n_emlrtRSI;
  cholmod(&st, b_P_ttm, P_tt);
  //  k by k
  if ((Ym.size(0) < 1) || (Ym.size(0) > f_ttm.size(1))) {
    emlrtDynamicBoundsCheckR2012b(Ym.size(0), 1, f_ttm.size(1), &ac_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  //  k by 1
  st.site = &o_emlrtRSI;
  b_st.site = &o_emlrtRSI;
  coder::randn(&b_st, static_cast<real_T>(H.size(1)), f_tt);
  b_st.site = &fb_emlrtRSI;
  coder::dynamic_size_checks(&b_st, P_tt, f_tt, P_tt.size(0), f_tt.size(0));
  b_st.site = &eb_emlrtRSI;
  coder::internal::blas::b_mtimes(&b_st, P_tt, f_tt, f_tl);
  if (f_ttm.size(0) != f_tl.size(0)) {
    emlrtSizeEqCheck1DR2012b(f_ttm.size(0), f_tl.size(0), &u_emlrtECI,
                             (emlrtCTX)sp);
  }
  loop_ub = f_ttm.size(0);
  f_tl.set_size(&hc_emlrtRTEI, sp, f_ttm.size(0));
  for (i = 0; i < loop_ub; i++) {
    f_tl[i] = f_ttm[i + f_ttm.size(0) * (Ym.size(0) - 1)] + f_tl[i];
  }
  //  k by 1
  if (Ym.size(0) < 1) {
    emlrtDynamicBoundsCheckR2012b(Ym.size(0), 1, Ym.size(0), &dc_emlrtBCI,
                                  (emlrtCTX)sp);
  }
  input_sizes[0] = 1;
  input_sizes[1] = H.size(1);
  b_result[0] = 1;
  b_result[1] = f_tl.size(0);
  emlrtSubAssignSizeCheckR2012b(&input_sizes[0], 2, &b_result[0], 2,
                                &x_emlrtECI, (emlrtCTX)sp);
  loop_ub = f_tl.size(0);
  for (i = 0; i < loop_ub; i++) {
    Um[(Ym.size(0) + Um.size(0) * i) - 1] = f_tl[i];
  }
  //  1 by by k
  i = static_cast<int32_T>(
      ((-1.0 - (static_cast<real_T>(Ym.size(0)) - 1.0)) + 1.0) / -1.0);
  emlrtForLoopVectorCheckR2021a(static_cast<real_T>(Ym.size(0)) - 1.0, -1.0,
                                1.0, mxDOUBLE_CLASS, i, &l_emlrtRTEI,
                                (emlrtCTX)sp);
  if (0 <= i - 1) {
    i3 = P_ttm.size(0);
    b_loop_ub = P_ttm.size(0);
    i4 = P_ttm.size(1);
    c_loop_ub = P_ttm.size(1);
    d_loop_ub = P_ttm.size(0);
    i5 = P_ttm.size(1);
    e_loop_ub = P_ttm.size(1);
    i6 = f_ttm.size(0);
    i7 = f_ttm.size(0);
    f_loop_ub = f_ttm.size(0);
    i8 = f_ttm.size(0);
    g_loop_ub = f_ttm.size(0);
    i9 = P_ttm.size(1);
    h_loop_ub = P_ttm.size(0);
    i10 = P_ttm.size(1);
    i_loop_ub = P_ttm.size(1);
    j_loop_ub = P_ttm.size(0);
    i11 = P_ttm.size(1);
    k_loop_ub = P_ttm.size(1);
    i12 = f_ttm.size(0);
    i13 = f_ttm.size(0);
    l_loop_ub = f_ttm.size(0);
    i14 = P_ttm.size(0);
    m_loop_ub = P_ttm.size(0);
    i15 = P_ttm.size(1);
    n_loop_ub = P_ttm.size(1);
    o_loop_ub = P_ttm.size(0);
    i16 = P_ttm.size(1);
    p_loop_ub = P_ttm.size(1);
    i17 = P_ttm.size(0);
    i18 = P_ttm.size(1);
    q_loop_ub = P_ttm.size(0);
    i19 = P_ttm.size(1);
    r_loop_ub = P_ttm.size(1);
    b_result[0] = 1;
  }
  for (t = 0; t < i; t++) {
    result = (Ym.size(0) - t) - 2;
    if ((result + 1 < 1) || (result + 1 > f_ttm.size(1))) {
      emlrtDynamicBoundsCheckR2012b(result + 1, 1, f_ttm.size(1), &bc_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    //  km3 by 1
    if ((result + 1 < 1) || (result + 1 > P_ttm.size(2))) {
      emlrtDynamicBoundsCheckR2012b(result + 1, 1, P_ttm.size(2), &cc_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    //  km3 by km3
    st.site = &p_emlrtRSI;
    b_P_ttm.set_size(&ic_emlrtRTEI, &st, b_loop_ub, i4);
    for (i1 = 0; i1 < c_loop_ub; i1++) {
      for (i2 = 0; i2 < b_loop_ub; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            P_ttm[(i2 + P_ttm.size(0) * i1) +
                  P_ttm.size(0) * P_ttm.size(1) * result];
      }
    }
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, F, b_P_ttm, F.size(1), i3);
    b_P_ttm.set_size(&ic_emlrtRTEI, &st, d_loop_ub, i5);
    for (i1 = 0; i1 < e_loop_ub; i1++) {
      for (i2 = 0; i2 < d_loop_ub; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            P_ttm[(i2 + P_ttm.size(0) * i1) +
                  P_ttm.size(0) * P_ttm.size(1) * result];
      }
    }
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, F, b_P_ttm, y);
    st.site = &p_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, F, y.size(1), F.size(1));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, y, F, var_tlinv);
    iv[0] = (*(int32_T(*)[2])var_tlinv.size())[0];
    iv[1] = (*(int32_T(*)[2])var_tlinv.size())[1];
    input_sizes[0] = (*(int32_T(*)[2])Q.size())[0];
    input_sizes[1] = (*(int32_T(*)[2])Q.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &input_sizes[0], &v_emlrtECI,
                             (emlrtCTX)sp);
    loop_ub = var_tlinv.size(0) * var_tlinv.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      var_tlinv[i1] = var_tlinv[i1] + Q[i1];
    }
    //  k by k
    input_sizes[0] = var_tlinv.size(1);
    input_sizes[1] = var_tlinv.size(0);
    iv[0] = (*(int32_T(*)[2])var_tlinv.size())[0];
    iv[1] = (*(int32_T(*)[2])var_tlinv.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &input_sizes[0], &w_emlrtECI,
                             (emlrtCTX)sp);
    b_P_ttm.set_size(&jc_emlrtRTEI, sp, var_tlinv.size(0), var_tlinv.size(1));
    loop_ub = var_tlinv.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      result_idx_1 = var_tlinv.size(0);
      for (i2 = 0; i2 < result_idx_1; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            (var_tlinv[i2 + var_tlinv.size(0) * i1] +
             var_tlinv[i1 + var_tlinv.size(0) * i2]) /
            2.0;
      }
    }
    var_tlinv.set_size(&kc_emlrtRTEI, sp, b_P_ttm.size(0), b_P_ttm.size(1));
    loop_ub = b_P_ttm.size(0) * b_P_ttm.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      var_tlinv[i1] = b_P_ttm[i1];
    }
    st.site = &q_emlrtRSI;
    invpd(&st, var_tlinv, Kalgain);
    //  k by k
    i1 = Um.size(0);
    if ((result + 2 < 1) || (result + 2 > i1)) {
      emlrtDynamicBoundsCheckR2012b(result + 2, 1, i1, &ec_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    loop_ub = Um.size(1);
    e_tl.set_size(&lc_emlrtRTEI, sp, loop_ub);
    for (i1 = 0; i1 < loop_ub; i1++) {
      e_tl[i1] = Um[(result + Um.size(0) * i1) + 1];
    }
    st.site = &r_emlrtRSI;
    f_tt.set_size(&mc_emlrtRTEI, &st, i7);
    for (i1 = 0; i1 < f_loop_ub; i1++) {
      f_tt[i1] = f_ttm[i1 + f_ttm.size(0) * result];
    }
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, F, f_tt, F.size(1), i6);
    f_tt.set_size(&mc_emlrtRTEI, &st, i8);
    for (i1 = 0; i1 < g_loop_ub; i1++) {
      f_tt[i1] = f_ttm[i1 + f_ttm.size(0) * result];
    }
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, F, f_tt, r);
    if (e_tl.size(0) != r.size(0)) {
      emlrtSizeEqCheck1DR2012b(e_tl.size(0), r.size(0), &y_emlrtECI,
                               (emlrtCTX)sp);
    }
    loop_ub = e_tl.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      e_tl[i1] = e_tl[i1] - r[i1];
    }
    //  k by 1
    st.site = &s_emlrtRSI;
    b_P_ttm.set_size(&ic_emlrtRTEI, &st, h_loop_ub, i10);
    for (i1 = 0; i1 < i_loop_ub; i1++) {
      for (i2 = 0; i2 < h_loop_ub; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            P_ttm[(i2 + P_ttm.size(0) * i1) +
                  P_ttm.size(0) * P_ttm.size(1) * result];
      }
    }
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, b_P_ttm, F, i9, F.size(1));
    b_P_ttm.set_size(&ic_emlrtRTEI, &st, j_loop_ub, i11);
    for (i1 = 0; i1 < k_loop_ub; i1++) {
      for (i2 = 0; i2 < j_loop_ub; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            P_ttm[(i2 + P_ttm.size(0) * i1) +
                  P_ttm.size(0) * P_ttm.size(1) * result];
      }
    }
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, b_P_ttm, F, y);
    st.site = &s_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, Kalgain, y.size(1), Kalgain.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, Kalgain, P_tt);
    //  km3 by k
    st.site = &t_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, P_tt, e_tl, P_tt.size(1), e_tl.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, P_tt, e_tl, f_tl);
    if (i12 != f_tl.size(0)) {
      emlrtSizeEqCheck1DR2012b(i12, f_tl.size(0), &ab_emlrtECI, (emlrtCTX)sp);
    }
    f_tl.set_size(&nc_emlrtRTEI, sp, i13);
    for (i1 = 0; i1 < l_loop_ub; i1++) {
      f_tl[i1] = f_ttm[i1 + f_ttm.size(0) * result] + f_tl[i1];
    }
    //  km3 by 1
    st.site = &u_emlrtRSI;
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, P_tt, F, P_tt.size(1), F.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, P_tt, F, y);
    st.site = &u_emlrtRSI;
    b_P_ttm.set_size(&ic_emlrtRTEI, &st, m_loop_ub, i15);
    for (i1 = 0; i1 < n_loop_ub; i1++) {
      for (i2 = 0; i2 < m_loop_ub; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            P_ttm[(i2 + P_ttm.size(0) * i1) +
                  P_ttm.size(0) * P_ttm.size(1) * result];
      }
    }
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, y, b_P_ttm, y.size(1), i14);
    b_P_ttm.set_size(&ic_emlrtRTEI, &st, o_loop_ub, i16);
    for (i1 = 0; i1 < p_loop_ub; i1++) {
      for (i2 = 0; i2 < o_loop_ub; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            P_ttm[(i2 + P_ttm.size(0) * i1) +
                  P_ttm.size(0) * P_ttm.size(1) * result];
      }
    }
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, y, b_P_ttm, P_tt);
    //  km3 by km3
    input_sizes[0] = i17;
    input_sizes[1] = i18;
    iv[0] = (*(int32_T(*)[2])P_tt.size())[0];
    iv[1] = (*(int32_T(*)[2])P_tt.size())[1];
    emlrtSizeEqCheckNDR2012b(&input_sizes[0], &iv[0], &bb_emlrtECI,
                             (emlrtCTX)sp);
    P_tt.set_size(&oc_emlrtRTEI, sp, q_loop_ub, i19);
    for (i1 = 0; i1 < r_loop_ub; i1++) {
      for (i2 = 0; i2 < q_loop_ub; i2++) {
        P_tt[i2 + P_tt.size(0) * i1] =
            P_ttm[(i2 + P_ttm.size(0) * i1) +
                  P_ttm.size(0) * P_ttm.size(1) * result] -
            P_tt[i2 + P_tt.size(0) * i1];
      }
    }
    input_sizes[0] = P_tt.size(1);
    input_sizes[1] = P_tt.size(0);
    iv[0] = (*(int32_T(*)[2])P_tt.size())[0];
    iv[1] = (*(int32_T(*)[2])P_tt.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &input_sizes[0], &cb_emlrtECI,
                             (emlrtCTX)sp);
    b_P_ttm.set_size(&pc_emlrtRTEI, sp, P_tt.size(0), P_tt.size(1));
    loop_ub = P_tt.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      result_idx_1 = P_tt.size(0);
      for (i2 = 0; i2 < result_idx_1; i2++) {
        b_P_ttm[i2 + b_P_ttm.size(0) * i1] =
            (P_tt[i2 + P_tt.size(0) * i1] + P_tt[i1 + P_tt.size(0) * i2]) / 2.0;
      }
    }
    P_tt.set_size(&qc_emlrtRTEI, sp, b_P_ttm.size(0), b_P_ttm.size(1));
    loop_ub = b_P_ttm.size(0) * b_P_ttm.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      P_tt[i1] = b_P_ttm[i1];
    }
    st.site = &v_emlrtRSI;
    cholmod(&st, P_tt, var_tlinv);
    st.site = &w_emlrtRSI;
    b_st.site = &w_emlrtRSI;
    coder::randn(&b_st, static_cast<real_T>(sizes_idx_1), f_tt);
    b_st.site = &fb_emlrtRSI;
    coder::dynamic_size_checks(&b_st, var_tlinv, f_tt, var_tlinv.size(0),
                               f_tt.size(0));
    b_st.site = &eb_emlrtRSI;
    coder::internal::blas::b_mtimes(&b_st, var_tlinv, f_tt, r);
    if (f_tl.size(0) != r.size(0)) {
      emlrtSizeEqCheck1DR2012b(f_tl.size(0), r.size(0), &db_emlrtECI,
                               (emlrtCTX)sp);
    }
    loop_ub = f_tl.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      f_tl[i1] = f_tl[i1] + r[i1];
    }
    i1 = Um.size(0);
    if ((result + 1 < 1) || (result + 1 > i1)) {
      emlrtDynamicBoundsCheckR2012b(result + 1, 1, i1, &fc_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    input_sizes[0] = 1;
    input_sizes[1] = Um.size(1);
    b_result[1] = f_tl.size(0);
    emlrtSubAssignSizeCheckR2012b(&input_sizes[0], 2, &b_result[0], 2,
                                  &eb_emlrtECI, (emlrtCTX)sp);
    loop_ub = f_tl.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      Um[result + Um.size(0) * i1] = f_tl[i1];
    }
    //  1 by by k
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  Uttm.set_size(&rc_emlrtRTEI, sp, f_ttm.size(1), f_ttm.size(0));
  loop_ub = f_ttm.size(0);
  for (i = 0; i < loop_ub; i++) {
    result_idx_1 = f_ttm.size(1);
    for (i1 = 0; i1 < result_idx_1; i1++) {
      Uttm[i1 + Uttm.size(0) * i] = f_ttm[i + f_ttm.size(0) * i1];
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (Gen_Um.cpp)
