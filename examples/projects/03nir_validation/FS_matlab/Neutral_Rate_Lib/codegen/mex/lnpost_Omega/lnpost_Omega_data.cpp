//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpost_Omega_data.cpp
//
// Code generation for function 'lnpost_Omega_data'
//

// Include files
#include "lnpost_Omega_data.h"
#include "rt_nonfinite.h"

// Variable Definitions
emlrtCTX emlrtRootTLSGlobal{nullptr};

const volatile char_T *emlrtBreakCheckR2012bFlagVar{nullptr};

emlrtContext emlrtContextGlobal{
    true,                                                 // bFirstTime
    false,                                                // bInitialized
    131610U,                                              // fVersionInfo
    nullptr,                                              // fErrorFunction
    "lnpost_Omega",                                       // fFunctionName
    nullptr,                                              // fRTCallStack
    false,                                                // bDebugMode
    {2045744189U, 2170104910U, 2743257031U, 4284093946U}, // fSigWrd
    nullptr                                               // fSigMem
};

emlrtRSInfo
    f_emlrtRSI{
        91,                  // lineNo
        "eml_mtimes_helper", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pathName
    };

emlrtRSInfo
    g_emlrtRSI{
        60,                  // lineNo
        "eml_mtimes_helper", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pathName
    };

emlrtRSInfo h_emlrtRSI{
    142,      // lineNo
    "mtimes", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "blas\\mtimes.m" // pathName
};

emlrtRSInfo i_emlrtRSI{
    21,                               // lineNo
    "eml_int_forloop_overflow_check", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\eml\\eml_int_forloop_"
    "overflow_check.m" // pathName
};

emlrtRSInfo j_emlrtRSI{
    178,           // lineNo
    "mtimes_blas", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "blas\\mtimes.m" // pathName
};

emlrtRSInfo fb_emlrtRSI{
    20,    // lineNo
    "sum", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\sum.m" // pathName
};

emlrtRSInfo gb_emlrtRSI{
    99,        // lineNo
    "sumprod", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumpro"
    "d.m" // pathName
};

emlrtRSInfo hb_emlrtRSI{
    74,                      // lineNo
    "combineVectorElements", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\combin"
    "eVectorElements.m" // pathName
};

emlrtRSInfo ib_emlrtRSI{
    107,                // lineNo
    "blockedSummation", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\blocke"
    "dSummation.m" // pathName
};

emlrtRSInfo jb_emlrtRSI{
    22,                    // lineNo
    "sumMatrixIncludeNaN", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

emlrtRSInfo lb_emlrtRSI{
    42,                 // lineNo
    "sumMatrixColumns", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

emlrtRSInfo nb_emlrtRSI{
    57,                 // lineNo
    "sumMatrixColumns", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\private\\sumMat"
    "rixIncludeNaN.m" // pathName
};

emlrtRSInfo
    lc_emlrtRSI{
        96,    // lineNo
        "eye", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\eye.m" // pathName
    };

emlrtRSInfo dd_emlrtRSI{
    27,       // lineNo
    "xgetrf", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pathName
};

emlrtRSInfo ed_emlrtRSI{
    91,             // lineNo
    "ceval_xgetrf", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pathName
};

emlrtRSInfo fd_emlrtRSI{
    58,             // lineNo
    "ceval_xgetrf", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pathName
};

omp_lock_t emlrtLockGlobal;

omp_nest_lock_t emlrtNestLockGlobal;

emlrtRTEInfo
    emlrtRTEI{
        130,                   // lineNo
        23,                    // colNo
        "dynamic_size_checks", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pName
    };

emlrtRTEInfo
    b_emlrtRTEI{
        135,                   // lineNo
        23,                    // colNo
        "dynamic_size_checks", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pName
    };

emlrtRTEInfo p_emlrtRTEI{
    47,          // lineNo
    13,          // colNo
    "infocheck", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\infocheck.m" // pName
};

emlrtRTEInfo q_emlrtRTEI{
    44,          // lineNo
    13,          // colNo
    "infocheck", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\infocheck.m" // pName
};

emlrtRTEInfo db_emlrtRTEI{
    218,      // lineNo
    20,       // colNo
    "mtimes", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "blas\\mtimes.m" // pName
};

emlrtRTEInfo mc_emlrtRTEI{
    1,        // lineNo
    37,       // colNo
    "xgetrf", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pName
};

emlrtRTEInfo nc_emlrtRTEI{
    58,       // lineNo
    29,       // colNo
    "xgetrf", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pName
};

emlrtRTEInfo oc_emlrtRTEI{
    89,       // lineNo
    27,       // colNo
    "xgetrf", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\+"
    "lapack\\xgetrf.m" // pName
};

emlrtRTEInfo pc_emlrtRTEI{
    53,       // lineNo
    9,        // colNo
    "repmat", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\repmat.m" // pName
};

const char_T cv[19]{'L', 'A', 'P', 'A', 'C', 'K', 'E', '_', 'd', 'g',
                    'e', 't', 'r', 'f', '_', 'w', 'o', 'r', 'k'};

// End of code generation (lnpost_Omega_data.cpp)
