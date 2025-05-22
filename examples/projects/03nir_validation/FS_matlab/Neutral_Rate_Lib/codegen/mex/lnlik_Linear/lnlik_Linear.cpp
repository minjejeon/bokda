//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnlik_Linear.cpp
//
// Code generation for function 'lnlik_Linear'
//

// Include files 
#include "lnlik_Linear.h"
#include "cholmod.h"
#include "diag.h"
#include "eml_int_forloop_overflow_check.h"
#include "eye.h"
#include "lnlik_Linear_data.h"
#include "mtimes.h"
#include "rt_nonfinite.h"
#include "sum.h"
#include "blas.h"
#include "coder_array.h"
#include "mwmathutil.h"
#include <cstddef>

// Variable Definitions 
static emlrtRSInfo emlrtRSI{ 5, // lineNo
"lnlik_Linear", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pathName
 };

static emlrtRSInfo b_emlrtRSI{ 13, // lineNo
"lnlik_Linear", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pathName
 };

static emlrtRSInfo c_emlrtRSI{ 16, // lineNo
"lnlik_Linear", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pathName
 };

static emlrtRSInfo d_emlrtRSI{ 19, // lineNo
"lnlik_Linear", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pathName
 };

static emlrtRSInfo l_emlrtRSI{ 23, // lineNo
"invpd", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m"// pathName
 };

static emlrtRSInfo m_emlrtRSI{ 22, // lineNo
"invpd", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m"// pathName
 };

static emlrtRSInfo n_emlrtRSI{ 17, // lineNo
"invpd", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m"// pathName
 };

static emlrtRSInfo pc_emlrtRSI{ 14, // lineNo
"invuptr", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pathName
 };

static emlrtRSInfo qc_emlrtRSI{ 23, // lineNo
"invuptr", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pathName
 };

static emlrtRSInfo sc_emlrtRSI{ 3, // lineNo
"lnpdfmvn1", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1.m"// pathName
 };

static emlrtRSInfo tc_emlrtRSI{ 4, // lineNo
"lnpdfmvn1", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1.m"// pathName
 };

static emlrtRSInfo uc_emlrtRSI{ 5, // lineNo
"lnpdfmvn1", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1.m"// pathName
 };

static emlrtRSInfo vc_emlrtRSI{ 2, // lineNo
"lndet1", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lndet1.m"// pathName
 };

static emlrtRSInfo wc_emlrtRSI{ 17, // lineNo
"log", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m"// pathName
 };

static emlrtRSInfo xc_emlrtRSI{ 33, // lineNo
"applyScalarFunctionInPlace", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\applyScalarFunctionInPlace.m"// pathName
 };

static emlrtRSInfo yc_emlrtRSI{ 4, // lineNo
"sumc", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\sumc.m"// pathName
 };

static emlrtMCInfo emlrtMCI{ 11, // lineNo
10, // colNo
"invpd", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m"// pName
 };

static emlrtMCInfo c_emlrtMCI{ 11, // lineNo
10, // colNo
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pName
 };

static emlrtMCInfo d_emlrtMCI{ 17, // lineNo
13, // colNo
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pName
 };

static emlrtRTEInfo c_emlrtRTEI{ 14, // lineNo
9, // colNo
"log", // fName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m"// pName
 };

static emlrtECInfo emlrtECI{ -1, // nDims
6, // lineNo
12, // colNo
"lnpdfn1", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfn1.m"// pName
 };

static emlrtBCInfo emlrtBCI{ -1, // iFirst
-1, // iLast
23, // lineNo
37, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo b_emlrtBCI{ -1, // iFirst
-1, // iLast
23, // lineNo
35, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo c_emlrtBCI{ -1, // iFirst
-1, // iLast
23, // lineNo
31, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo d_emlrtBCI{ -1, // iFirst
-1, // iLast
23, // lineNo
26, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo e_emlrtBCI{ -1, // iFirst
-1, // iLast
23, // lineNo
22, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo f_emlrtBCI{ -1, // iFirst
-1, // iLast
23, // lineNo
20, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtRTEInfo d_emlrtRTEI{ 21, // lineNo
12, // colNo
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pName
 };

static emlrtRTEInfo e_emlrtRTEI{ 15, // lineNo
10, // colNo
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pName
 };

static emlrtECInfo b_emlrtECI{ -1, // nDims
4, // lineNo
8, // colNo
"lnpdfmvn1", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1.m"// pName
 };

static emlrtECInfo c_emlrtECI{ 2, // nDims
25, // lineNo
13, // colNo
"invpd", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m"// pName
 };

static emlrtECInfo d_emlrtECI{ 2, // nDims
17, // lineNo
20, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pName
 };

static emlrtECInfo e_emlrtECI{ 2, // nDims
15, // lineNo
17, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pName
 };

static emlrtBCInfo g_emlrtBCI{ -1, // iFirst
-1, // iLast
12, // lineNo
21, // colNo
"YLm", // aName
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m", // pName
0// checkKind
 };

static emlrtDCInfo emlrtDCI{ 12, // lineNo
21, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m", // pName
1// checkKind
 };

static emlrtBCInfo h_emlrtBCI{ -1, // iFirst
-1, // iLast
11, // lineNo
14, // colNo
"Y0", // aName
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m", // pName
0// checkKind
 };

static emlrtDCInfo b_emlrtDCI{ 11, // lineNo
14, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m", // pName
1// checkKind
 };

static emlrtRTEInfo f_emlrtRTEI{ 10, // lineNo
9, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pName
 };

static emlrtBCInfo i_emlrtBCI{ -1, // iFirst
-1, // iLast
16, // lineNo
16, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo j_emlrtBCI{ -1, // iFirst
-1, // iLast
20, // lineNo
15, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo k_emlrtBCI{ -1, // iFirst
-1, // iLast
20, // lineNo
4, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo l_emlrtBCI{ -1, // iFirst
-1, // iLast
24, // lineNo
20, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtBCInfo m_emlrtBCI{ -1, // iFirst
-1, // iLast
24, // lineNo
6, // colNo
"T", // aName
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
0// checkKind
 };

static emlrtRTEInfo n_emlrtRTEI{ 11, // lineNo
5, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pName
 };

static emlrtRTEInfo o_emlrtRTEI{ 12, // lineNo
5, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pName
 };

static emlrtRTEInfo p_emlrtRTEI{ 15, // lineNo
5, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pName
 };

static emlrtRTEInfo q_emlrtRTEI{ 23, // lineNo
18, // colNo
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pName
 };

static emlrtRTEInfo r_emlrtRTEI{ 23, // lineNo
29, // colNo
"invuptr", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pName
 };

static emlrtRTEInfo t_emlrtRTEI{ 16, // lineNo
5, // colNo
"lnlik_Linear", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear.m"// pName
 };

static emlrtRTEInfo u_emlrtRTEI{ 25, // lineNo
12, // colNo
"invpd", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m"// pName
 };

static emlrtRTEInfo v_emlrtRTEI{ 77, // lineNo
13, // colNo
"eml_mtimes_helper", // fName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_helper.m"// pName
 };

static emlrtRTEInfo w_emlrtRTEI{ 6, // lineNo
12, // colNo
"lnpdfn1", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfn1.m"// pName
 };

static emlrtRSInfo ad_emlrtRSI{ 11, // lineNo
"invpd", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m"// pathName
 };

static emlrtRSInfo bd_emlrtRSI{ 11, // lineNo
"invuptr", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pathName
 };

static emlrtRSInfo cd_emlrtRSI{ 17, // lineNo
"invuptr", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m"// pathName
 };


// Function Declarations 
static void disp(const emlrtStack *sp, const mxArray *b, emlrtMCInfo *location);


// Function Definitions 
static void disp(const emlrtStack *sp, const mxArray *b, emlrtMCInfo *location)
{
    const mxArray *pArray;
    pArray = b;
    emlrtCallMATLABR2012b((emlrtCTX)sp, 0, nullptr, 1, &pArray, (const char_T *)"disp", true, location);
}

real_T lnlik_Linear(const emlrtStack *sp, const coder::array<real_T, 2U> &Y0, const coder::array<real_T, 3U> &YLm, const coder::array<real_T, 1U> &beta, const coder::array<real_T, 2U> &Omega)
{
    static const int32_T iv1[2]{ 1, 23 };
    static const int32_T iv2[2]{ 1, 23 };
    static const int32_T iv3[2]{ 1, 20 };
    static const char_T b_u[23]{ 'm', 'a', 't', 'r', 'i', 'x', ' ', 'T', ' ', ' ', 'i', 's', ' ', 'n', 'o', 't', ' ', 's', 'q', 'u', 'a', 'r', 'e' };
    static const char_T u[23]{ 'm', 'a', 't', 'r', 'i', 'x', ' ', 'A', ' ', ' ', 'i', 's', ' ', 'n', 'o', 't', ' ', 's', 'q', 'u', 'a', 'r', 'e' };
    static const char_T c_u[20]{ 'm', 'a', 't', 'r', 'i', 'x', ' ', 'T', ' ', 'i', 's', ' ', 's', 'i', 'n', 'g', 'u', 'l', 'a', 'r' };
    ptrdiff_t incx_t;
    ptrdiff_t incy_t;
    ptrdiff_t lda_t;
    ptrdiff_t ldb_t;
    ptrdiff_t ldc_t;
    ptrdiff_t n_t;
    coder::array<real_T, 2U> Hinv;
    coder::array<real_T, 2U> a;
    coder::array<real_T, 2U> ed_emlrtRSI;
    coder::array<real_T, 2U> x_t;
    coder::array<real_T, 1U> e;
    coder::array<real_T, 1U> y_t;
    coder::array<real_T, 1U> y_tL;
    emlrtStack b_st;
    emlrtStack c_st;
    emlrtStack d_st;
    emlrtStack e_st;
    emlrtStack f_st;
    emlrtStack g_st;
    emlrtStack st;
    const mxArray *b_y;
    const mxArray *c_y;
    const mxArray *m;
    const mxArray *y;
    real_T P;
    real_T T;
    real_T beta1;
    real_T lnL;
    int32_T b_Omega[2];
    int32_T iv[2];
    int32_T i;
    char_T TRANSA1;
    char_T TRANSB1;
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
    emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
    //  우도함수 계산하기
    st.site = &emlrtRSI;
    b_st.site = &e_emlrtRSI;
    c_st.site = &f_emlrtRSI;
    P = static_cast<real_T>(YLm.size(1)) / (static_cast<real_T>(YLm.size(0)) * static_cast<real_T>(YLm.size(0)));
    T = static_cast<real_T>(YLm.size(2)) + P;
    lnL = 0.0;
    i = static_cast<int32_T>(T + (1.0 - (P + 1.0)));
    emlrtForLoopVectorCheckR2021a(P + 1.0, 1.0, T, mxDOUBLE_CLASS, i, &f_emlrtRTEI, (emlrtCTX)sp);
    for ( int32_T t{0}; t < i; t++) {
        int32_T i1;
        int32_T i2;
        int32_T k;
        int32_T loop_ub;
        int32_T nx;
        boolean_T p;
        T = ((P + 1.0) + static_cast<real_T>(t)) - P;
        beta1 = static_cast<int32_T>(muDoubleScalarFloor(T));
        if (T != beta1) {
            emlrtIntegerCheckR2012b(T, &b_emlrtDCI, (emlrtCTX)sp);
        }
        if ((static_cast<int32_T>(T) < 1) || (static_cast<int32_T>(T) > Y0.size(0))) {
            emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(T), 1, Y0.size(0), &h_emlrtBCI, (emlrtCTX)sp);
        }
        loop_ub = Y0.size(1);
        y_t.set_size(&n_emlrtRTEI, sp, Y0.size(1));
        for (i1 = 0; i1 < loop_ub; i1++) {
            y_t[i1] = Y0[(static_cast<int32_T>(T) + Y0.size(0) * i1) - 1];
        }
        if (T != beta1) {
            emlrtIntegerCheckR2012b(T, &emlrtDCI, (emlrtCTX)sp);
        }
        if ((static_cast<int32_T>(T) < 1) || (static_cast<int32_T>(T) > YLm.size(2))) {
            emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(T), 1, YLm.size(2), &g_emlrtBCI, (emlrtCTX)sp);
        }
        loop_ub = YLm.size(0);
        nx = YLm.size(1);
        x_t.set_size(&o_emlrtRTEI, sp, YLm.size(0), YLm.size(1));
        for (i1 = 0; i1 < nx; i1++) {
            for (i2 = 0; i2 < loop_ub; i2++) {
                x_t[i2 + x_t.size(0) * i1] = YLm[(i2 + YLm.size(0) * i1) + YLm.size(0) * YLm.size(1) * (static_cast<int32_T>(T) - 1)];
            }
        }
        st.site = &b_emlrtRSI;
        b_st.site = &h_emlrtRSI;
        if (beta.size(0) != YLm.size(1)) {
            if (((YLm.size(0) == 1) && (YLm.size(1) == 1)) || (beta.size(0) == 1)) {
                emlrtErrorWithMessageIdR2018a(&b_st, &emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion", "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
            } else {
                emlrtErrorWithMessageIdR2018a(&b_st, &b_emlrtRTEI, "MATLAB:innerdim", "MATLAB:innerdim", 0);
            }
        }
        b_st.site = &g_emlrtRSI;
        coder::internal::blas::mtimes(&b_st, x_t, beta, y_tL);
        b_Omega[0] = Omega.size(1);
        b_Omega[1] = Omega.size(0);
        iv[0] = (*(int32_T (*)[2])((coder::array<real_T, 2U> *)&Omega)->size())[0];
        iv[1] = (*(int32_T (*)[2])((coder::array<real_T, 2U> *)&Omega)->size())[1];
        emlrtSizeEqCheckNDR2012b(&iv[0], &b_Omega[0], &e_emlrtECI, (emlrtCTX)sp);
        loop_ub = Omega.size(1);
        x_t.set_size(&p_emlrtRTEI, sp, Omega.size(0), Omega.size(1));
        for (i1 = 0; i1 < loop_ub; i1++) {
            nx = Omega.size(0);
            for (i2 = 0; i2 < nx; i2++) {
                x_t[i2 + x_t.size(0) * i1] = 0.5 * (Omega[i2 + Omega.size(0) * i1] + Omega[i1 + Omega.size(0) * i2]);
            }
        }
        st.site = &c_emlrtRSI;
        // 'invpd' Inverse of a symmetric positive definite matrix using Cholesky factorization 
        // Ainv = invpd(A) computes the inverse of a symmetric positive 
        // definite matrix A using its Cholesky factor H.
        // inv(A) = inv(H)inv(H)'. 
        // input  : Matrix A
        // output : Ainv, err (=1 if error, and 0 if no error) 
        if (x_t.size(0) != x_t.size(1)) {
            y = nullptr;
            m = emlrtCreateCharArray(2, &iv1[0]);
            emlrtInitCharArrayR2013a(&st, 23, m, &u[0]);
            emlrtAssign(&y, m);
            b_st.site = &ad_emlrtRSI;
            disp(&b_st, y, &emlrtMCI);
            x_t.set_size(&t_emlrtRTEI, &st, 0, 0);
        } else {
            int32_T n;
            b_st.site = &n_emlrtRSI;
            cholmod(&b_st, x_t, Hinv, &T);
            b_st.site = &m_emlrtRSI;
            // INVUPTR Inverse of an upper triangular matrix 
            // T = invuptr(T) computes the inverse of a nonsingular upper triangular
            // matrix T.  The output matrix T contains the inverse of T. 
            // This program implements Algorithm 4.2.2 of the book.
            // Input  : Matrix T 
            // output : Matrix T
            n = Hinv.size(1);
            if (Hinv.size(0) != n) {
                b_y = nullptr;
                m = emlrtCreateCharArray(2, &iv2[0]);
                emlrtInitCharArrayR2013a(&b_st, 23, m, &b_u[0]);
                emlrtAssign(&b_y, m);
                c_st.site = &bd_emlrtRSI;
                disp(&c_st, b_y, &c_emlrtMCI);
            } else {
                boolean_T exitg1;
                c_st.site = &pc_emlrtRSI;
                coder::eye(&c_st, static_cast<real_T>(n), static_cast<real_T>(n), ed_emlrtRSI);
                i1 = static_cast<int32_T>(((-1.0 - static_cast<real_T>(n)) + 1.0) / -1.0);
                emlrtForLoopVectorCheckR2021a(static_cast<real_T>(n), -1.0, 1.0, mxDOUBLE_CLASS, i1, &e_emlrtRTEI, &b_st);
                k = 0;
                exitg1 = false;
                while ((!exitg1) && (k <= i1 - 1)) {
                    int32_T b_k;
                    b_k = n - k;
                    if ((b_k < 1) || (b_k > Hinv.size(0))) {
                        emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(0), &i_emlrtBCI, &b_st);
                    }
                    if (b_k > Hinv.size(1)) {
                        emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &i_emlrtBCI, &b_st);
                    }
                    T = Hinv[(b_k + Hinv.size(0) * (b_k - 1)) - 1];
                    if (T == 0.0) {
                        c_y = nullptr;
                        m = emlrtCreateCharArray(2, &iv3[0]);
                        emlrtInitCharArrayR2013a(&b_st, 20, m, &c_u[0]);
                        emlrtAssign(&c_y, m);
                        c_st.site = &cd_emlrtRSI;
                        disp(&c_st, c_y, &d_emlrtMCI);
                        exitg1 = true;
                    } else {
                        if (b_k > Hinv.size(0)) {
                            emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(0), &j_emlrtBCI, &b_st);
                        }
                        if (b_k > Hinv.size(1)) {
                            emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &j_emlrtBCI, &b_st);
                        }
                        if (b_k > Hinv.size(0)) {
                            emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(0), &k_emlrtBCI, &b_st);
                        }
                        if (b_k > Hinv.size(1)) {
                            emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &k_emlrtBCI, &b_st);
                        }
                        Hinv[(b_k + Hinv.size(0) * (b_k - 1)) - 1] = 1.0 / T;
                        i2 = static_cast<int32_T>(((-1.0 - (static_cast<real_T>(b_k) - 1.0)) + 1.0) / -1.0);
                        emlrtForLoopVectorCheckR2021a(static_cast<real_T>(b_k) - 1.0, -1.0, 1.0, mxDOUBLE_CLASS, i2, &d_emlrtRTEI, &b_st);
                        for ( int32_T b_i{0}; b_i < i2; b_i++) {
                            int32_T c_i;
                            int32_T i3;
                            int32_T i4;
                            int32_T i5;
                            int32_T i6;
                            c_i = b_k - b_i;
                            if (c_i > b_k) {
                                i3 = -1;
                                i4 = -1;
                                i5 = -1;
                                i6 = -1;
                            } else {
                                if ((c_i < 1) || (c_i > Hinv.size(1))) {
                                    emlrtDynamicBoundsCheckR2012b(c_i, 1, Hinv.size(1), &e_emlrtBCI, &b_st);
                                }
                                i3 = c_i - 2;
                                if (b_k > Hinv.size(1)) {
                                    emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &d_emlrtBCI, &b_st);
                                }
                                i4 = b_k - 1;
                                if (c_i > Hinv.size(0)) {
                                    emlrtDynamicBoundsCheckR2012b(c_i, 1, Hinv.size(0), &c_emlrtBCI, &b_st);
                                }
                                i5 = c_i - 2;
                                if (b_k > Hinv.size(0)) {
                                    emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(0), &b_emlrtBCI, &b_st);
                                }
                                i6 = b_k - 1;
                            }
                            c_st.site = &qc_emlrtRSI;
                            if ((c_i - 1 < 1) || (c_i - 1 > Hinv.size(0))) {
                                emlrtDynamicBoundsCheckR2012b(c_i - 1, 1, Hinv.size(0), &f_emlrtBCI, &c_st);
                            }
                            loop_ub = i4 - i3;
                            a.set_size(&q_emlrtRTEI, &c_st, 1, loop_ub);
                            for (nx = 0; nx < loop_ub; nx++) {
                                a[nx] = Hinv[(c_i + Hinv.size(0) * ((i3 + nx) + 1)) - 2];
                            }
                            if (b_k > Hinv.size(1)) {
                                emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &emlrtBCI, &c_st);
                            }
                            nx = i6 - i5;
                            e.set_size(&r_emlrtRTEI, &c_st, nx);
                            for (i6 = 0; i6 < nx; i6++) {
                                e[i6] = Hinv[((i5 + i6) + Hinv.size(0) * (b_k - 1)) + 1];
                            }
                            d_st.site = &h_emlrtRSI;
                            if (loop_ub != nx) {
                                if ((loop_ub == 1) || (nx == 1)) {
                                    emlrtErrorWithMessageIdR2018a(&d_st, &emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion", "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
                                } else {
                                    emlrtErrorWithMessageIdR2018a(&d_st, &b_emlrtRTEI, "MATLAB:innerdim", "MATLAB:innerdim", 0);
                                }
                            }
                            if (loop_ub < 1) {
                                T = 0.0;
                            } else {
                                n_t = (ptrdiff_t)(i4 - i3);
                                incx_t = (ptrdiff_t)1;
                                incy_t = (ptrdiff_t)1;
                                T = ddot(&n_t, &a[0], &incx_t, &(e.data())[0], &incy_t);
                            }
                            if ((c_i - 1 < 1) || (c_i - 1 > Hinv.size(0))) {
                                emlrtDynamicBoundsCheckR2012b(c_i - 1, 1, Hinv.size(0), &l_emlrtBCI, &b_st);
                            }
                            if ((c_i - 1 < 1) || (c_i - 1 > Hinv.size(1))) {
                                emlrtDynamicBoundsCheckR2012b(c_i - 1, 1, Hinv.size(1), &l_emlrtBCI, &b_st);
                            }
                            if ((c_i - 1 < 1) || (c_i - 1 > Hinv.size(0))) {
                                emlrtDynamicBoundsCheckR2012b(c_i - 1, 1, Hinv.size(0), &m_emlrtBCI, &b_st);
                            }
                            if (b_k > Hinv.size(1)) {
                                emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &m_emlrtBCI, &b_st);
                            }
                            Hinv[(c_i + Hinv.size(0) * (b_k - 1)) - 2] = -T / Hinv[(c_i + Hinv.size(0) * (c_i - 2)) - 2];
                            if (*emlrtBreakCheckR2012bFlagVar != 0) {
                                emlrtBreakCheckR2012b(&b_st);
                            }
                        }
                        k++;
                        if (*emlrtBreakCheckR2012bFlagVar != 0) {
                            emlrtBreakCheckR2012b(&b_st);
                        }
                    }
                }
            }
            b_st.site = &l_emlrtRSI;
            c_st.site = &h_emlrtRSI;
            c_st.site = &g_emlrtRSI;
            if ((Hinv.size(0) == 0) || (Hinv.size(1) == 0)) {
                x_t.set_size(&t_emlrtRTEI, &c_st, Hinv.size(0), Hinv.size(0));
                loop_ub = Hinv.size(0) * Hinv.size(0);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    x_t[i1] = 0.0;
                }
            } else {
                d_st.site = &i_emlrtRSI;
                e_st.site = &k_emlrtRSI;
                TRANSB1 = 'T';
                TRANSA1 = 'N';
                T = 1.0;
                beta1 = 0.0;
                incx_t = (ptrdiff_t)Hinv.size(0);
                n_t = (ptrdiff_t)Hinv.size(0);
                incy_t = (ptrdiff_t)Hinv.size(1);
                lda_t = (ptrdiff_t)Hinv.size(0);
                ldb_t = (ptrdiff_t)Hinv.size(0);
                ldc_t = (ptrdiff_t)Hinv.size(0);
                x_t.set_size(&s_emlrtRTEI, &e_st, Hinv.size(0), Hinv.size(0));
                dgemm(&TRANSA1, &TRANSB1, &incx_t, &n_t, &incy_t, &T, &(Hinv.data())[0], &lda_t, &(Hinv.data())[0], &ldb_t, &beta1, &(x_t.data())[0], &ldc_t);
            }
            b_Omega[0] = x_t.size(1);
            b_Omega[1] = x_t.size(0);
            iv[0] = (*(int32_T (*)[2])x_t.size())[0];
            iv[1] = (*(int32_T (*)[2])x_t.size())[1];
            emlrtSizeEqCheckNDR2012b(&iv[0], &b_Omega[0], &c_emlrtECI, &st);
            Hinv.set_size(&u_emlrtRTEI, &st, x_t.size(0), x_t.size(1));
            loop_ub = x_t.size(1);
            for (i1 = 0; i1 < loop_ub; i1++) {
                nx = x_t.size(0);
                for (i2 = 0; i2 < nx; i2++) {
                    Hinv[i2 + Hinv.size(0) * i1] = (x_t[i2 + x_t.size(0) * i1] + x_t[i1 + x_t.size(0) * i2]) / 2.0;
                }
            }
            x_t.set_size(&t_emlrtRTEI, &st, Hinv.size(0), Hinv.size(1));
            loop_ub = Hinv.size(0) * Hinv.size(1);
            for (i1 = 0; i1 < loop_ub; i1++) {
                x_t[i1] = Hinv[i1];
            }
        }
        b_Omega[0] = x_t.size(1);
        b_Omega[1] = x_t.size(0);
        iv[0] = (*(int32_T (*)[2])x_t.size())[0];
        iv[1] = (*(int32_T (*)[2])x_t.size())[1];
        emlrtSizeEqCheckNDR2012b(&iv[0], &b_Omega[0], &d_emlrtECI, (emlrtCTX)sp);
        st.site = &d_emlrtRSI;
        //  uses precision instead of var $/
        Hinv.set_size(&v_emlrtRTEI, &st, x_t.size(0), x_t.size(1));
        loop_ub = x_t.size(1);
        for (i1 = 0; i1 < loop_ub; i1++) {
            nx = x_t.size(0);
            for (i2 = 0; i2 < nx; i2++) {
                Hinv[i2 + Hinv.size(0) * i1] = 0.5 * (x_t[i2 + x_t.size(0) * i1] + x_t[i1 + x_t.size(0) * i2]);
            }
        }
        b_st.site = &sc_emlrtRSI;
        cholmod(&b_st, Hinv, x_t);
        //  the matrix that makes the y uncorrelated $/
        if (y_t.size(0) != y_tL.size(0)) {
            emlrtSizeEqCheck1DR2012b(y_t.size(0), y_tL.size(0), &b_emlrtECI, &st);
        }
        b_st.site = &tc_emlrtRSI;
        loop_ub = y_t.size(0);
        for (i1 = 0; i1 < loop_ub; i1++) {
            y_t[i1] = y_t[i1] - y_tL[i1];
        }
        c_st.site = &h_emlrtRSI;
        if (y_t.size(0) != x_t.size(1)) {
            if (((x_t.size(0) == 1) && (x_t.size(1) == 1)) || (y_t.size(0) == 1)) {
                emlrtErrorWithMessageIdR2018a(&c_st, &emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion", "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
            } else {
                emlrtErrorWithMessageIdR2018a(&c_st, &b_emlrtRTEI, "MATLAB:innerdim", "MATLAB:innerdim", 0);
            }
        }
        c_st.site = &g_emlrtRSI;
        coder::internal::blas::mtimes(&c_st, x_t, y_t, e);
        //  standard normals: k times m matrix $/
        b_st.site = &uc_emlrtRSI;
        c_st.site = &vc_emlrtRSI;
        d_st.site = &vc_emlrtRSI;
        e_st.site = &vc_emlrtRSI;
        coder::diag(&e_st, x_t, y_tL);
        p = false;
        i1 = y_tL.size(0);
        for (k = 0; k < i1; k++) {
            if (p || (y_tL[k] < 0.0)) {
                p = true;
            }
        }
        if (p) {
            emlrtErrorWithMessageIdR2018a(&d_st, &c_emlrtRTEI, "Coder:toolbox:ElFunDomainError", "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
        }
        e_st.site = &wc_emlrtRSI;
        nx = y_tL.size(0);
        f_st.site = &xc_emlrtRSI;
        if ((1 <= y_tL.size(0)) && (y_tL.size(0) > 2147483646)) {
            g_st.site = &j_emlrtRSI;
            coder::check_forloop_overflow_error(&g_st);
        }
        for (k = 0; k < nx; k++) {
            y_tL[k] = muDoubleScalarLog(y_tL[k]);
        }
        //  gauss function
        d_st.site = &yc_emlrtRSI;
        T = coder::sum(&d_st, y_tL);
        b_st.site = &uc_emlrtRSI;
        //  log pdf of standard normal $/
        y_tL.set_size(&w_emlrtRTEI, &b_st, e.size(0));
        loop_ub = e.size(0);
        for (i1 = 0; i1 < loop_ub; i1++) {
            y_tL[i1] = 0.5 * e[i1];
        }
        if (y_tL.size(0) != e.size(0)) {
            emlrtSizeEqCheck1DR2012b(y_tL.size(0), e.size(0), &emlrtECI, &b_st);
        }
        b_st.site = &uc_emlrtRSI;
        //  gauss function
        loop_ub = y_tL.size(0);
        for (i1 = 0; i1 < loop_ub; i1++) {
            y_tL[i1] = -0.91893853320467267 - y_tL[i1] * e[i1];
        }
        c_st.site = &yc_emlrtRSI;
        beta1 = coder::sum(&c_st, y_tL);
        //  the log of the density $/
        lnL += 0.5 * (2.0 * T) + beta1;
        if (*emlrtBreakCheckR2012bFlagVar != 0) {
            emlrtBreakCheckR2012b((emlrtCTX)sp);
        }
    }
    emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
    return lnL;
}


// End of code generation (lnlik_Linear.cpp) 
