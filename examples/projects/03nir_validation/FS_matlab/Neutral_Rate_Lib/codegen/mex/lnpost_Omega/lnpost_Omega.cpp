//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpost_Omega.cpp
//
// Code generation for function 'lnpost_Omega'
//

// Include files 
#include "lnpost_Omega.h"
#include "applyScalarFunctionInPlace.h"
#include "det.h"
#include "eml_int_forloop_overflow_check.h"
#include "invpd.h"
#include "lnpost_Omega_data.h"
#include "mrdivide_helper.h"
#include "mtimes.h"
#include "rt_nonfinite.h"
#include "sumMatrixIncludeNaN.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions 
static emlrtRSInfo emlrtRSI{ 11, // lineNo
"lnpost_Omega", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pathName
 };

static emlrtRSInfo b_emlrtRSI{ 15, // lineNo
"lnpost_Omega", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pathName
 };

static emlrtRSInfo c_emlrtRSI{ 16, // lineNo
"lnpost_Omega", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pathName
 };

static emlrtRSInfo d_emlrtRSI{ 21, // lineNo
"lnpost_Omega", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pathName
 };

static emlrtRSInfo e_emlrtRSI{ 27, // lineNo
"lnpost_Omega", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pathName
 };

static emlrtRSInfo oc_emlrtRSI{ 117, // lineNo
"colon", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m"// pathName
 };

static emlrtRSInfo pc_emlrtRSI{ 311, // lineNo
"eml_float_colon", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m"// pathName
 };

static emlrtRSInfo qc_emlrtRSI{ 320, // lineNo
"eml_float_colon", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m"// pathName
 };

static emlrtRSInfo wc_emlrtRSI{ 16, // lineNo
"lnpdfIW", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo xc_emlrtRSI{ 17, // lineNo
"lnpdfIW", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo yc_emlrtRSI{ 18, // lineNo
"lnpdfIW", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo ad_emlrtRSI{ 20, // lineNo
"lnpdfIW", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo bd_emlrtRSI{ 21, // lineNo
"lnpdfIW", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo hd_emlrtRSI{ 20, // lineNo
"mrdivide_helper", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_helper.m"// pathName
 };

static emlrtRSInfo ue_emlrtRSI{ 32, // lineNo
"log_MGamf", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo ve_emlrtRSI{ 33, // lineNo
"log_MGamf", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo we_emlrtRSI{ 34, // lineNo
"log_MGamf", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo xe_emlrtRSI{ 35, // lineNo
"log_MGamf", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pathName
 };

static emlrtRSInfo ye_emlrtRSI{ 10, // lineNo
"gamma", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\specfun\\gamma.m"// pathName
 };

static emlrtRSInfo bf_emlrtRSI{ 17, // lineNo
"log", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m"// pathName
 };

static emlrtRSInfo cf_emlrtRSI{ 33, // lineNo
"applyScalarFunctionInPlace", // fcnName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\applyScalarFunctionInPlace.m"// pathName
 };

static emlrtRSInfo df_emlrtRSI{ 4, // lineNo
"sumc", // fcnName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\sumc.m"// pathName
 };

static emlrtRTEInfo d_emlrtRTEI{ 14, // lineNo
9, // colNo
"log", // fName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m"// pName
 };

static emlrtRTEInfo e_emlrtRTEI{ 417, // lineNo
15, // colNo
"assert_pmaxsize", // fName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m"// pName
 };

static emlrtRTEInfo f_emlrtRTEI{ 11, // lineNo
15, // colNo
"trace", // fName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\matfun\\trace.m"// pName
 };

static emlrtRTEInfo g_emlrtRTEI{ 16, // lineNo
19, // colNo
"mrdivide_helper", // fName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\mrdivide_helper.m"// pName
 };

static emlrtECInfo emlrtECI{ 2, // nDims
20, // lineNo
19, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtECInfo b_emlrtECI{ 2, // nDims
15, // lineNo
9, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtECInfo c_emlrtECI{ 2, // nDims
12, // lineNo
13, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtECInfo d_emlrtECI{ -1, // nDims
11, // lineNo
12, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtBCInfo emlrtBCI{ -1, // iFirst
-1, // iLast
11, // lineNo
14, // colNo
"Y", // aName
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m", // pName
0// checkKind
 };

static emlrtBCInfo b_emlrtBCI{ -1, // iFirst
-1, // iLast
10, // lineNo
16, // colNo
"X", // aName
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m", // pName
0// checkKind
 };

static emlrtRTEInfo t_emlrtRTEI{ 7, // lineNo
1, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtRTEInfo u_emlrtRTEI{ 11, // lineNo
12, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtRTEInfo v_emlrtRTEI{ 10, // lineNo
10, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtRTEInfo w_emlrtRTEI{ 12, // lineNo
21, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtRTEInfo x_emlrtRTEI{ 15, // lineNo
9, // colNo
"lnpost_Omega", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_Omega.m"// pName
 };

static emlrtRTEInfo y_emlrtRTEI{ 77, // lineNo
13, // colNo
"eml_mtimes_helper", // fName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_helper.m"// pName
 };

static emlrtRTEInfo ab_emlrtRTEI{ 312, // lineNo
20, // colNo
"colon", // fName
"C:\\Program Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\colon.m"// pName
 };

static emlrtRTEInfo bb_emlrtRTEI{ 32, // lineNo
1, // colNo
"lnpdfIW", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pName
 };

static emlrtRTEInfo cb_emlrtRTEI{ 34, // lineNo
18, // colNo
"lnpdfIW", // fName
"D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfIW.m"// pName
 };


// Function Definitions 
emlrtCTX emlrtGetRootTLSGlobal()
{
    return emlrtRootTLSGlobal;
}

void emlrtLockerFunction(EmlrtLockeeFunction aLockee, emlrtConstCTX aTLS, void *aData)
{
    omp_set_lock(&emlrtLockGlobal);
    emlrtCallLockeeFunction(aLockee, aTLS, aData);
    omp_unset_lock(&emlrtLockGlobal);
}

real_T lnpost_Omega(const emlrtStack *sp, const coder::array<real_T, 2U> &Y, const coder::array<real_T, 3U> &X, const coder::array<real_T, 1U> &beta, real_T nu, const coder::array<real_T, 2U> &R0, const coder::array<real_T, 2U> &Omega_st)
{
    coder::array<real_T, 2U> A;
    coder::array<real_T, 2U> b_X;
    coder::array<real_T, 2U> c_ehat2;
    coder::array<real_T, 2U> ehat2;
    coder::array<real_T, 1U> ehat;
    coder::array<real_T, 1U> x;
    emlrtStack b_st;
    emlrtStack c_st;
    emlrtStack d_st;
    emlrtStack e_st;
    emlrtStack f_st;
    emlrtStack g_st;
    emlrtStack h_st;
    emlrtStack i_st;
    emlrtStack j_st;
    emlrtStack st;
    real_T a;
    real_T b_x;
    real_T c_x;
    real_T kd;
    real_T lnpost_pdf;
    real_T nu1;
    real_T t;
    int32_T b_ehat2[2];
    int32_T iv[2];
    int32_T i;
    int32_T ib;
    int32_T nleft;
    int32_T nm1d2;
    int32_T nx;
    boolean_T p;
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
    j_st.prev = &i_st;
    j_st.tls = i_st.tls;
    emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
    //  Omega 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%
    //  number of columns
    ehat2.set_size(&t_emlrtRTEI, sp, Y.size(1), Y.size(1));
    nx = Y.size(1) * Y.size(1);
    for (i = 0; i < nx; i++) {
        ehat2[i] = 0.0;
    }
    //  잔차항의 제곱의 합
    i = Y.size(0);
    for (nleft = 0; nleft < i; nleft++) {
        int32_T i1;
        if (nleft + 1 > X.size(2)) {
            emlrtDynamicBoundsCheckR2012b(nleft + 1, 1, X.size(2), &b_emlrtBCI, (emlrtCTX)sp);
        }
        if (nleft + 1 > Y.size(0)) {
            emlrtDynamicBoundsCheckR2012b(nleft + 1, 1, Y.size(0), &emlrtBCI, (emlrtCTX)sp);
        }
        nx = Y.size(1);
        ehat.set_size(&u_emlrtRTEI, sp, Y.size(1));
        for (ib = 0; ib < nx; ib++) {
            ehat[ib] = Y[nleft + Y.size(0) * ib];
        }
        st.site = &emlrtRSI;
        b_st.site = &g_emlrtRSI;
        nx = X.size(1);
        if (beta.size(0) != X.size(1)) {
            if (((X.size(0) == 1) && (X.size(1) == 1)) || (beta.size(0) == 1)) {
                emlrtErrorWithMessageIdR2018a(&b_st, &emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion", "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
            } else {
                emlrtErrorWithMessageIdR2018a(&b_st, &b_emlrtRTEI, "MATLAB:innerdim", "MATLAB:innerdim", 0);
            }
        }
        nm1d2 = X.size(0);
        b_X.set_size(&v_emlrtRTEI, &st, X.size(0), X.size(1));
        for (ib = 0; ib < nx; ib++) {
            for (i1 = 0; i1 < nm1d2; i1++) {
                b_X[i1 + b_X.size(0) * ib] = X[(i1 + X.size(0) * ib) + X.size(0) * X.size(1) * nleft];
            }
        }
        b_st.site = &f_emlrtRSI;
        coder::internal::blas::mtimes(&b_st, b_X, beta, x);
        if (ehat.size(0) != x.size(0)) {
            emlrtSizeEqCheck1DR2012b(ehat.size(0), x.size(0), &d_emlrtECI, (emlrtCTX)sp);
        }
        nx = ehat.size(0);
        for (ib = 0; ib < nx; ib++) {
            ehat[ib] = ehat[ib] - x[ib];
        }
        //  잔차항
        b_X.set_size(&w_emlrtRTEI, sp, ehat.size(0), ehat.size(0));
        nx = ehat.size(0);
        for (ib = 0; ib < nx; ib++) {
            nm1d2 = ehat.size(0);
            for (i1 = 0; i1 < nm1d2; i1++) {
                b_X[i1 + b_X.size(0) * ib] = ehat[i1] * ehat[ib];
            }
        }
        iv[0] = (*(int32_T (*)[2])ehat2.size())[0];
        iv[1] = (*(int32_T (*)[2])ehat2.size())[1];
        b_ehat2[0] = (*(int32_T (*)[2])b_X.size())[0];
        b_ehat2[1] = (*(int32_T (*)[2])b_X.size())[1];
        emlrtSizeEqCheckNDR2012b(&iv[0], &b_ehat2[0], &c_emlrtECI, (emlrtCTX)sp);
        nx = ehat2.size(0) * ehat2.size(1);
        for (ib = 0; ib < nx; ib++) {
            ehat2[ib] = ehat2[ib] + b_X[ib];
        }
        //  k by k
        if (*emlrtBreakCheckR2012bFlagVar != 0) {
            emlrtBreakCheckR2012b((emlrtCTX)sp);
        }
    }
    st.site = &b_emlrtRSI;
    invpd(&st, R0, b_X);
    iv[0] = (*(int32_T (*)[2])ehat2.size())[0];
    iv[1] = (*(int32_T (*)[2])ehat2.size())[1];
    b_ehat2[0] = (*(int32_T (*)[2])b_X.size())[0];
    b_ehat2[1] = (*(int32_T (*)[2])b_X.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_ehat2[0], &b_emlrtECI, (emlrtCTX)sp);
    c_ehat2.set_size(&x_emlrtRTEI, sp, ehat2.size(0), ehat2.size(1));
    nx = ehat2.size(0) * ehat2.size(1);
    for (i = 0; i < nx; i++) {
        c_ehat2[i] = ehat2[i] + b_X[i];
    }
    st.site = &c_emlrtRSI;
    invpd(&st, c_ehat2, ehat2);
    nu1 = nu + static_cast<real_T>(Y.size(0));
    nx = ehat2.size(0) * ehat2.size(1);
    for (i = 0; i < nx; i++) {
        ehat2[i] = ehat2[i] * nu1;
    }
    b_ehat2[0] = ehat2.size(1);
    b_ehat2[1] = ehat2.size(0);
    iv[0] = (*(int32_T (*)[2])ehat2.size())[0];
    iv[1] = (*(int32_T (*)[2])ehat2.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_ehat2[0], &emlrtECI, (emlrtCTX)sp);
    b_X.set_size(&y_emlrtRTEI, sp, ehat2.size(0), ehat2.size(1));
    nx = ehat2.size(1);
    for (i = 0; i < nx; i++) {
        nm1d2 = ehat2.size(0);
        for (ib = 0; ib < nm1d2; ib++) {
            b_X[ib + b_X.size(0) * i] = 0.5 * (ehat2[ib + ehat2.size(0) * i] + ehat2[i + ehat2.size(0) * ib]);
        }
    }
    st.site = &d_emlrtRSI;
    invpd(&st, b_X, ehat2);
    nx = ehat2.size(0) * ehat2.size(1);
    for (i = 0; i < nx; i++) {
        ehat2[i] = ehat2[i] * nu1;
    }
    //  R0 = Omega0_inv/nu; 
    //  Omega0_inv = R0*nu;
    //  Psi0 = Omega0*nu;
    st.site = &e_emlrtRSI;
    //  https://en.wikipedia.org/wiki/Inverse-Wishart_distribution
    //  
    //    [pdf] = IWpdf(X, Psi, v) ;
    // 
    //  X = argument matrix, should be positive definite, p by p
    //  Psi = p x p symmetric, postitive definite "scale" matrix 
    //  v = "precision" parameter = "degrees of freeedom"
    //    With this density definitions,
    //    mean(X) = Psi/(v-p-1), v > p+1
    //    Psi = mean(X)*(v-p-1)
    //    mode(X) = Psi/(v+p+1).
    b_st.site = &wc_emlrtRSI;
    c_st.site = &wc_emlrtRSI;
    b_x = coder::det(&c_st, ehat2);
    if (b_x < 0.0) {
        emlrtErrorWithMessageIdR2018a(&b_st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError", "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
    }
    b_x = muDoubleScalarLog(b_x);
    b_st.site = &xc_emlrtRSI;
    if (Omega_st.size(1) != ehat2.size(1)) {
        emlrtErrorWithMessageIdR2018a(&b_st, &g_emlrtRTEI, "MATLAB:dimagree", "MATLAB:dimagree", 0);
    }
    c_st.site = &hd_emlrtRSI;
    coder::internal::mrdiv(&c_st, ehat2, Omega_st, b_X);
    b_st.site = &xc_emlrtRSI;
    if (b_X.size(0) != b_X.size(1)) {
        emlrtErrorWithMessageIdR2018a(&b_st, &f_emlrtRTEI, "Coder:MATLAB:square", "Coder:MATLAB:square", 0);
    }
    t = 0.0;
    i = b_X.size(0);
    for (nleft = 0; nleft < i; nleft++) {
        t += b_X[nleft + b_X.size(0) * nleft];
    }
    b_st.site = &yc_emlrtRSI;
    c_st.site = &yc_emlrtRSI;
    c_x = coder::det(&c_st, Omega_st);
    if (c_x < 0.0) {
        emlrtErrorWithMessageIdR2018a(&b_st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError", "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
    }
    c_x = muDoubleScalarLog(c_x);
    b_st.site = &ad_emlrtRSI;
    b_st.site = &bd_emlrtRSI;
    a = nu1 / 2.0;
    //  log multivariate gamma function
    //  https://en.wikipedia.org/wiki/Multivariate_gamma_function
    c_st.site = &ue_emlrtRSI;
    kd = a + (1.0 - static_cast<real_T>(Omega_st.size(0))) / 2.0;
    if (muDoubleScalarIsNaN(a + 0.5) || muDoubleScalarIsNaN(kd)) {
        A.set_size(&bb_emlrtRTEI, &c_st, 1, 1);
        A[0] = rtNaN;
    } else if (a + 0.5 < kd) {
        A.set_size(&bb_emlrtRTEI, &c_st, 1, 0);
    } else if ((muDoubleScalarIsInf(a + 0.5) || muDoubleScalarIsInf(kd)) && (a + 0.5 == kd)) {
        A.set_size(&bb_emlrtRTEI, &c_st, 1, 1);
        A[0] = rtNaN;
    } else {
        real_T apnd;
        real_T cdiff;
        real_T ndbl;
        d_st.site = &oc_emlrtRSI;
        ndbl = muDoubleScalarFloor((kd - (a + 0.5)) / -0.5 + 0.5);
        apnd = (a + 0.5) + ndbl * -0.5;
        cdiff = kd - apnd;
        if (muDoubleScalarAbs(cdiff) < 4.4408920985006262E-16 * muDoubleScalarMax(muDoubleScalarAbs(a + 0.5), muDoubleScalarAbs(kd))) {
            ndbl++;
            apnd = kd;
        } else if (cdiff > 0.0) {
            apnd = (a + 0.5) + (ndbl - 1.0) * -0.5;
        } else {
            ndbl++;
        }
        if (ndbl >= 0.0) {
            nx = static_cast<int32_T>(ndbl);
        } else {
            nx = 0;
        }
        e_st.site = &pc_emlrtRSI;
        if (ndbl > 2.147483647E+9) {
            emlrtErrorWithMessageIdR2018a(&e_st, &e_emlrtRTEI, "Coder:MATLAB:pmaxsize", "Coder:MATLAB:pmaxsize", 0);
        }
        A.set_size(&ab_emlrtRTEI, &d_st, 1, nx);
        if (nx > 0) {
            A[0] = a + 0.5;
            if (nx > 1) {
                A[nx - 1] = apnd;
                nm1d2 = (nx - 1) / 2;
                e_st.site = &qc_emlrtRSI;
                for (nleft = 0; nleft <= nm1d2 - 2; nleft++) {
                    kd = (static_cast<real_T>(nleft) + 1.0) * -0.5;
                    A[nleft + 1] = (a + 0.5) + kd;
                    A[(nx - nleft) - 2] = apnd - kd;
                }
                if (nm1d2 << 1 == nx - 1) {
                    A[nm1d2] = ((a + 0.5) + apnd) / 2.0;
                } else {
                    kd = static_cast<real_T>(nm1d2) * -0.5;
                    A[nm1d2] = (a + 0.5) + kd;
                    A[nm1d2 + 1] = apnd - kd;
                }
            }
        }
    }
    c_st.site = &ve_emlrtRSI;
    d_st.site = &ye_emlrtRSI;
    coder::internal::applyScalarFunctionInPlace(&d_st, A);
    c_st.site = &we_emlrtRSI;
    d_st.site = &we_emlrtRSI;
    x.set_size(&cb_emlrtRTEI, &d_st, A.size(1));
    nx = A.size(1);
    for (i = 0; i < nx; i++) {
        x[i] = A[i];
    }
    p = false;
    i = x.size(0);
    for (nleft = 0; nleft < i; nleft++) {
        if (p || (x[nleft] < 0.0)) {
            p = true;
        }
    }
    if (p) {
        emlrtErrorWithMessageIdR2018a(&d_st, &d_emlrtRTEI, "Coder:toolbox:ElFunDomainError", "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
    }
    e_st.site = &bf_emlrtRSI;
    nx = x.size(0);
    f_st.site = &cf_emlrtRSI;
    if ((1 <= x.size(0)) && (x.size(0) > 2147483646)) {
        g_st.site = &i_emlrtRSI;
        coder::check_forloop_overflow_error(&g_st);
    }
    for (nleft = 0; nleft < nx; nleft++) {
        x[nleft] = muDoubleScalarLog(x[nleft]);
    }
    //  gauss function
    d_st.site = &df_emlrtRSI;
    e_st.site = &fb_emlrtRSI;
    f_st.site = &gb_emlrtRSI;
    g_st.site = &hb_emlrtRSI;
    if (x.size(0) == 0) {
        kd = 0.0;
    } else {
        h_st.site = &ib_emlrtRSI;
        i_st.site = &jb_emlrtRSI;
        if (x.size(0) < 4096) {
            j_st.site = &lb_emlrtRSI;
            kd = coder::sumColumnB(&j_st, x, x.size(0));
        } else {
            nx = x.size(0) / 4096;
            nm1d2 = nx << 12;
            nleft = x.size(0) - nm1d2;
            kd = coder::sumColumnB4(x, 1);
            for (ib = 2; ib <= nx; ib++) {
                kd += coder::sumColumnB4(x, ((ib - 1) << 12) + 1);
            }
            if (nleft > 0) {
                j_st.site = &nb_emlrtRSI;
                kd += coder::sumColumnB(&j_st, x, nleft, nm1d2 + 1);
            }
        }
    }
    c_st.site = &xe_emlrtRSI;
    lnpost_pdf = (((0.5 * nu1 * b_x + -0.5 * t) + -0.5 * ((nu1 + static_cast<real_T>(Omega_st.size(0))) + 1.0) * c_x) - nu1 * static_cast<real_T>(Omega_st.size(0)) / 2.0 * 0.69314718055994529) - (static_cast<real_T>(Omega_st.size(0)) * (static_cast<real_T>(Omega_st.size(0)) - 1.0) * 1.1447298858494002 + kd);
    emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
    return lnpost_pdf;
}


// End of code generation (lnpost_Omega.cpp) 
