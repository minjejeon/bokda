//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// invpd.cpp
//
// Code generation for function 'invpd'
//

// Include files
#include "invpd.h"
#include "cholmod.h"
#include "eml_int_forloop_overflow_check.h"
#include "lnpost_Omega_data.h"
#include "rt_nonfinite.h"
#include "blas.h"
#include "coder_array.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo k_emlrtRSI{
    23,      // lineNo
    "invpd", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pathName
};

static emlrtRSInfo l_emlrtRSI{
    22,      // lineNo
    "invpd", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pathName
};

static emlrtRSInfo m_emlrtRSI{
    17,      // lineNo
    "invpd", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pathName
};

static emlrtRSInfo tc_emlrtRSI{
    14,        // lineNo
    "invuptr", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pathName
};

static emlrtRSInfo uc_emlrtRSI{
    23,        // lineNo
    "invuptr", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pathName
};

static emlrtRSInfo
    vc_emlrtRSI{
        42,    // lineNo
        "eye", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\eye.m" // pathName
    };

static emlrtMCInfo emlrtMCI{
    11,      // lineNo
    10,      // colNo
    "invpd", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pName
};

static emlrtMCInfo c_emlrtMCI{
    11,        // lineNo
    10,        // colNo
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pName
};

static emlrtMCInfo d_emlrtMCI{
    17,        // lineNo
    13,        // colNo
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pName
};

static emlrtBCInfo c_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    24,        // lineNo
    6,         // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo d_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    24,        // lineNo
    20,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo e_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    20,        // lineNo
    4,         // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo f_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    20,        // lineNo
    15,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo g_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    16,        // lineNo
    16,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo h_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    23,        // lineNo
    37,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo i_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    23,        // lineNo
    35,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo j_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    23,        // lineNo
    31,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo k_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    23,        // lineNo
    26,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo l_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    23,        // lineNo
    22,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtBCInfo m_emlrtBCI{
    -1,        // iFirst
    -1,        // iLast
    23,        // lineNo
    20,        // colNo
    "T",       // aName
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m", // pName
    0 // checkKind
};

static emlrtRTEInfo i_emlrtRTEI{
    21,        // lineNo
    12,        // colNo
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pName
};

static emlrtRTEInfo j_emlrtRTEI{
    15,        // lineNo
    10,        // colNo
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pName
};

static emlrtECInfo e_emlrtECI{
    2,       // nDims
    25,      // lineNo
    13,      // colNo
    "invpd", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pName
};

static emlrtRTEInfo fb_emlrtRTEI{
    23,        // lineNo
    18,        // colNo
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pName
};

static emlrtRTEInfo gb_emlrtRTEI{
    23,        // lineNo
    29,        // colNo
    "invuptr", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pName
};

static emlrtRTEInfo hb_emlrtRTEI{
    23,      // lineNo
    10,      // colNo
    "invpd", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pName
};

static emlrtRTEInfo ib_emlrtRTEI{
    25,      // lineNo
    12,      // colNo
    "invpd", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pName
};

static emlrtRTEInfo jb_emlrtRTEI{
    25,      // lineNo
    5,       // colNo
    "invpd", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pName
};

static emlrtRTEInfo kb_emlrtRTEI{
    13,      // lineNo
    13,      // colNo
    "invpd", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pName
};

static emlrtRSInfo ef_emlrtRSI{
    11,      // lineNo
    "invpd", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invpd.m" // pathName
};

static emlrtRSInfo ff_emlrtRSI{
    11,        // lineNo
    "invuptr", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pathName
};

static emlrtRSInfo gf_emlrtRSI{
    17,        // lineNo
    "invuptr", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\invuptr.m" // pathName
};

// Function Declarations
static void disp(const emlrtStack *sp, const mxArray *b, emlrtMCInfo *location);

// Function Definitions
static void disp(const emlrtStack *sp, const mxArray *b, emlrtMCInfo *location)
{
  const mxArray *pArray;
  pArray = b;
  emlrtCallMATLABR2012b((emlrtCTX)sp, 0, nullptr, 1, &pArray,
                        (const char_T *)"disp", true, location);
}

void invpd(const emlrtStack *sp, const coder::array<real_T, 2U> &A,
           coder::array<real_T, 2U> &Ainv)
{
  static const int32_T iv[2]{1, 23};
  static const int32_T iv1[2]{1, 23};
  static const int32_T iv3[2]{1, 20};
  static const char_T b_u[23]{'m', 'a', 't', 'r', 'i', 'x', ' ', 'T',
                              ' ', ' ', 'i', 's', ' ', 'n', 'o', 't',
                              ' ', 's', 'q', 'u', 'a', 'r', 'e'};
  static const char_T u[23]{'m', 'a', 't', 'r', 'i', 'x', ' ', 'A',
                            ' ', ' ', 'i', 's', ' ', 'n', 'o', 't',
                            ' ', 's', 'q', 'u', 'a', 'r', 'e'};
  static const char_T c_u[20]{'m', 'a', 't', 'r', 'i', 'x', ' ', 'T', ' ', 'i',
                              's', ' ', 's', 'i', 'n', 'g', 'u', 'l', 'a', 'r'};
  ptrdiff_t incx_t;
  ptrdiff_t incy_t;
  ptrdiff_t lda_t;
  ptrdiff_t ldb_t;
  ptrdiff_t ldc_t;
  ptrdiff_t n_t;
  coder::array<real_T, 2U> Hinv;
  coder::array<real_T, 2U> a;
  coder::array<real_T, 1U> b;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack st;
  const mxArray *b_y;
  const mxArray *c_y;
  const mxArray *m;
  const mxArray *y;
  real_T beta1;
  real_T sum;
  int32_T b_Ainv[2];
  int32_T iv2[2];
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
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  // 'invpd' Inverse of a symmetric positive definite matrix using Cholesky
  // factorization Ainv = invpd(A) computes the inverse of a symmetric positive
  // definite matrix A using its Cholesky factor H.
  // inv(A) = inv(H)inv(H)'.
  // input  : Matrix A
  // output : Ainv, err (=1 if error, and 0 if no error)
  if (A.size(0) != A.size(1)) {
    y = nullptr;
    m = emlrtCreateCharArray(2, &iv[0]);
    emlrtInitCharArrayR2013a((emlrtCTX)sp, 23, m, &u[0]);
    emlrtAssign(&y, m);
    st.site = &ef_emlrtRSI;
    disp(&st, y, &emlrtMCI);
    Ainv.set_size(&kb_emlrtRTEI, sp, 0, 0);
  } else {
    int32_T b_loop_ub;
    int32_T i;
    int32_T i1;
    int32_T loop_ub;
    int32_T n;
    st.site = &m_emlrtRSI;
    cholmod(&st, A, Hinv, &sum);
    st.site = &l_emlrtRSI;
    // INVUPTR Inverse of an upper triangular matrix
    // T = invuptr(T) computes the inverse of a nonsingular upper triangular
    // matrix T.  The output matrix T contains the inverse of T.
    // This program implements Algorithm 4.2.2 of the book.
    // Input  : Matrix T
    // output : Matrix T
    n = Hinv.size(1);
    if (Hinv.size(0) != n) {
      b_y = nullptr;
      m = emlrtCreateCharArray(2, &iv1[0]);
      emlrtInitCharArrayR2013a(&st, 23, m, &b_u[0]);
      emlrtAssign(&b_y, m);
      b_st.site = &ff_emlrtRSI;
      disp(&b_st, b_y, &c_emlrtMCI);
    } else {
      int32_T k;
      boolean_T exitg1;
      b_st.site = &tc_emlrtRSI;
      c_st.site = &vc_emlrtRSI;
      if (n > 0) {
        c_st.site = &lc_emlrtRSI;
        if (n > 2147483646) {
          d_st.site = &i_emlrtRSI;
          coder::check_forloop_overflow_error(&d_st);
        }
      }
      i = static_cast<int32_T>(((-1.0 - static_cast<real_T>(n)) + 1.0) / -1.0);
      emlrtForLoopVectorCheckR2021a(static_cast<real_T>(n), -1.0, 1.0,
                                    mxDOUBLE_CLASS, i, &j_emlrtRTEI, &st);
      k = 0;
      exitg1 = false;
      while ((!exitg1) && (k <= i - 1)) {
        int32_T b_k;
        b_k = n - k;
        if ((b_k < 1) || (b_k > Hinv.size(1))) {
          emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &g_emlrtBCI, &st);
        }
        if (b_k > Hinv.size(0)) {
          emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(0), &g_emlrtBCI, &st);
        }
        sum = Hinv[(b_k + Hinv.size(0) * (b_k - 1)) - 1];
        if (sum == 0.0) {
          c_y = nullptr;
          m = emlrtCreateCharArray(2, &iv3[0]);
          emlrtInitCharArrayR2013a(&st, 20, m, &c_u[0]);
          emlrtAssign(&c_y, m);
          b_st.site = &gf_emlrtRSI;
          disp(&b_st, c_y, &d_emlrtMCI);
          exitg1 = true;
        } else {
          if (b_k > Hinv.size(1)) {
            emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &e_emlrtBCI,
                                          &st);
          }
          if (b_k > Hinv.size(0)) {
            emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(0), &e_emlrtBCI,
                                          &st);
          }
          if (b_k > Hinv.size(1)) {
            emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &f_emlrtBCI,
                                          &st);
          }
          if (b_k > Hinv.size(0)) {
            emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(0), &f_emlrtBCI,
                                          &st);
          }
          Hinv[(b_k + Hinv.size(0) * (b_k - 1)) - 1] = 1.0 / sum;
          i1 = static_cast<int32_T>(
              ((-1.0 - (static_cast<real_T>(b_k) - 1.0)) + 1.0) / -1.0);
          emlrtForLoopVectorCheckR2021a(static_cast<real_T>(b_k) - 1.0, -1.0,
                                        1.0, mxDOUBLE_CLASS, i1, &i_emlrtRTEI,
                                        &st);
          for (int32_T b_i{0}; b_i < i1; b_i++) {
            int32_T c_i;
            int32_T i2;
            int32_T i3;
            int32_T i4;
            int32_T i5;
            c_i = b_k - b_i;
            if (c_i > b_k) {
              i2 = -1;
              i3 = -1;
              i4 = -1;
              i5 = -1;
            } else {
              if ((c_i < 1) || (c_i > Hinv.size(1))) {
                emlrtDynamicBoundsCheckR2012b(c_i, 1, Hinv.size(1), &l_emlrtBCI,
                                              &st);
              }
              i2 = c_i - 2;
              if (b_k > Hinv.size(1)) {
                emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &k_emlrtBCI,
                                              &st);
              }
              i3 = b_k - 1;
              if (c_i > Hinv.size(0)) {
                emlrtDynamicBoundsCheckR2012b(c_i, 1, Hinv.size(0), &j_emlrtBCI,
                                              &st);
              }
              i4 = c_i - 2;
              if (b_k > Hinv.size(0)) {
                emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(0), &i_emlrtBCI,
                                              &st);
              }
              i5 = b_k - 1;
            }
            b_st.site = &uc_emlrtRSI;
            if ((c_i - 1 < 1) || (c_i - 1 > Hinv.size(0))) {
              emlrtDynamicBoundsCheckR2012b(c_i - 1, 1, Hinv.size(0),
                                            &m_emlrtBCI, &b_st);
            }
            loop_ub = i3 - i2;
            a.set_size(&fb_emlrtRTEI, &b_st, 1, loop_ub);
            for (b_loop_ub = 0; b_loop_ub < loop_ub; b_loop_ub++) {
              a[b_loop_ub] =
                  Hinv[(c_i + Hinv.size(0) * ((i2 + b_loop_ub) + 1)) - 2];
            }
            if (b_k > Hinv.size(1)) {
              emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &h_emlrtBCI,
                                            &b_st);
            }
            b_loop_ub = i5 - i4;
            b.set_size(&gb_emlrtRTEI, &b_st, b_loop_ub);
            for (i5 = 0; i5 < b_loop_ub; i5++) {
              b[i5] = Hinv[((i4 + i5) + Hinv.size(0) * (b_k - 1)) + 1];
            }
            c_st.site = &g_emlrtRSI;
            if (loop_ub != b_loop_ub) {
              if ((loop_ub == 1) || (b_loop_ub == 1)) {
                emlrtErrorWithMessageIdR2018a(
                    &c_st, &emlrtRTEI,
                    "Coder:toolbox:mtimes_noDynamicScalarExpansion",
                    "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
              } else {
                emlrtErrorWithMessageIdR2018a(&c_st, &b_emlrtRTEI,
                                              "MATLAB:innerdim",
                                              "MATLAB:innerdim", 0);
              }
            }
            if (loop_ub < 1) {
              sum = 0.0;
            } else {
              n_t = (ptrdiff_t)(i3 - i2);
              incx_t = (ptrdiff_t)1;
              incy_t = (ptrdiff_t)1;
              sum = ddot(&n_t, &a[0], &incx_t, &(b.data())[0], &incy_t);
            }
            if (b_k > Hinv.size(1)) {
              emlrtDynamicBoundsCheckR2012b(b_k, 1, Hinv.size(1), &c_emlrtBCI,
                                            &st);
            }
            if ((c_i - 1 < 1) || (c_i - 1 > Hinv.size(0))) {
              emlrtDynamicBoundsCheckR2012b(c_i - 1, 1, Hinv.size(0),
                                            &c_emlrtBCI, &st);
            }
            if ((c_i - 1 < 1) || (c_i - 1 > Hinv.size(1))) {
              emlrtDynamicBoundsCheckR2012b(c_i - 1, 1, Hinv.size(1),
                                            &d_emlrtBCI, &st);
            }
            if ((c_i - 1 < 1) || (c_i - 1 > Hinv.size(0))) {
              emlrtDynamicBoundsCheckR2012b(c_i - 1, 1, Hinv.size(0),
                                            &d_emlrtBCI, &st);
            }
            Hinv[(c_i + Hinv.size(0) * (b_k - 1)) - 2] =
                -sum / Hinv[(c_i + Hinv.size(0) * (c_i - 2)) - 2];
            if (*emlrtBreakCheckR2012bFlagVar != 0) {
              emlrtBreakCheckR2012b(&st);
            }
          }
          k++;
          if (*emlrtBreakCheckR2012bFlagVar != 0) {
            emlrtBreakCheckR2012b(&st);
          }
        }
      }
    }
    st.site = &k_emlrtRSI;
    b_st.site = &g_emlrtRSI;
    b_st.site = &f_emlrtRSI;
    if ((Hinv.size(0) == 0) || (Hinv.size(1) == 0)) {
      Ainv.set_size(&hb_emlrtRTEI, &b_st, Hinv.size(0), Hinv.size(0));
      loop_ub = Hinv.size(0) * Hinv.size(0);
      for (i = 0; i < loop_ub; i++) {
        Ainv[i] = 0.0;
      }
    } else {
      c_st.site = &h_emlrtRSI;
      d_st.site = &j_emlrtRSI;
      TRANSB1 = 'T';
      TRANSA1 = 'N';
      sum = 1.0;
      beta1 = 0.0;
      incx_t = (ptrdiff_t)Hinv.size(0);
      n_t = (ptrdiff_t)Hinv.size(0);
      incy_t = (ptrdiff_t)Hinv.size(1);
      lda_t = (ptrdiff_t)Hinv.size(0);
      ldb_t = (ptrdiff_t)Hinv.size(0);
      ldc_t = (ptrdiff_t)Hinv.size(0);
      Ainv.set_size(&db_emlrtRTEI, &d_st, Hinv.size(0), Hinv.size(0));
      dgemm(&TRANSA1, &TRANSB1, &incx_t, &n_t, &incy_t, &sum, &(Hinv.data())[0],
            &lda_t, &(Hinv.data())[0], &ldb_t, &beta1, &(Ainv.data())[0],
            &ldc_t);
    }
    b_Ainv[0] = Ainv.size(1);
    b_Ainv[1] = Ainv.size(0);
    iv2[0] = (*(int32_T(*)[2])Ainv.size())[0];
    iv2[1] = (*(int32_T(*)[2])Ainv.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv2[0], &b_Ainv[0], &e_emlrtECI, (emlrtCTX)sp);
    Hinv.set_size(&ib_emlrtRTEI, sp, Ainv.size(0), Ainv.size(1));
    loop_ub = Ainv.size(1);
    for (i = 0; i < loop_ub; i++) {
      b_loop_ub = Ainv.size(0);
      for (i1 = 0; i1 < b_loop_ub; i1++) {
        Hinv[i1 + Hinv.size(0) * i] =
            (Ainv[i1 + Ainv.size(0) * i] + Ainv[i + Ainv.size(0) * i1]) / 2.0;
      }
    }
    Ainv.set_size(&jb_emlrtRTEI, sp, Hinv.size(0), Hinv.size(1));
    loop_ub = Hinv.size(0) * Hinv.size(1);
    for (i = 0; i < loop_ub; i++) {
      Ainv[i] = Hinv[i];
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (invpd.cpp)
