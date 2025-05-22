//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// applyScalarFunctionInPlace.cpp
//
// Code generation for function 'applyScalarFunctionInPlace'
//

// Include files
#include "applyScalarFunctionInPlace.h"
#include "eml_int_forloop_overflow_check.h"
#include "lnpost_Omega_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo af_emlrtRSI{
    26,                           // lineNo
    "applyScalarFunctionInPlace", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\applyScalarFunctionInPlace.m" // pathName
};

// Function Definitions
namespace coder {
namespace internal {
void applyScalarFunctionInPlace(const emlrtStack *sp,
                                ::coder::array<real_T, 2U> &x)
{
  static const real_T gam[23]{1.0,
                              1.0,
                              2.0,
                              6.0,
                              24.0,
                              120.0,
                              720.0,
                              5040.0,
                              40320.0,
                              362880.0,
                              3.6288E+6,
                              3.99168E+7,
                              4.790016E+8,
                              6.2270208E+9,
                              8.71782912E+10,
                              1.307674368E+12,
                              2.0922789888E+13,
                              3.55687428096E+14,
                              6.402373705728E+15,
                              1.21645100408832E+17,
                              2.43290200817664E+18,
                              5.109094217170944E+19,
                              1.1240007277776077E+21};
  static const real_T p[8]{-1.716185138865495,  24.76565080557592,
                           -379.80425647094563, 629.3311553128184,
                           866.96620279041326,  -31451.272968848367,
                           -36144.413418691176, 66456.143820240541};
  static const real_T q[8]{-30.840230011973897, 315.35062697960416,
                           -1015.1563674902192, -3107.7716715723109,
                           22538.11842098015,   4755.8462775278813,
                           -134659.95986496931, -115132.25967555349};
  static const real_T c[7]{-0.001910444077728,      0.00084171387781295,
                           -0.00059523799130430121, 0.0007936507935003503,
                           -0.0027777777777776816,  0.083333333333333329,
                           0.0057083835261};
  emlrtStack b_st;
  emlrtStack st;
  real_T b_sum;
  real_T fact;
  real_T x_tmp;
  real_T xden;
  real_T xkold;
  real_T yint;
  int32_T i;
  int32_T k;
  int32_T n;
  int32_T ub_loop;
  boolean_T guard1{false};
  boolean_T negateSinpi;
  boolean_T parity;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &af_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  if ((1 <= x.size(1)) && (x.size(1) > 2147483646)) {
    b_st.site = &i_emlrtRSI;
    check_forloop_overflow_error(&b_st);
  }
  ub_loop = x.size(1) - 1;
  emlrtEnterParallelRegion((emlrtCTX)sp,
                           static_cast<boolean_T>(omp_in_parallel()));
#pragma omp parallel for num_threads(emlrtAllocRegionTLSs(                     \
    sp->tls, omp_in_parallel(), omp_get_max_threads(),                         \
    omp_get_num_procs())) private(x_tmp, k, yint, fact, n, parity,             \
                                  negateSinpi, xkold, i, b_sum, xden)          \
    firstprivate(guard1)

  for (k = 0; k <= ub_loop; k++) {
    x_tmp = x[k];
    if ((x_tmp >= 1.0) && (x_tmp <= 23.0) &&
        (x_tmp == muDoubleScalarFloor(x_tmp))) {
      x[k] = gam[static_cast<int32_T>(x_tmp) - 1];
    } else {
      guard1 = false;
      if (x_tmp < 1.0) {
        yint = x[k];
        if (yint == muDoubleScalarFloor(yint)) {
          x[k] = rtInf;
        } else {
          guard1 = true;
        }
      } else {
        guard1 = true;
      }
      if (guard1) {
        if (muDoubleScalarIsNaN(x[k])) {
          x[k] = rtNaN;
        } else {
          yint = x[k];
          if (muDoubleScalarIsInf(yint)) {
            x[k] = rtInf;
          } else {
            fact = 1.0;
            n = -1;
            parity = false;
            if (yint <= 0.0) {
              yint = muDoubleScalarFloor(-yint);
              parity = (yint != muDoubleScalarFloor(-x[k] / 2.0) * 2.0);
              yint = -x[k] - yint;
              if (yint < 0.0) {
                yint = -yint;
                negateSinpi = true;
              } else {
                negateSinpi = false;
              }
              if (yint < 0.25) {
                yint = muDoubleScalarSin(yint * 3.1415926535897931);
              } else {
                yint -= 2.0 * muDoubleScalarFloor(yint / 2.0);
                if (yint < 0.25) {
                  yint = muDoubleScalarSin(yint * 3.1415926535897931);
                } else if (yint < 0.75) {
                  yint = 0.5 - yint;
                  yint = muDoubleScalarCos(yint * 3.1415926535897931);
                } else if (yint < 1.25) {
                  yint = 1.0 - yint;
                  yint = muDoubleScalarSin(yint * 3.1415926535897931);
                } else if (yint < 1.75) {
                  yint -= 1.5;
                  yint = -muDoubleScalarCos(yint * 3.1415926535897931);
                } else {
                  yint -= 2.0;
                  yint = muDoubleScalarSin(yint * 3.1415926535897931);
                }
              }
              if (negateSinpi) {
                yint = -yint;
              }
              fact = -3.1415926535897931 / yint;
              x_tmp = -x[k] + 1.0;
            }
            if (x_tmp < 12.0) {
              xkold = x_tmp;
              if (x_tmp < 1.0) {
                yint = x_tmp;
                x_tmp++;
              } else {
                i = static_cast<int32_T>(muDoubleScalarFloor(x_tmp));
                n = i - 2;
                x_tmp -= static_cast<real_T>(i) - 1.0;
                yint = x_tmp - 1.0;
              }
              b_sum = 0.0 * yint;
              xden = 1.0;
              for (i = 0; i < 8; i++) {
                b_sum = (b_sum + p[i]) * yint;
                xden = xden * yint + q[i];
              }
              yint = b_sum / xden + 1.0;
              if (xkold < x_tmp) {
                yint /= xkold;
              } else if (xkold > x_tmp) {
                for (i = 0; i <= n; i++) {
                  yint *= x_tmp;
                  x_tmp++;
                }
              }
            } else {
              yint = x_tmp * x_tmp;
              b_sum = 0.0057083835261;
              for (i = 0; i < 6; i++) {
                b_sum = b_sum / yint + c[i];
              }
              b_sum = (b_sum / x_tmp - x_tmp) + 0.91893853320467278;
              b_sum += (x_tmp - 0.5) * muDoubleScalarLog(x_tmp);
              yint = muDoubleScalarExp(b_sum);
            }
            if (parity) {
              yint = -yint;
            }
            if (fact != 1.0) {
              yint = fact / yint;
            }
            x[k] = yint;
          }
        }
      }
    }
  }
  emlrtExitParallelRegion((emlrtCTX)sp,
                          static_cast<boolean_T>(omp_in_parallel()));
}

} // namespace internal
} // namespace coder

// End of code generation (applyScalarFunctionInPlace.cpp)
