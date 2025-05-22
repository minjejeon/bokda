//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_lnpost_beta_api.cpp
//
// Code generation for function '_coder_lnpost_beta_api'
//

// Include files
#include "_coder_lnpost_beta_api.h"
#include "lnpost_beta.h"
#include "lnpost_beta_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Declarations
static void b_emlrt_marshallIn(const emlrtStack *sp,
                               const mxArray *NonZeroRestriction,
                               const char_T *identifier,
                               coder::array<boolean_T, 1U> &y);

static void c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                               const emlrtMsgIdentifier *parentId,
                               coder::array<boolean_T, 1U> &y);

static void d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 2U> &ret);

static void d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 3U> &ret);

static void d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 1U> &ret);

static real_T d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                 const emlrtMsgIdentifier *msgId);

static void e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<boolean_T, 1U> &ret);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *Y0,
                             const char_T *identifier,
                             coder::array<real_T, 2U> &y);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 2U> &y);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *YLm,
                             const char_T *identifier,
                             coder::array<real_T, 3U> &y);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 3U> &y);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *beta_st,
                             const char_T *identifier,
                             coder::array<real_T, 1U> &y);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 1U> &y);

static real_T emlrt_marshallIn(const emlrtStack *sp, const mxArray *p,
                               const char_T *identifier);

static real_T emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                               const emlrtMsgIdentifier *parentId);

static const mxArray *emlrt_marshallOut(const real_T u);

// Function Definitions
static void b_emlrt_marshallIn(const emlrtStack *sp,
                               const mxArray *NonZeroRestriction,
                               const char_T *identifier,
                               coder::array<boolean_T, 1U> &y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  c_emlrt_marshallIn(sp, emlrtAlias(NonZeroRestriction), &thisId, y);
  emlrtDestroyArray(&NonZeroRestriction);
}

static void c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                               const emlrtMsgIdentifier *parentId,
                               coder::array<boolean_T, 1U> &y)
{
  e_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static void d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 2U> &ret)
{
  static const int32_T dims[2]{-1, -1};
  int32_T iv[2];
  const boolean_T bv[2]{true, true};
  emlrtCheckVsBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                            false, 2U, (void *)&dims[0], &bv[0], &iv[0]);
  ret.prealloc(iv[0] * iv[1]);
  ret.set_size((emlrtRTEInfo *)nullptr, sp, iv[0], iv[1]);
  ret.set((real_T *)emlrtMxGetData(src), ret.size(0), ret.size(1));
  emlrtDestroyArray(&src);
}

static void d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 3U> &ret)
{
  static const int32_T dims[3]{-1, -1, -1};
  int32_T iv[3];
  const boolean_T bv[3]{true, true, true};
  emlrtCheckVsBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                            false, 3U, (void *)&dims[0], &bv[0], &iv[0]);
  ret.prealloc(iv[0] * iv[1] * iv[2]);
  ret.set_size((emlrtRTEInfo *)nullptr, sp, iv[0], iv[1], iv[2]);
  ret.set((real_T *)emlrtMxGetData(src), ret.size(0), ret.size(1), ret.size(2));
  emlrtDestroyArray(&src);
}

static void d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 1U> &ret)
{
  static const int32_T dims{-1};
  int32_T i;
  const boolean_T b{true};
  emlrtCheckVsBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                            false, 1U, (void *)&dims, &b, &i);
  ret.prealloc(i);
  ret.set_size((emlrtRTEInfo *)nullptr, sp, i);
  ret.set((real_T *)emlrtMxGetData(src), ret.size(0));
  emlrtDestroyArray(&src);
}

static real_T d_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                                 const emlrtMsgIdentifier *msgId)
{
  static const int32_T dims{0};
  real_T ret;
  emlrtCheckBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"double",
                          false, 0U, (void *)&dims);
  ret = *(real_T *)emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

static void e_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<boolean_T, 1U> &ret)
{
  static const int32_T dims{-1};
  int32_T i;
  const boolean_T b{true};
  emlrtCheckVsBuiltInR2012b((emlrtCTX)sp, msgId, src, (const char_T *)"logical",
                            false, 1U, (void *)&dims, &b, &i);
  ret.prealloc(i);
  ret.set_size((emlrtRTEInfo *)nullptr, sp, i);
  ret.set((boolean_T *)emlrtMxGetData(src), ret.size(0));
  emlrtDestroyArray(&src);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *Y0,
                             const char_T *identifier,
                             coder::array<real_T, 2U> &y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  emlrt_marshallIn(sp, emlrtAlias(Y0), &thisId, y);
  emlrtDestroyArray(&Y0);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 2U> &y)
{
  d_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *YLm,
                             const char_T *identifier,
                             coder::array<real_T, 3U> &y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  emlrt_marshallIn(sp, emlrtAlias(YLm), &thisId, y);
  emlrtDestroyArray(&YLm);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 3U> &y)
{
  d_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *beta_st,
                             const char_T *identifier,
                             coder::array<real_T, 1U> &y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  emlrt_marshallIn(sp, emlrtAlias(beta_st), &thisId, y);
  emlrtDestroyArray(&beta_st);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 1U> &y)
{
  d_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static real_T emlrt_marshallIn(const emlrtStack *sp, const mxArray *p,
                               const char_T *identifier)
{
  emlrtMsgIdentifier thisId;
  real_T y;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  y = emlrt_marshallIn(sp, emlrtAlias(p), &thisId);
  emlrtDestroyArray(&p);
  return y;
}

static real_T emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                               const emlrtMsgIdentifier *parentId)
{
  real_T y;
  y = d_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}

static const mxArray *emlrt_marshallOut(const real_T u)
{
  const mxArray *m;
  const mxArray *y;
  y = nullptr;
  m = emlrtCreateDoubleScalar(u);
  emlrtAssign(&y, m);
  return y;
}

void lnpost_beta_api(const mxArray *const prhs[8], const mxArray **plhs)
{
  coder::array<real_T, 3U> YLm;
  coder::array<real_T, 2U> Omega_inv;
  coder::array<real_T, 2U> Y0;
  coder::array<real_T, 2U> var_;
  coder::array<real_T, 1U> b_;
  coder::array<real_T, 1U> beta_st;
  coder::array<boolean_T, 1U> NonZeroRestriction;
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  const mxArray *prhs_copy_idx_7;
  real_T p;
  st.tls = emlrtRootTLSGlobal;
  emlrtHeapReferenceStackEnterFcnR2012b(&st);
  prhs_copy_idx_7 = emlrtProtectR2012b(prhs[7], 7, false, -1);
  // Marshall function inputs
  Y0.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[0]), "Y0", Y0);
  YLm.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[1]), "YLm", YLm);
  beta_st.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[2]), "beta_st", beta_st);
  p = emlrt_marshallIn(&st, emlrtAliasP(prhs[3]), "p");
  b_.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[4]), "b_", b_);
  var_.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[5]), "var_", var_);
  Omega_inv.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[6]), "Omega_inv", Omega_inv);
  NonZeroRestriction.no_free();
  b_emlrt_marshallIn(&st, emlrtAlias(prhs_copy_idx_7), "NonZeroRestriction",
                     NonZeroRestriction);
  // Invoke the target function
  p = lnpost_beta(&st, Y0, YLm, beta_st, p, b_, var_, Omega_inv,
                  NonZeroRestriction);
  // Marshall function outputs
  *plhs = emlrt_marshallOut(p);
  emlrtHeapReferenceStackLeaveFcnR2012b(&st);
}

// End of code generation (_coder_lnpost_beta_api.cpp)
