//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_Gen_Mu_api.cpp
//
// Code generation for function '_coder_Gen_Mu_api'
//

// Include files
#include "_coder_Gen_Mu_api.h"
#include "Gen_Mu.h"
#include "Gen_Mu_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Declarations
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 2U> &ret);

static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 3U> &ret);

static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 1U> &ret);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *Mu,
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

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *beta,
                             const char_T *identifier,
                             coder::array<real_T, 1U> &y);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 1U> &y);

static void emlrt_marshallOut(const coder::array<real_T, 2U> &u,
                              const mxArray *y);

static const mxArray *emlrt_marshallOut(const coder::array<real_T, 2U> &u);

// Function Definitions
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
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

static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
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

static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
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

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *Mu,
                             const char_T *identifier,
                             coder::array<real_T, 2U> &y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  emlrt_marshallIn(sp, emlrtAlias(Mu), &thisId, y);
  emlrtDestroyArray(&Mu);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 2U> &y)
{
  b_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
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
  b_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *beta,
                             const char_T *identifier,
                             coder::array<real_T, 1U> &y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  emlrt_marshallIn(sp, emlrtAlias(beta), &thisId, y);
  emlrtDestroyArray(&beta);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 1U> &y)
{
  b_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static void emlrt_marshallOut(const coder::array<real_T, 2U> &u,
                              const mxArray *y)
{
  emlrtMxSetData((mxArray *)y, &(((coder::array<real_T, 2U> *)&u)->data())[0]);
  emlrtSetDimensions((mxArray *)y, ((coder::array<real_T, 2U> *)&u)->size(), 2);
}

static const mxArray *emlrt_marshallOut(const coder::array<real_T, 2U> &u)
{
  static const int32_T iv[2]{0, 0};
  const mxArray *m;
  const mxArray *y;
  y = nullptr;
  m = emlrtCreateNumericArray(2, (const void *)&iv[0], mxDOUBLE_CLASS, mxREAL);
  emlrtMxSetData((mxArray *)m, &(((coder::array<real_T, 2U> *)&u)->data())[0]);
  emlrtSetDimensions((mxArray *)m, ((coder::array<real_T, 2U> *)&u)->size(), 2);
  emlrtAssign(&y, m);
  return y;
}

void Gen_Mu_api(const mxArray *const prhs[8], int32_T nlhs,
                const mxArray *plhs[2])
{
  coder::array<real_T, 3U> YLm;
  coder::array<real_T, 2U> G_ttm;
  coder::array<real_T, 2U> Mu;
  coder::array<real_T, 2U> Omega;
  coder::array<real_T, 2U> Phi;
  coder::array<real_T, 2U> Y0;
  coder::array<real_T, 1U> b_gamma;
  coder::array<real_T, 1U> beta;
  coder::array<real_T, 1U> diag_Sigma;
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  const mxArray *prhs_copy_idx_0;
  st.tls = emlrtRootTLSGlobal;
  emlrtHeapReferenceStackEnterFcnR2012b(&st);
  prhs_copy_idx_0 = emlrtProtectR2012b(prhs[0], 0, true, -1);
  // Marshall function inputs
  Mu.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs_copy_idx_0), "Mu", Mu);
  Y0.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[1]), "Y0", Y0);
  YLm.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[2]), "YLm", YLm);
  beta.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[3]), "beta", beta);
  Phi.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[4]), "Phi", Phi);
  Omega.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[5]), "Omega", Omega);
  diag_Sigma.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[6]), "diag_Sigma", diag_Sigma);
  b_gamma.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[7]), "gamma", b_gamma);
  // Invoke the target function
  Gen_Mu(&st, Mu, Y0, YLm, beta, Phi, Omega, diag_Sigma, b_gamma, G_ttm);
  // Marshall function outputs
  Mu.no_free();
  emlrt_marshallOut(Mu, prhs_copy_idx_0);
  plhs[0] = prhs_copy_idx_0;
  if (nlhs > 1) {
    G_ttm.no_free();
    plhs[1] = emlrt_marshallOut(G_ttm);
  }
  emlrtHeapReferenceStackLeaveFcnR2012b(&st);
}

// End of code generation (_coder_Gen_Mu_api.cpp)
