//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// _coder_Gen_Um_api.cpp
//
// Code generation for function '_coder_Gen_Um_api'
//

// Include files
#include "_coder_Gen_Um_api.h"
#include "Gen_Um.h"
#include "Gen_Um_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Declarations
static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 2U> &ret);

static void b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
                               const emlrtMsgIdentifier *msgId,
                               coder::array<real_T, 1U> &ret);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *Ym,
                             const char_T *identifier,
                             coder::array<real_T, 2U> &y);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 2U> &y);

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *rho,
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

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *Ym,
                             const char_T *identifier,
                             coder::array<real_T, 2U> &y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  emlrt_marshallIn(sp, emlrtAlias(Ym), &thisId, y);
  emlrtDestroyArray(&Ym);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *u,
                             const emlrtMsgIdentifier *parentId,
                             coder::array<real_T, 2U> &y)
{
  b_emlrt_marshallIn(sp, emlrtAlias(u), parentId, y);
  emlrtDestroyArray(&u);
}

static void emlrt_marshallIn(const emlrtStack *sp, const mxArray *rho,
                             const char_T *identifier,
                             coder::array<real_T, 1U> &y)
{
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = const_cast<const char_T *>(identifier);
  thisId.fParent = nullptr;
  thisId.bParentIsCell = false;
  emlrt_marshallIn(sp, emlrtAlias(rho), &thisId, y);
  emlrtDestroyArray(&rho);
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

void Gen_Um_api(const mxArray *const prhs[6], int32_T nlhs,
                const mxArray *plhs[2])
{
  coder::array<real_T, 2U> Omega;
  coder::array<real_T, 2U> Phi;
  coder::array<real_T, 2U> Sigma;
  coder::array<real_T, 2U> Um;
  coder::array<real_T, 2U> Uttm;
  coder::array<real_T, 2U> Ym;
  coder::array<real_T, 1U> rho;
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  const mxArray *prhs_copy_idx_1;
  st.tls = emlrtRootTLSGlobal;
  emlrtHeapReferenceStackEnterFcnR2012b(&st);
  prhs_copy_idx_1 = emlrtProtectR2012b(prhs[1], 1, true, -1);
  // Marshall function inputs
  Ym.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[0]), "Ym", Ym);
  Um.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs_copy_idx_1), "Um", Um);
  rho.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[2]), "rho", rho);
  Phi.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[3]), "Phi", Phi);
  Sigma.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[4]), "Sigma", Sigma);
  Omega.no_free();
  emlrt_marshallIn(&st, emlrtAlias(prhs[5]), "Omega", Omega);
  // Invoke the target function
  Gen_Um(&st, Ym, Um, rho, Phi, Sigma, Omega, Uttm);
  // Marshall function outputs
  Um.no_free();
  emlrt_marshallOut(Um, prhs_copy_idx_1);
  plhs[0] = prhs_copy_idx_1;
  if (nlhs > 1) {
    Uttm.no_free();
    plhs[1] = emlrt_marshallOut(Uttm);
  }
  emlrtHeapReferenceStackLeaveFcnR2012b(&st);
}

// End of code generation (_coder_Gen_Um_api.cpp)
