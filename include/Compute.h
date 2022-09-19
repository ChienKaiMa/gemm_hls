/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License.

#pragma once

#include "MatrixMultiplication.h"

void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn, const unsigned locationN,
                       const unsigned size_n, const unsigned size_k,
                       const unsigned size_m);

// TODO
// xstream, ystream's pack should be 1?

void DiscretizeX(Stream<ComputePackN_t> &xIn,
                 Stream<ComputePackN_t> &xBoolOut);

void UpdateY1(Stream<ComputePackN_t> &xIn,
              Stream<ComputePackN_t> &xOut,
              Stream<ComputePackN_t> &yIn,
              Stream<ComputePackN_t> &yOut,
              Data_t da);

void UpdateY2(Stream<ComputePackN_t> &jxIn,
              Stream<ComputePackN_t> &yIn,
              Stream<ComputePackN_t> &yOut,
              Data_t c0,
              Data_t dt);

void UpdateX(Stream<ComputePackN_t> &xIn,
             Stream<ComputePackN_t> &xOut,
             Stream<ComputePackN_t> &yIn,
             Stream<ComputePackN_t> &yOut,
             Data_t dt);

void Bound(Stream<ComputePackN_t> &xIn,
           Stream<ComputePackN_t> &xOut,
           Stream<ComputePackN_t> &yIn,
           Stream<ComputePackN_t> &yOut);
