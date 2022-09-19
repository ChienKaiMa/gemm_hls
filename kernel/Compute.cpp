/// @author    Johannes de Fine Licht (definelicht@inf.ethz.ch)
/// @copyright This software is copyrighted under the BSD 3-Clause License. 

#include "hlslib/xilinx/Stream.h"
#include "hlslib/xilinx/Simulation.h"
#include "hlslib/xilinx/Utility.h"
#include "MatrixMultiplication.h"
#include "Memory.h"
#include <cassert>

void ProcessingElement(Stream<ComputePackN_t> &aIn,
                       Stream<ComputePackN_t> &aOut,
                       Stream<ComputePackM_t> &bIn,
                       Stream<ComputePackM_t> &bOut,
                       Stream<ComputePackM_t> &cOut,
                       Stream<ComputePackM_t> &cIn, const unsigned locationN,
                       const unsigned size_n, const unsigned size_k,
                       const unsigned size_m) {
  // A is double-buffered, such that new values can be read while the 
  // previous outer product is being computed. This is required to achieve
  // a perfect pipeline across the K-dimension, which is necessary for
  // many processing elements (kInnerTileSizeN).
  ComputePackN_t aBuffer[2 * kInnerTilesN];

  // This is where we spend all our T^2 fast memory
  ComputePackM_t cBuffer[kInnerTilesN * kInnerTilesM][kComputeTileSizeN];
  #pragma HLS ARRAY_PARTITION variable=cBuffer complete dim=2

  // Populate the buffer for the first outer product 
InitializeABuffer_Inner:
  for (unsigned n2 = 0; n2 < kInnerTilesN; ++n2) {
    if (locationN < kComputeTilesN - 1) {
      // All but the last processing element 
    InitializeABuffer_Outer:
      for (unsigned n1 = 0; n1 < kComputeTilesN - locationN; ++n1) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_FLATTEN
        const auto read = aIn.Pop();
        if (n1 == 0) {
          aBuffer[n2] = read;
        } else {
          aOut.Push(read);
        }
      }
    } else {
      // Last processing element gets a special case, because Vivado HLS
      // refuses to flatten and pipeline loops with trip count 1
      #pragma HLS PIPELINE II=1
      aBuffer[n2] = aIn.Pop();
    }
  }

OuterTile_N:
  for (unsigned n0 = 0; n0 < OuterTilesN(size_n); ++n0) {
#pragma HLS loop_tripcount max=32
  OuterTile_M:
    for (unsigned m0 = 0; m0 < OuterTilesM(size_m); ++m0) {
#pragma HLS loop_tripcount max=1024

      // We do not tile K further, but loop over the entire outer tile here
    Collapse_K:
      for (unsigned k = 0; k < size_k; ++k) {
        // Begin outer tile ---------------------------------------------------
#pragma HLS loop_tripcount max=1024
      Pipeline_N:
        for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {
#pragma HLS loop_tripcount max=8

        Pipeline_M:
          for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {
#pragma HLS loop_tripcount max=256

            // Begin compute tile ---------------------------------------------
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_FLATTEN

            static_assert(kInnerTilesM >= kInnerTilesN,
                          "Double buffering does not work if there are more "
                          "N-tiles than M-tiles");

            // Double-buffering scheme. This hijacks the m1-index to perform
            // the buffering and forwarding of values for the following outer
            // product, required to flatten the K-loop.
            if ((n0 < OuterTilesN(size_n) - 1 || m0 < OuterTilesM(size_m) - 1 ||
                 k < size_k - 1) &&
                m1 >= locationN            // Start at own index.
                && m1 < kComputeTilesN) {  // Number of PEs in front.
              const auto read = aIn.Pop();
              if (m1 == locationN) {
                // Double buffering
                aBuffer[n1 + (k % 2 == 0 ? kInnerTilesN : 0)] = read;
                #pragma HLS DEPENDENCE variable=aBuffer false
              } else {
                // Without this check, Vivado HLS thinks aOut can be written
                // from the last processing element and fails dataflow
                // checking.
                if (locationN < kComputeTilesN - 1) {
                  aOut.Push(read);
                }
              }
            }

            // Double buffering, read from the opposite end of where the buffer
            // is being written
            const auto aVal = aBuffer[n1 + (k % 2 == 0 ? 0 : kInnerTilesN)];
            #pragma HLS DEPENDENCE variable=aBuffer false
            const auto bVal = bIn.Pop();
            if (locationN < kComputeTilesN - 1) {
              bOut.Push(bVal);
            }

          Unroll_N:
            for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
              #pragma HLS UNROLL

              const bool inBoundsN = ((n0 * kInnerTilesN * kComputeTileSizeN +
                                       n1 * kComputeTileSizeN + n2) < size_n);

              ComputePackM_t cStore;
              const auto cPrev = (k > 0)
                                     ? cBuffer[n1 * kInnerTilesM + m1][n2]
                                     : ComputePackM_t(static_cast<Data_t>(0));

            Unroll_M:
              for (unsigned m2 = 0; m2 < kComputeTileSizeM; ++m2) {
                #pragma HLS UNROLL

                const bool inBoundsM = ((m0 * kInnerTilesM * kComputeTileSizeM +
                                         m1 * kComputeTileSizeM + m2) < size_m);

                const bool inBounds = inBoundsN && inBoundsM;

                const auto mapped = OperatorMap::Apply(aVal[n2], bVal[m2]);
                MM_MULT_RESOURCE_PRAGMA(mapped);
                const auto prev = cPrev[m2];

                const auto reduced = OperatorReduce::Apply(prev, mapped);
                MM_ADD_RESOURCE_PRAGMA(reduced);
                // If out of bounds, propagate the existing value instead of
                // storing the newly computed value
                cStore[m2] = inBounds ? reduced : prev; 
                #pragma HLS DEPENDENCE variable=cBuffer false
              }

              cBuffer[n1 * kInnerTilesM + m1][n2] = cStore;
            }

            // End compute tile -----------------------------------------------
          }
        }

        // End outer tile -----------------------------------------------------
      }

      // Write back tile of C -------------------------------------------------
      // 
      // This uses a flattened implementation of the loops, as we otherwise
      // introduce a lot of pipeline drains, which can have a small performance
      // impact for large designs.
      //
      const unsigned writeFlattenedInner =
          (kComputeTileSizeN * kInnerTilesM +
           (kComputeTilesN - locationN - 1) * kComputeTileSizeN * kInnerTilesM);
      const unsigned writeFlattened = kInnerTilesN * writeFlattenedInner;
      ap_uint<hlslib::ConstLog2(kInnerTilesN)> n1 = 0;
      ap_uint<((kComputeTileSizeN > 1) ? hlslib::ConstLog2(kComputeTileSizeN)
                                       : 1)>
          n2 = 0;
      ap_uint<hlslib::ConstLog2(kInnerTilesM)> m1 = 0;
      unsigned inner = 0;
    WriteC_Flattened:
      for (unsigned i = 0; i < writeFlattened; ++i) {
        #pragma HLS PIPELINE II=1
        if (inner < kComputeTileSizeN * kInnerTilesM) {
          cOut.Push(cBuffer[n1 * kInnerTilesM + m1][n2]);
          if (m1 == kInnerTilesM - 1) {
            m1 = 0;
            if (n2 == kComputeTileSizeN - 1) {
              n2 = 0;
            } else {
              ++n2;
            }
          } else {
            ++m1;
          }
        } else {
          if (locationN < kComputeTilesN - 1) {
            cOut.Push(cIn.Pop());
          }
        }
        if (inner == writeFlattenedInner - 1) {
          inner = 0;
          ++n1;
        } else {
          ++inner;
        }
      }

      // Non-flattened implementation below
    // WriteC_N1:
    //   for (unsigned n1 = 0; n1 < kInnerTilesN; ++n1) {
    //   WriteC_N2:
    //     for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
    //     WriteC_M1:
    //       for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {
    //         #pragma HLS PIPELINE II=1
    //         #pragma HLS LOOP_FLATTEN
    //         cOut.Push(cBuffer[n1 * kInnerTilesM + m1][n2]);
    //       }
    //     }
    //
    //     // Forward other tiles of C
    //     // ----------------------------------------------
    //     // We send values upwards, so first tile forwards N-1 times, and the
    //     // last tile forwards 0 times.
    //     if (locationN < kComputeTilesN - 1) {
    //     ForwardC_Others:
    //       for (unsigned l = 0; l < kComputeTilesN - locationN - 1; ++l) {
    //       ForwardC_N2:
    //         for (unsigned n2 = 0; n2 < kComputeTileSizeN; ++n2) {
    //         ForwardC_M1:
    //           for (unsigned m1 = 0; m1 < kInnerTilesM; ++m1) {
    //             #pragma HLS PIPELINE II=1
    //             #pragma HLS LOOP_FLATTEN
    //             cOut.Push(cIn.Pop());
    //           }
    //         }
    //       }
    //     }
    //   }

      // -----------------------------------------------------------------------
    }
  }
}

void DiscretizeX(Stream<ComputePackN_t> &xIn,
                 Stream<ComputePackN_t> &xBoolOut) {
  auto xVal = xIn.Pop();
  #ifndef MM_SYNTHESIS
    std::cerr << "xVal = " << xVal << "\n";
  #endif
  ComputePackN_t pack;
  for (unsigned w = 0; w < ComputePackN_t::kWidth; ++w) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
    pack[w] = xVal.Get(w) > 0;
  }
  xBoolOut.Push(pack);
}

void UpdateY1(Stream<ComputePackN_t> &xIn,
              Stream<ComputePackN_t> &xOut,
              Stream<ComputePackN_t> &yIn,
              Stream<ComputePackN_t> &yOut,
              Data_t da) {
  const auto xVal = xIn.Pop();
  const auto yVal = yIn.Pop();
  #ifndef MM_SYNTHESIS
    std::cerr << "UpdateY1: xVal = " << xVal.Get(0) << "\n";
  #endif
  xOut.Push(xVal);
  ComputePackN_t pack;
  for (unsigned w = 0; w < ComputePackN_t::kWidth; ++w) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
    pack[w] = yVal.Get(w) + xVal.Get(w) * da;
  }
  yOut.Push(pack);
}

void UpdateY2(Stream<ComputePackN_t> &jxIn,
              Stream<ComputePackN_t> &yIn,
              Stream<ComputePackN_t> &yOut,
              Data_t c0,
              Data_t dt) {
  const auto jxVal = jxIn.Pop();
  const auto yVal = yIn.Pop();
  ComputePackN_t pack;
  for (unsigned w = 0; w < ComputePackN_t::kWidth; ++w) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_FLATTEN
    pack[w] = jxVal.Get(w) * c0 * dt + yVal.Get(w);
  }
  #ifndef MM_SYNTHESIS
    std::cerr << "UpdateY2: pack = " << pack << "\n";
  #endif
  yOut.Push(pack);
}

void UpdateX(Stream<ComputePackN_t> &xIn,
             Stream<ComputePackN_t> &xOut,
             Stream<ComputePackN_t> &yIn,
             Stream<ComputePackN_t> &yOut,
             Data_t dt) {
  const auto yVal = yIn.Pop();
  yOut.Push(yVal);
  xOut.Push(xIn.Pop() + yVal * dt);
}

void Bound(Stream<ComputePackN_t> &xIn,
           Stream<ComputePackN_t> &xOut,
           Stream<ComputePackN_t> &yIn,
           Stream<ComputePackN_t> &yOut) {
  const auto xVal = xIn.Pop();
  if (xVal > 1) {
    xOut.Push(1);
    yOut.Push(0);
  } else if (xVal < -1) {
    xOut.Push(-1);
    yOut.Push(0);
  } else {
    xOut.Push(xVal);
    yOut.Push(yIn.Pop());
  }
}

/*
typedef union {
    unsigned int u;
    float f;
} ap32;

void SimulatedBifurcationOptimizer::matrixVectorProduct(int blockNum, unsigned short blockSize) {
    // TODO: blockSize tells the program how many _signOfX should we get
Multiply_matrix:
    for (unsigned int i = 0; i < CACHE_FACTOR; ++i) {
    Multiply_rows:
        for (unsigned int j = 0; j < CACHE_FACTOR; ++j) {
            ap32 temp = {0};
            temp.f = _jCache[i][j];
            ap_uint<32> tempu = temp.u;
            // Flip the sign bit when sign of x is -1 (0)
            tempu[31] = tempu[31] ^ !(_signOfX[blockNum * CACHE_FACTOR + j]);
            temp.u = tempu;
            if (i == 0)
                _productCache[j] = temp.f;
            else
                _productCache[j] += temp.f;
        }
    }
}
*/


/*
void SimulatedBifurcationOptimizer::packSolution(unsigned int * solution_uint, bool * solution_bool) {
Copy_solution_blocks:
    for (int i = 0; i < _qubits / 32; ++i) {
#pragma HLS loop_tripcount min=2 max=32
        ap_uint<32> u = {0};
    Copy_spins:
        for (int j = 0; j < 32; ++j) {
            u[j] = solution_bool[32 * i + j];
        }
#ifndef __SYNTHESIS__
        std::cout << "u = " << u << "\n";
#endif
        solution_uint[i] = u;
    }
}
*/
