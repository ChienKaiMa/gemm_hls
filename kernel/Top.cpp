#include "MatrixMultiplication.h"
#include "Compute.h"
#include "Memory.h"
#include "hlslib/xilinx/Simulation.h"

#ifdef MM_TRANSPOSED_A
void MatrixMultiplicationKernel(MemoryPackN_t const a[],
                                MemoryPackM_t const b[], MemoryPackM_t c[]
#else
void MatrixMultiplicationKernel(MemoryPackK_t const a[],
                                MemoryPackM_t const b[], MemoryPackM_t c[]
#endif
#ifdef MM_DYNAMIC_SIZES
                                ,
                                const unsigned size_n, const unsigned size_k,
                                const unsigned size_m
#endif
) {

  #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem2

  #pragma HLS DATAFLOW

#ifndef MM_DYNAMIC_SIZES
  const unsigned size_n = kSizeN;
  const unsigned size_k = kSizeK;
  const unsigned size_m = kSizeM;
#endif

  // Memory accesses and pipes for A 
#ifndef MM_TRANSPOSED_A
  Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth];
  #pragma HLS STREAM variable=aSplit depth=2*kOuterTileSizeN
  Stream<Data_t> aConvert("aConvert");
#else
  Stream<MemoryPackN_t, 2 * kOuterTileSizeNMemory> aMemory("aMemory");
#endif
  Stream<ComputePackN_t, kPipeDepth> aPipes[kComputeTilesN + 1];

  // Memory accesses and pipes for B 
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> bMemory("bMemory");
  Stream<ComputePackM_t, kPipeDepth> bPipes[kComputeTilesN + 1];

  // Pipes for C
  Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];

#ifndef HLSLIB_SYNTHESIS
  // Name the arrays of channels for debugging purposes
#ifndef MM_TRANSPOSED_A
  for (unsigned i = 0; i < kTransposeWidth; ++i) {
    aSplit[i].set_name(("aSplit[" + std::to_string(i) + "]").c_str());
  }
#endif
  for (unsigned n = 0; n < kComputeTilesN; ++n) {
    aPipes[n].set_name(("aPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
    bPipes[n].set_name(("bPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
    cPipes[n].set_name(("cPipes[" + std::to_string(n) + "]").c_str());
  }
#endif

  HLSLIB_DATAFLOW_INIT();

  // Only convert memory width if necessary
#ifndef MM_TRANSPOSED_A
  HLSLIB_DATAFLOW_FUNCTION(ReadA, a, aSplit, size_n, size_k, size_m);
#ifdef MM_CONVERT_A
  HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aConvert, size_n, size_k,
                           size_m);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthA, aConvert, aPipes[0], size_n, size_k,
                           size_m);
#else
  HLSLIB_DATAFLOW_FUNCTION(TransposeA, aSplit, aPipes[0], size_n, size_k,
                           size_m);
#endif
#else
  HLSLIB_DATAFLOW_FUNCTION(ReadATransposed, a, aMemory, size_n, size_k, size_m);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthATransposed, aMemory, aPipes[0], size_n,
                           size_k, size_m);
#endif

  HLSLIB_DATAFLOW_FUNCTION(ReadB, b, bMemory, size_n, size_k, size_m);

    // Only convert memory width if necessary
#ifdef MM_CONVERT_B
  Stream<ComputePackM_t> bFeed("bFeed");
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, bMemory, bFeed, size_n, size_k,
                           size_m);
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bFeed, bPipes[0], size_n, size_k, size_m);
#else
  HLSLIB_DATAFLOW_FUNCTION(FeedB, bMemory, bPipes[0], size_n, size_k, size_m);
#endif

  for (unsigned pe = 0; pe < kComputeTilesN; ++pe) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ProcessingElement,
                             aPipes[pe],
                             aPipes[pe + 1],
                             bPipes[pe],
                             bPipes[pe + 1],
                             cPipes[pe],
                             cPipes[pe + 1],
                             pe, size_n, size_k, size_m);
  }

  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> cMemory("cMemory");
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthC, cPipes[0], cMemory, size_n, size_k,
                           size_m);
  HLSLIB_DATAFLOW_FUNCTION(WriteC, cMemory, c, size_n, size_k, size_m);

  HLSLIB_DATAFLOW_FINALIZE();
}

void SimulatedBifurcationKernel(MemoryPackN_t isingJ[], MemoryPackM_t x[], MemoryPackM_t y[],
                                Data_t const delta_a[], const Data_t c0, const Data_t dt,
                                const unsigned int numOfQubits, const unsigned short numOfSteps,
                                unsigned int solution[], Data_t x_history[]) {
  #pragma HLS INTERFACE m_axi port=isingJ offset=slave bundle=gmem0
  #pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem1
  #pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem2
  #pragma HLS INTERFACE m_axi port=delta_a offset=slave bundle=gmem2
  #pragma HLS INTERFACE m_axi port=solution offset=slave bundle=gmem3
  #pragma HLS INTERFACE m_axi port=x_history offset=slave bundle=gmem4

  // TODO

  for (unsigned int i = 0; i < numOfSteps; ++i) {
    #ifndef HLSLIB_SYNTHESIS
    std::cout << "Step " << i << "\n";
    #endif
    SimulatedBifurcationStep(isingJ, x, y, delta_a[i], c0, dt, numOfQubits);
    // TODO
    // More efficient way to save x_history
    for (unsigned int j = 0; j < numOfQubits; ++j)
    {
      x_history[i * numOfQubits + j] = x[j];
    }
  }
  // TODO
  // DiscretizeAll
  // WriteC(xPipes, solution, numOfQubits, numOfQubits, 1);
}

void SimulatedBifurcationStep(MemoryPackN_t isingJ[], MemoryPackM_t x[], MemoryPackM_t y[],
                              const Data_t delta_a, const Data_t c0, const Data_t dt,
                              const unsigned int numOfQubits) {
  // TODO
  // Memory accesses and pipes for A 
#ifndef MM_TRANSPOSED_A
  Stream<Data_t, 2 * kOuterTileSizeN> aSplit[kTransposeWidth];
  #pragma HLS STREAM variable=aSplit depth=2*kOuterTileSizeN
  Stream<Data_t> aConvert("aConvert");
#else
  Stream<MemoryPackN_t, 2 * kOuterTileSizeNMemory> aMemory("aMemory");
#endif
  Stream<ComputePackN_t, kPipeDepth> aPipes[kComputeTilesN + 1];

  // Memory accesses and pipes for x, y
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> xMemory("xMemory");
  // TODO fix data type
  Stream<ComputePackM_t, kPipeDepth> xBoolPipes[kComputeTilesN + 1];
  Stream<MemoryPackM_t, 2 * kOuterTileSizeMMemory> yMemory("yMemory");
  Stream<ComputePackM_t, kPipeDepth> xPipes[6];
  Stream<ComputePackM_t, kPipeDepth> yPipes[5];

  // Pipes for C
  Stream<ComputePackM_t> cPipes[kComputeTilesN + 1];

#ifndef HLSLIB_SYNTHESIS
  // TODO
  // Name the arrays of channels for debugging purposes
#ifndef MM_TRANSPOSED_A
  for (unsigned i = 0; i < kTransposeWidth; ++i) {
    aSplit[i].set_name(("aSplit[" + std::to_string(i) + "]").c_str());
  }
#endif
  for (unsigned n = 0; n < kComputeTilesN; ++n) {
    aPipes[n].set_name(("aPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < 6; ++n) {
    xPipes[n].set_name(("xPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < 5; ++n) {
    yPipes[n].set_name(("yPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
    xBoolPipes[n].set_name(("xBoolPipes[" + std::to_string(n) + "]").c_str());
  }
  for (unsigned n = 0; n < kComputeTilesN + 1; ++n) {
    cPipes[n].set_name(("cPipes[" + std::to_string(n) + "]").c_str());
  }
#endif

  #pragma HLS DATAFLOW
  HLSLIB_DATAFLOW_INIT();

  HLSLIB_DATAFLOW_FUNCTION(ReadATransposed, isingJ, aMemory, numOfQubits, numOfQubits, 1);
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthATransposed, aMemory, aPipes[0], numOfQubits, numOfQubits, 1);
  HLSLIB_DATAFLOW_FUNCTION(ReadB, x, xMemory, numOfQubits, numOfQubits, 1);
  HLSLIB_DATAFLOW_FUNCTION(ReadB, y, yMemory, numOfQubits, numOfQubits, 1);

    // Only convert memory width if necessary
#ifdef MM_CONVERT_X
  Stream<ComputePackM_t> xFeed("xFeed");
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, xMemory, xFeed, numOfQubits, numOfQubits, 1);
  HLSLIB_DATAFLOW_FUNCTION(FeedB, xFeed, xPipes[0], numOfQubits, numOfQubits, 1);
#else
  HLSLIB_DATAFLOW_FUNCTION(FeedB, xMemory, xPipes[0], numOfQubits, numOfQubits, 1);
#endif
  HLSLIB_DATAFLOW_FUNCTION(DoubleStreamX, xPipes[0], xPipes[1], xPipes[2]);

#ifdef MM_CONVERT_Y
  Stream<ComputePackM_t> yFeed("yFeed");
  HLSLIB_DATAFLOW_FUNCTION(ConvertWidthB, yMemory, yFeed, numOfQubits, numOfQubits, 1);
  HLSLIB_DATAFLOW_FUNCTION(FeedB, yFeed, yPipes[0], numOfQubits, numOfQubits, 1);
#else
  HLSLIB_DATAFLOW_FUNCTION(FeedB, yMemory, yPipes[0], numOfQubits, numOfQubits, 1);
#endif

  HLSLIB_DATAFLOW_FUNCTION(DiscretizeX, xPipes[1], xBoolPipes[0]);
  HLSLIB_DATAFLOW_FUNCTION(UpdateY1, xPipes[2], xPipes[3], yPipes[0], yPipes[1], delta_a);
  for (unsigned pe = 0; pe < kComputeTilesN; ++pe) {
    #pragma HLS UNROLL
    HLSLIB_DATAFLOW_FUNCTION(ProcessingElement,
                             aPipes[pe],
                             aPipes[pe + 1],
                             xBoolPipes[pe],
                             xBoolPipes[pe + 1],
                             cPipes[pe],
                             cPipes[pe + 1],
                             pe, numOfQubits, numOfQubits, 1);
  }
  // TODO
  // Connect to correct pipes
  HLSLIB_DATAFLOW_FUNCTION(UpdateY2, cPipes[0], yPipes[1], yPipes[2], c0, dt);
  HLSLIB_DATAFLOW_FUNCTION(UpdateX, xPipes[3], xPipes[4], yPipes[2], yPipes[3], dt);
  HLSLIB_DATAFLOW_FUNCTION(Bound, xPipes[4], xPipes[5], yPipes[3], yPipes[4]);
  HLSLIB_DATAFLOW_FINALIZE();
  // TODO
  // Write back x and y
  WriteC(xPipes[5], x, numOfQubits, numOfQubits, 1);
  WriteC(yPipes[4], y, numOfQubits, numOfQubits, 1);

}

/*
void SimulatedBifurcationStep(float* isingJ, float* x, float* y) {
Discretize_all:
    discretize(x, _signOfX);
    int counter = 0;
Update_blocks:
    for (unsigned int i = 0; i < _qubits / float(CACHE_FACTOR); ++i) {
DO_PRAGMA(HLS loop_tripcount min=1 max=4)
// TODO �?a0
// constexpr, MACRO...?
        // TODO: Prefetch x and y to xCache and yCache
    Prepare_data_xy:
        for (unsigned int k = 0; k < CACHE_FACTOR; ++k) {
            _yCache[k] = y[counter + k];
            _xCache[k] = x[counter + k];
        }
    Update_y_stage1:
        for (unsigned int k = 0; k < CACHE_FACTOR; ++k) {
            _yCache[k] -= _xCache[k] * _c2;
        }
    Update_y_stage2:
        for (unsigned int j = 0; j < _qubits / CACHE_FACTOR; ++j) {
    #pragma HLS loop_tripcount min=1 max=16
            // TODO: Prefetch J to jCache
        Prepare_data_j:
            for (unsigned int k = 0; k < CACHE_FACTOR; ++k) {
                for (unsigned int l = 0; l < CACHE_FACTOR; ++l) {
                    _jCache[k][l] = isingJ[(i * CACHE_FACTOR + k) * CACHE_FACTOR + j * CACHE_FACTOR + l];
                }
            }
            matrixVectorProduct(j, CACHE_FACTOR); // TODO: The last block may have blockSize less than CACHE_FACTOR
        }
    Update_and_write_xy:
        for (unsigned int k = 0; k < CACHE_FACTOR; ++k) {
            _yCache[k] -= _productCache[k] * _c1;
            _xCache[k] += _yCache[k] * _dt;
            bound(_xCache[k], _yCache[k]);
            // matrixVectorProduct initializes the variable
            // _productCache[k] = 0;
            x[counter + k] = _xCache[k];
            y[counter + k] = _yCache[k];
        }
        counter += CACHE_FACTOR;
    }
}
*/