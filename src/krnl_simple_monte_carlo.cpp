/**
Copyright (c) 2021, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
    Vector Addition Kernel Implementation
    Arguments:
        in1      (input)     --> Input Vector1
        in2      (input)     --> Input Vector2
        out      (output)    --> Output Vector
        elements (input)     --> Number the elements in the Vector
*/

#include "xf_fintech/rng.hpp"

extern "C" {

void krnl_simple_monte_carlo(double* x, // Write-Only Vector 1
               double* y, 				// Write-Only Vector 2
               int* out,        		// Output Result
               int elements,   			// Number of elements
			   int index				// Vector index where the result is stored
			) {
//#pragma HLS INTERFACE mode=m_axi bundle=gmem0 port=in1
//#pragma HLS INTERFACE mode=m_axi bundle=gmem1 port=in2
//#pragma HLS INTERFACE mode=m_axi bundle=gmem2 port=out

	xf::fintech::MT19937 uniformRNG(42);

// Simple vector addition kernel.
vadd1:
    for (int i = 0; i < elements; i++) {
//#pragma HLS LOOP_TRIPCOUNT avg=4096 max=4096 min=4096
//#pragma HLS UNROLL factor=16
    	x[i] = uniformRNG.next();
    	y[i] = uniformRNG.next();
        if(x[i]*x[i] +y[i]*y[i] <= 1) out[index]++;
    }
}
}
