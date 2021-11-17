/*-------------------------------------------------------------------------
 *
 * CUDA functions for Steepest descend in POCS-type algorithms.
 *
 * This file will iteratively minimize by stepest descend the total variation 
 * of the input image, with the parameters given, using GPUs.
 *
 * CODE by       Ander Biguri & Alexey Smirnov
 *
---------------------------------------------------------------------------
---------------------------------------------------------------------------
Copyright (c) 2015, University of Bath and CERN- European Organization for 
Nuclear Research
All rights reserved.

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, 
this list of conditions and the following disclaimer in the documentation 
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
 ---------------------------------------------------------------------------

Contact: tigre.toolbox@gmail.com
Codes  : https://github.com/CERN/TIGRE
--------------------------------------------------------------------------- 
 */







#define MAXTHREADS 1024

#include "POCS_TV.hpp"




#define cudaCheckErrors(msg) \
do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
                mexPrintf("ERROR in: %s \n",msg);\
                mexErrMsgIdAndTxt("err",cudaGetErrorString(__err));\
        } \
} while (0)
    
// CUDA kernels
//https://stackoverflow.com/questions/21332040/simple-cuda-kernel-optimization/21340927#21340927
    __global__ void divideArrayScalar(float* vec,float scalar,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]/=scalar;
        }
    }
    __global__ void multiplyArrayScalar(float* vec,float scalar,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]*=scalar;
        }
    }
    __global__ void substractArrays(float* vec,float* vec2,const size_t n)
    {
        unsigned long long i = (blockIdx.x * blockDim.x) + threadIdx.x;
        for(; i<n; i+=gridDim.x*blockDim.x) {
            vec[i]-=vec2[i];
        }
    }
    
    __device__ __inline__
            void gradient(const float* u, float* grad,
            long z, long y, long x,
            long depth, long rows, long cols)
    {
        unsigned long size2d = rows*cols;
        unsigned long long idx = z * size2d + y * cols + x;
        
        float uidx = u[idx];
        
        if ( z - 1 >= 0 && z<depth) {
            grad[0] = (uidx-u[(z-1)*size2d + y*cols + x]) ;
        }
        
        if ( y - 1 >= 0 && y<rows){
            grad[1] = (uidx-u[z*size2d + (y-1)*cols + x]) ;
        }
        
        if ( x - 1 >= 0 && x<cols) {
            grad[2] = (uidx-u[z*size2d + y*cols + (x-1)]);
        }
    }
    
	__device__ __inline__
            void apgm_v(const float* phi_old, 
			const float* v_old, float *v_new,
			const float* df, const float *dfi,
			const float *dfj, const float *dfii,
			const float *dfjj, const float *dfij,
			const float *dfji,
            long z, long y, long x,
            long depth, long rows, long cols)
    {
        unsigned long size2d = rows*cols;
		unsigned long size3d = depth * rows * cols + rows * cols + cols;
        unsigned long long idx = z * size2d + y * cols + x;
        
		unsigned mu = 1e5;
		unsigned gamma = 0.33;
		
		if ( x >= cols || y >= rows || z >= depth || x < 1 || y < 1 || z < 1 )
            return;
		
		float fv[2] 	={0,0}; 
		float fvi[2] 	={0,0};
		float fvii[2] 	={0,0};
		float fvij[2] 	={0,0};
		float p1[2] 	={0,0};
		float p2[2] 	={0,0};
		float p3[2] 	={0,0};
		
    		float p0x[1] 	= 0.0; 
		float p0y[1] 	= 0.0;
		float p1x[1] 	= 0.0;
		float p1y[1] 	= 0.0;
		float p2x[1] 	= 0.0;
		float p2y[1] 	= 0.0;
		float p3x[1] 	= 0.0;
		float p3y[1] 	= 0.0;
		
		float dfi0 = dfi[0];
		float df0 = df[0];
		float df1 = df[1];
		float dfij0 = dfij[0];
		float dfji1 = dfji[1];
		float dfii0 = dfii[0];
		float dfjj1 = dfjj[1];
		
		float phi_old0x = phi_old[idx-1];
		float phi_old0y = phi_old[idx+size3d-cols];
		float phi_old1x = phi_old[idx];
		float phi_old1y = phi_old[idx+size3d];
		
		float phi_old2x = phi_old[idx+cols];
		float phi_old2y = phi_old[idx+size3d+1];
		float phi_old3x = phi_old[idx+cols-1];
		float phi_old3y = phi_old[idx+size3d-cols+1];
		
		float vold00 = v_old[idx]; 
		float vold01 = v_old[idx+2*size3d];
		float vold02 = v_old[idx+2*size3d-cols];
		float vold03 = v_old[idx+2*size3d+1];
		float vold04 = v_old[idx+2*size3d-cols+1];
		float vold05 = v_old[idx+4*size3d];
		float vold06 = v_old[idx+4*size3d+1];
		
		float vold10 = v_old[idx+3*size3d];
		float vold11 = v_old[idx+size3d];
		float vold12 = v_old[idx+size3d+cols];
		float vold13 = v_old[idx+size3d-1];
		float vold14 = v_old[idx+size3d+cols-1];
		float vold15 = v_old[idx+5*size3d];
		float vold16 = v_old[idx+5*size3d+cols];
		
		float vold20 = v_old[idx-1];
		float vold21 = v_old[idx+2*size3d-1];
		float vold22 = v_old[idx+2*size3d-cols-1];
		float vold23 = v_old[idx+2*size3d];
		float vold24 = v_old[idx+2*size3d-cols];
		float vold25 = v_old[idx+4*size3d-1];
		float vold26 = v_old[idx+4*size3d];
		
		float vold30 = v_old[idx+3*size3d-cols];
		float vold31 = v_old[idx+size3d-cols];
		float vold32 = v_old[idx+size3d];
		float vold33 = v_old[idx+size3d-cols-1];
		float vold34 = v_old[idx+size3d-1];
		float vold35 = v_old[idx+5*size3d-cols];
		float vold36 = v_old[idx+5*size3d];
		
		float vold40 = v_old[idx+cols-1];
		float vold41 = v_old[idx+2*size3d+cols-1];
		float vold42 = v_old[idx+2*size3d-1];
		float vold43 = v_old[idx+2*size3d+cols];
		float vold44 = v_old[idx+2*size3d];
		float vold45 = v_old[idx+4*size3d+cols-1];
		float vold46 = v_old[idx+4*size3d+cols];
		
		float vold50 = v_old[idx+cols];
		float vold51 = v_old[idx+2*size3d+cols];
		float vold52 = v_old[idx+2*size3d];
		float vold53 = v_old[idx+2*size3d+cols+1];
		float vold54 = v_old[idx+2*size3d+1];
		float vold55 = v_old[idx+4*size3d+cols];
		float vold56 = v_old[idx+4*size3d+cols+1];
		
		float vold60 = v_old[idx+3*size3d+1];
		float vold61 = v_old[idx+size3d+1];
		float vold62 = v_old[idx+size3d+cols+1];
		float vold63 = v_old[idx+size3d];
		float vold64 = v_old[idx+size3d+cols];
		float vold65 = v_old[idx+5*size3d+1];
		float vold66 = v_old[idx+5*size3d+cols+1];
		
		float vold70 = v_old[idx+3*size3d-cols+1];
		float vold71 = v_old[idx+size3d-cols+1];
		float vold72 = v_old[idx+size3d+1];
		float vold73 = v_old[idx+size3d-cols];
		float vold74 = v_old[idx+size3d];
		float vold75 = v_old[idx+5*size3d-cols+1];
		float vold76 = v_old[idx+5*size3d+1];
		
		if ( x - 1 >= 0 && x<cols)
			
		fv[0] = v_old00 + 0.25 * ( v_old01 + v_old02 + v_old03 + v_old04 ) + 0.5 * ( v_old05 + v_old06 ) ;
		fv[1] = v_old10 + 0.25 * ( v_old11 + v_old12 + v_old13 + v_old14 ) + 0.5 * ( v_old15 + v_old16 ) ; 
		fvi[0] = v_old20 + 0.25 * ( v_old21 + v_old22 + v_old23 + v_old24 ) + 0.5 * ( v_old25 + v_old26 ) ;
		fvi[1] = v_old30 + 0.25 * ( v_old31 + v_old32 + v_old33 + v_old34 ) + 0.5 * ( v_old35 + v_old36 ) ;
		fvii[0] = v_old40 + 0.25 * ( v_old41 + v_old42 + v_old43 + v_old44 ) + 0.5 * ( v_old45 + v_old46 ) ; 
		fvij[0] = v_old50 + 0.25 * ( v_old51 + v_old52 + v_old53 + v_old54 ) + 0.5 * ( v_old55 + v_old56 ) ; 
		fvii[1] = v_old60 + 0.25 * ( v_old61 + v_old62 + v_old63 + v_old64 ) + 0.5 * ( v_old65 + v_old66 ) ;
		fvij[1] = v_old70 + 0.25 * ( v_old71 + v_old72 + v_old73 + v_old74 ) + 0.5 * ( v_old75 + v_old76 ) ;
		
		p0x[0] = dfi0 - fvi[0] + mu * phi_old0x ;
		p0y[0] = dfj1 - fvi[1] + mu * phi_old0y ;
		p1x[0] = df0 - fv[0] + mu * phi_old1x ;
		p1y[0] = df1 - fv[1] + mu * phi_old1y ;
		p2x[0] = dfij0 - fvij[0] + mu * phi_old2x ;
		p2y[0] = dfji1 - fvii[1] + mu * phi_old2y ;
		p3x[0] = dfii0 - fvii[0] + mu * phi_old3x ;
		p3y[0] = dfjj1 - fvij[1] + mu * phi_old3y ;
		
		p1[0] = p1x[0] ;
		p1[1] = 0.25 * ( p1y[0] + p0y[0] + p2y[0] + p3y[0] ) ;
		p2[0] = 0.25 * ( p1x[0] + p2x[0] + p0x[0] + p3x[0] ) ;
		p2[1] = p1y[0] ;
		p3[0] = 0.5 * ( p1x[0] + p0x[0] ) ;
		p3[1] = 0.5 * ( p1y[0] + p0y[0] ) ;
		
		v_new[0] = ( v_old00 + gamma * p1[0] ) * ( 1 - 1 / max(mu*sqrt(v_old00*v_old00+v_old11*v_old11)/gamma, 1) ) ;
		v_new[2] = ( v_old01 + gamma * p2[0] ) * ( 1 - 1 / max(mu*sqrt(v_old01*v_old01+v_old10*v_old10)/gamma, 1) ) ;
		v_new[4] = ( v_old05 + gamma * p3[0] ) * ( 1 - 1 / max(mu*sqrt(v_old05*v_old05+v_old15*v_old15)/gamma, 1) ) ;
		
		v_new[1] = ( v_old11 + gamma * p1[1] ) * ( 1 - 1 / max(mu*sqrt(v_old00*v_old00+v_old11*v_old11)/gamma, 1) ) ;
		v_new[3] = ( v_old10 + gamma * p2[1] ) * ( 1 - 1 / max(mu*sqrt(v_old01*v_old01+v_old10*v_old10)/gamma, 1) ) ;
		v_new[5] = ( v_old15 + gamma * p3[1] ) * ( 1 - 1 / max(mu*sqrt(v_old05*v_old05+v_old15*v_old15)/gamma, 1) ) ;
		
		__syncthreads();
	}
	
	__device__ __inline__
            void apgm_phi(const float* phi_old, float *phi_new,
			const float* v_new,
			const float* df,
            long z, long y, long x,
            long depth, long rows, long cols)
    {
        unsigned long size2d = rows*cols;
		unsigned long size3d = depth * rows * cols + rows * cols + cols;
        unsigned long long idx = z * size2d + y * cols + x;
        
		unsigned mu = 1e5;
		
		if ( x >= cols || y >= rows || z >= depth || x < 1 || y < 1 || z < 1 )
            return;
		
		float fv[2] 	={0,0}; 
				
		fv[0] = v_new[idx] + 0.25 * ( v_new[idx+2*size3d] + v_new[idx+2*size3d-cols] + v_new[idx+2*size3d+1] + v_new[idx+2*size3d-cols+1] ) + 0.5 * ( v_new[idx+4*size3d] + v_new[idx+4*size3d+1] ) ;
		fv[1] = v_new[idx+3*size3d] + 0.25 * ( v_new[idx+size3d] + v_new[idx+size3d+cols] + v_new[idx+size3d-1] + v_new[idx+size3d+cols-1] ) + 0.5 * ( v_new[idx+5*size3d] + v_new[idx+5*size3d+cols] ) ; 
		phi_new[0] = phi_old[idx] + mu * ( df[0] - fv[0] ) ;
		phi_new[1] = phi_old[idx+size3d] + mu * ( df[1] - fv[1] ) ;
		
		__syncthreads();
	}
	
    __global__ void gradientTV(const float* f, float* dftv, float* phi, float* v, 
            long depth, long rows, long cols){
        unsigned long x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned long y = threadIdx.y + blockIdx.y * blockDim.y;
        unsigned long z = threadIdx.z + blockIdx.z * blockDim.z;
        unsigned long long idx = z * rows * cols + y * cols + x;
		unsigned long size3d = depth * rows * cols + rows * cols + cols;
		unsigned long maxIter = 270;
		
        if ( x >= cols || y >= rows || z >= depth || x < 1 || y < 1 || z < 1 )
            return;
        
		float phi_new[2]={0,0};
		float df[3] 	={0,0,0}; // df == \partial f_{i,j,k}
		float dfi[3] 	={0,0,0};
		float dfj[3] 	={0,0,0}; 
		float dfii[3] 	={0,0,0};
		float dfjj[3] 	={0,0,0};
		float dfij[3] 	={0,0,0};
		float dfji[3] 	={0,0,0};
		float val[4]	={0,0,0,0};
		float v_new[2] 	={0,0,0,0,0,0};
        
		gradient(f,df    ,z  ,y  ,x  , depth,rows,cols);
		gradient(f,dfi   ,z  ,y  ,x-1, depth,rows,cols);
		gradient(f,dfj   ,z  ,y-1,x  , depth,rows,cols);
		gradient(f,dfii  ,z  ,y+1,x-1, depth,rows,cols);
		gradient(f,dfjj  ,z  ,y-1,x+1, depth,rows,cols);
		gradient(f,dfij  ,z  ,y+1,x  , depth,rows,cols);
		gradient(f,dfji  ,z  ,y  ,x+1, depth,rows,cols);
		
		phi[idx] = 0; // initialize dual variables with 0's
		phi[idx + size3d] = 0; 
		for(unsigned int i=0;i<6;i++){
			v[idx + i*size3d] = 0;
		}
	    v[idx] = df[0]; // assign df1 to v1x
		v[idx + 3*size3d] = df[1]; // assign df2 to v2y
		
		for(unsigned int i=0;i<maxIter;i++){
			apgm_v(phi,v,v_new,df,z,y,x,depth,rows,cols); // perform one step of the algorithm
			for(unsigned int j=0;j<6;j++){
				v[idx + j*size3d] = v_new[j];
			}
			apgm_phi(phi,phi_new,v,df,z,y,x, depth,rows,cols);
			phi[idx] = phi_new[0];
			phi[idx] = phi_new[1];
		}
		
		val[0] = phi[idx];
		val[1] = phi[idx + size3d];
		val[2] = phi[idx - 1];
		val[3] = phi[idx + size3d - cols];
		dftv[idx] = -val[0]-val[1]+val[2]+val[3];			
    }
    
    __device__ void warpReduce(volatile float *sdata, size_t tid) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
    
    __global__ void  reduceNorm2(float *g_idata, float *g_odata, size_t n){
        extern __shared__ volatile float sdata[];
        //http://stackoverflow.com/a/35133396/1485872
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x*blockDim.x + tid;
        size_t gridSize = blockDim.x*gridDim.x;
        float mySum = 0;
        float value=0;
        while (i < n) {
            value=g_idata[i]; //avoid reading twice
            mySum += value*value;
            i += gridSize;
        }
        sdata[tid] = mySum;
        __syncthreads();
        
        if (tid < 512)
            sdata[tid] += sdata[tid + 512];
        __syncthreads();
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
        
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
        
        if (tid <  64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
        
        
#if (__CUDA_ARCH__ >= 300)
        if ( tid < 32 )
        {
            mySum = sdata[tid] + sdata[tid + 32];
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                mySum += __shfl_down(mySum, offset);
            }
        }
#else
        if (tid < 32) {
            warpReduce(sdata, tid);
            mySum = sdata[0];
        }
#endif
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }
    __global__ void  reduceSum(float *g_idata, float *g_odata, size_t n){
        extern __shared__ volatile float sdata[];
        size_t tid = threadIdx.x;
        size_t i = blockIdx.x*blockDim.x + tid;
        size_t gridSize = blockDim.x*gridDim.x;
        float mySum = 0;
       // float value=0;
        while (i < n) {
            mySum += g_idata[i];
            i += gridSize;
        }
        sdata[tid] = mySum;
        __syncthreads();
        
        if (tid < 512)
            sdata[tid] += sdata[tid + 512];
        __syncthreads();
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
        
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
        
        if (tid <  64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
        
        
#if (__CUDA_ARCH__ >= 300)
        if ( tid < 32 )
        {
            mySum = sdata[tid] + sdata[tid + 32];
            for (int offset = warpSize/2; offset > 0; offset /= 2) {
                mySum += __shfl_down(mySum, offset);
            }
        }
#else
        if (tid < 32) {
            warpReduce(sdata, tid);
            mySum = sdata[0];
        }
#endif
        if (tid == 0) g_odata[blockIdx.x] = mySum;
    }
    
    
    
    
// main function
 void pocs_tv(const float* img,float* dst,float alpha,const long* image_size, int maxIter){
        
    
        size_t total_pixels = image_size[0] * image_size[1]  * image_size[2] ;
        size_t mem_size = sizeof(float) * total_pixels;
        
        float *d_image, *d_dimgTV,*d_norm2aux,*d_norm2,*phi,*grad_img;
        // memory for image
        cudaMalloc(&d_image, mem_size);
        cudaCheckErrors("Malloc Image error");
        cudaMemcpy(d_image, img, mem_size, cudaMemcpyHostToDevice);
        cudaCheckErrors("Memory Malloc and Memset: SRC");
        // memory for df
        cudaMalloc(&d_dimgTV, mem_size);
        cudaCheckErrors("Memory Malloc and Memset: TV");
        
        cudaMalloc(&d_norm2, mem_size);
        cudaCheckErrors("Memory Malloc and Memset: TV");
        
		// memory for dual variables
        cudaMalloc(&phi, 3*mem_size); // as if phi was in 3D
        cudaCheckErrors("Memory Malloc and Memset: PHI");
        
		cudaMalloc(&grad_img, 6*mem_size); // as if gradient field is in 2D (should be 12 for 3D)
        cudaCheckErrors("Memory Malloc and Memset: GRAD");
		
		// memory for L2norm auxiliar
        cudaMalloc(&d_norm2aux, sizeof(float)*(total_pixels + MAXTHREADS - 1) / MAXTHREADS);
        cudaCheckErrors("Memory Malloc and Memset: NORMAux");
        
        
        
        // For the gradient
        dim3 blockGrad(10, 10, 10);
        dim3 gridGrad((image_size[0]+blockGrad.x-1)/blockGrad.x, (image_size[1]+blockGrad.y-1)/blockGrad.y, (image_size[2]+blockGrad.z-1)/blockGrad.z);
        
        // For the reduction
        float sumnorm2;
        
        
        
        for(unsigned int i=0;i<maxIter;i++){
            
            
            // Compute the gradient of the TV norm
            gradientTV<<<gridGrad, blockGrad>>>(d_image,d_dimgTV,phi,grad_img,image_size[2], image_size[1],image_size[0]);
            cudaCheckErrors("Gradient");
//             cudaMemcpy(dst, d_dimgTV, mem_size, cudaMemcpyDeviceToHost);
            
            
            cudaMemcpy(d_norm2, d_dimgTV, mem_size, cudaMemcpyDeviceToDevice);
            cudaCheckErrors("Copy from gradient call error");
            // Compute the L2 norm of the gradint. For that, reduction is used.
            //REDUCE
            size_t dimblockRed = MAXTHREADS;
            size_t dimgridRed = (total_pixels + MAXTHREADS - 1) / MAXTHREADS;
            reduceNorm2 << <dimgridRed, dimblockRed, MAXTHREADS*sizeof(float) >> >(d_norm2, d_norm2aux, total_pixels);
            cudaCheckErrors("reduce1");
            if (dimgridRed > 1) {
                reduceSum << <1, dimblockRed, MAXTHREADS*sizeof(float) >> >(d_norm2aux, d_norm2, dimgridRed);
                cudaCheckErrors("reduce2");
                cudaMemcpy(&sumnorm2, d_norm2, sizeof(float), cudaMemcpyDeviceToHost);
                cudaCheckErrors("cudaMemcpy");
                
            }
            else {
                cudaMemcpy(&sumnorm2, d_norm2aux, sizeof(float), cudaMemcpyDeviceToHost);
                cudaCheckErrors("cudaMemcpy");
            }
            //mexPrintf("%f ",sqrt(sumnorm2));
            //NOMRALIZE
            //in a Tesla, maximum blocks =15 SM * 4 blocks/SM
            divideArrayScalar  <<<60,MAXTHREADS>>>(d_dimgTV,sqrt(sumnorm2),total_pixels);
            cudaCheckErrors("Division error");
            //MULTIPLY HYPERPARAMETER
            multiplyArrayScalar<<<60,MAXTHREADS>>>(d_dimgTV,alpha,   total_pixels);
            cudaCheckErrors("Multiplication error");
            //SUBSTRACT GRADIENT
            substractArrays    <<<60,MAXTHREADS>>>(d_image,d_dimgTV, total_pixels);
            cudaCheckErrors("Substraction error");
            sumnorm2=0;
        }
        
        cudaCheckErrors("TV minimization");
        
        cudaMemcpy(dst, d_image, mem_size, cudaMemcpyDeviceToHost);
        cudaCheckErrors("Copy result back");
        
        cudaFree(d_image);
        cudaFree(d_norm2aux);
        cudaFree(d_dimgTV);
        cudaFree(d_norm2);
		cudaFree(phi);
		cudaFree(grad_img);

        cudaCheckErrors("Memory free");
        cudaDeviceReset();
    }
    
