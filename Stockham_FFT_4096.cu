


#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cudaProfiler.h>

#include <device_functions.h>
#include <device_launch_parameters.h>

#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>



#define checkCuda( fn ) do { \
		cudaError_t error = (fn); \
		if ( cudaSuccess != error ) { \
			const char* errstr = cudaGetErrorString(error); \
			printf("%s returned error %s (code %d), line(%d)\n", #fn , errstr, error, __LINE__);\
			exit(EXIT_FAILURE); \
																		} \
																				} while (0)

typedef float2 Complex;
#define my_Pi 3.1415926535898 


int SARNRN = 8192;
int SARNAN = 4096;

size_t ADSampleComplx = SARNRN * SARNAN * sizeof(Complex);


#define IFFT_ON  1
#define IFFT_OFF 0

#define IFFT_DIV_N_ON  1
#define IFFT_DIV_N_OFF 0

#define FFTSHIFT_FRONT_ON  1
#define FFTSHIFT_FRONT_OFF 0

#define FFTSHIFT_BACK_ON  1
#define FFTSHIFT_BACK_OFF 0


extern "C"
__global__ void __launch_bounds__(256) Stockham_FFT_4096_SM(Complex *data_in, int ifft_flag, int ifft_divideN_flag, int fftshift_front, int fftshift_back);



int main()
{
	
	cuProfilerStart();

	printf("begin time is %ld \r\n", clock());

	//FILE* fp;
	float ms;

	//--------------------------------------------------------------
	//--------------------------SYS INIT----------------------------
	//--------------------------------------------------------------
#pragma region SYS INIT



	//---CPU MALLOC--------
#pragma region CPU MALLOC  	


	Complex *HostPinnedMemory;
	checkCuda(cudaHostAlloc((void **)&HostPinnedMemory, ADSampleComplx, cudaHostAllocDefault));

	printf("CPU MALLOC time is %ld \r\n", clock());

#pragma endregion
	//---CPU MALLOC--------

	//---GPU MALLOC--------
#pragma region GPU MALLOC  	

	//unsigned int FreeRamSpacePointerCounter = 0;

	Complex *d_ADcomplx;
	Complex *d_tmp;
	int *d_bitrevoder_128_table;
	int *d_bitrevoder_256_table;

	checkCuda(cudaMalloc((void **)&d_ADcomplx, ADSampleComplx));        //256MB
	checkCuda(cudaMalloc((void **)&d_tmp, ADSampleComplx));             //256MB

	

	printf("GPU MALLOC time is %ld \r\n", clock());
#pragma endregion
	//---GPU MALLOC--------

	//note:　FFT 配置 初始化 会占用 显存 同等大小的空间
	//---FFT PLAN----------
#pragma region FFT PLAN  


	int RANK = 1;   //1-D FFT
	int NX = 4096;//8192
	int BATCH = 8192 ;//4096
	int iembed = 4096;
	int istride = 1;   //连续无间隔
	int idist = 4096;  //the distance between the first element of two consecutive signals in a batch of the input data
	int oembed = 4096;
	int ostride = 1;
	int odist = 4096;


	cufftHandle planRow;

	//cufftPlanMany(&plan, RANK, NX, &iembed, istride, idist,&oembed, ostride, odist, CUFFT_C2C, BATCH);

	//plan creat : 130ms
	if (cufftPlanMany(&planRow, RANK, &NX, &iembed, istride, idist, &oembed, ostride, odist, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
		fprintf(stderr, "CUFFT error: Plan creation failed");
		return 0;
	}

#pragma endregion
	//---FFT PLAN----------


	//---PIC INIT----------

#pragma endregion
	//--------------------------------------------------------------
	//--------------------------SYS INIT----------------------------
	//--------------------------------------------------------------


	//------------------------ RUN CYCLE ---------------------------
	//--------------------------------------------------------------
	//--------------------------------------------------------------

#pragma region RUN CYCLE

	//cv::namedWindow("Display", CV_WINDOW_AUTOSIZE);



		//--------------------------------------------------------------
		//-------------------------DATA INPUT---------------------------
		//--------------------------------------------------------------

#pragma region DATA INPUT

		//---ADSample----------
#pragma region ADSample 


	for (int i = 0; i < SARNRN; i++)
	{
		for (int j = 0; j < SARNAN; j++)
		{
			HostPinnedMemory[i * SARNAN + j].x = j;//%256;
			HostPinnedMemory[i * SARNAN + j].y = j;//%256;
		}
	}
	

	checkCuda(cudaMemcpy((Complex *)d_ADcomplx, (Complex *)HostPinnedMemory, ADSampleComplx, cudaMemcpyHostToDevice));//36ms拷贝

	if (cudaDeviceSynchronize() != cudaSuccess){ fprintf(stderr, "Cuda error: Failed to synchronize\n"); return 0; }
	printf("AD Sample Input done %ld \r\n", clock());

	//--------------------AD Sample Transpose--------------------


#pragma endregion
		//---ADSample----------


		//GPU TIMER
		cudaEvent_t startEvent, stopEvent;
		checkCuda(cudaEventCreate(&startEvent));
		checkCuda(cudaEventCreate(&stopEvent));
		
		checkCuda(cudaEventRecord(startEvent, 0));


		Stockham_FFT_4096_SM<< <8192, 256 >> >(d_ADcomplx, IFFT_OFF, IFFT_DIV_N_OFF, FFTSHIFT_FRONT_OFF, FFTSHIFT_BACK_OFF);
	
		
		/*if (cufftExecC2C(planRow, (cufftComplex *)d_ADcomplx, (cufftComplex *)d_ADcomplx, CUFFT_FORWARD) != CUFFT_SUCCESS ){
			fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
			return;
		}
		if (cudaDeviceSynchronize() != cudaSuccess){ fprintf(stderr, "Cuda error: Failed to synchronize\n"); return; }*/



		checkCuda(cudaEventRecord(stopEvent, 0));
		checkCuda(cudaEventSynchronize(stopEvent));
		checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
		printf("!!!!!!!!!!1!!!!!!!!!! %f \r\n", ms);


		checkCuda(cudaMemcpy((Complex *)HostPinnedMemory, (Complex *)( d_ADcomplx ), ADSampleComplx, cudaMemcpyDeviceToHost));//36ms拷贝


		


		char txtdataFileName[1024];
		FILE * TxtWriter;


		sprintf(txtdataFileName, "stockham_FFT4096.txt");//F:/RDSAR/RDSAR/txtdata/z_matrix/
		TxtWriter = fopen(txtdataFileName, "wb");
		if (TxtWriter == 0)
		{
		printf("txt write creat error \r\n");
		return 0;
		}

		for (int i = 0; i < SARNAN; i++)
		{
			fprintf(TxtWriter, "%f \r\n", HostPinnedMemory[i].x);
			fprintf(TxtWriter, "%f \r\n", HostPinnedMemory[i].y);
		
		}

		fclose(TxtWriter);
		
		


#pragma endregion



	cuProfilerStop();


#pragma endregion


	//---CPU FREE-----------
#pragma region CPU FREE

	checkCuda(cudaFreeHost(HostPinnedMemory));

#pragma endregion
	//---CPU FREE-----------	

	//---GPU FREE-----------
#pragma region GPU FREE	

	cufftDestroy(planRow);
	checkCuda(cudaFree(d_ADcomplx));
	checkCuda(cudaFree(d_tmp));

#pragma endregion
	//---GPU FREE-----------		

	//exit(EXIT_SUCCESS);
	return 0;
}







static __device__ inline Complex exp_calcu(float data_in)
{
    Complex data_out;

    data_out.x = __cosf(data_in);
    data_out.y = __sinf(data_in);
	//__sincosf(data_in, (&data_out.x),(&data_out.y));

    return data_out;
}

static __device__  inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;

    c.x = a.x*b.x - a.y*b.y;
    c.y = a.x*b.y + a.y*b.x;

    return c;
}

static __device__  inline Complex ComplexAdd(Complex a, Complex b)
{
    Complex c;

    c.x = a.x + b.x;
    c.y = a.y + b.y;

    return c;
}

static __device__  inline Complex ComplexSub(Complex a, Complex b)
{
    Complex c;

    c.x = a.x - b.x;
    c.y = a.y - b.y;

    return c;
}


static __device__ inline int StockhamR4_block(Complex *a, Complex *b, Complex *c, Complex *d, Complex *wi, float wi_c, int index)
{

    Complex tmp;
    Complex alpha, beta, gamma, delta;

    alpha = *a;                     *wi = exp_calcu(index * wi_c);  tmp = *wi;
    beta  = ComplexMul(*wi, *b);    *wi = ComplexMul(*wi, *wi);
    gamma = ComplexMul(*wi, *c);    *wi = ComplexMul(tmp, *wi);
    delta = ComplexMul(*wi, *d);

    tmp   = alpha;
    alpha = ComplexAdd(alpha, gamma);
    gamma = ComplexSub(tmp,   gamma);
    tmp   = beta;
    beta  = ComplexAdd(beta, delta);
    delta = ComplexSub(tmp,  delta);
    //tau3*wi(0,1)
    tmp.x   = delta.x;
    delta.x = -delta.y;
    delta.y = tmp.x;

    *a = ComplexAdd(alpha,  beta);
    *b = ComplexSub(gamma, delta);
    *c = ComplexSub(alpha,  beta);
    *d = ComplexAdd(gamma, delta);

    return 0;

}


static __device__ inline void Stockham_4096_block
(
    Complex *r1,  Complex *r2,  Complex *r3,  Complex *r4,
    Complex *r5,  Complex *r6,  Complex *r7,  Complex *r8,
    Complex *r9,  Complex *r10, Complex *r11, Complex *r12,
    Complex *r13, Complex *r14, Complex *r15, Complex *r16,
    Complex *data_shared
)
{

    //---------------------init-------------------------
    int bx  = blockIdx.x;
    int tid = threadIdx.x;

    int index = 0;

    float wi_c;
    Complex wi;

    const float FFT_Pi = 3.14159265359;

    ////Level in 4////////////////////////////////////////////////////////////////

    wi_c = -2*FFT_Pi/4;
    StockhamR4_block(r1,  r2,  r3,  r4,  &wi, wi_c, 0);
    StockhamR4_block(r5,  r6,  r7,  r8,  &wi, wi_c, 0);
    StockhamR4_block(r9,  r10, r11, r12, &wi, wi_c, 0);
    StockhamR4_block(r13, r14, r15, r16, &wi, wi_c, 0);

    data_shared[tid + 1024*0 + 256*0] = *r1;
    data_shared[tid + 1024*1 + 256*0] = *r2;
    data_shared[tid + 1024*2 + 256*0] = *r3;
    data_shared[tid + 1024*3 + 256*0] = *r4;
    data_shared[tid + 1024*0 + 256*1] = *r5;
    data_shared[tid + 1024*1 + 256*1] = *r6;
    data_shared[tid + 1024*2 + 256*1] = *r7;
    data_shared[tid + 1024*3 + 256*1] = *r8;
    data_shared[tid + 1024*0 + 256*2] = *r9;
    data_shared[tid + 1024*1 + 256*2] = *r10;
    data_shared[tid + 1024*2 + 256*2] = *r11;
    data_shared[tid + 1024*3 + 256*2] = *r12;
    data_shared[tid + 1024*0 + 256*3] = *r13;
    data_shared[tid + 1024*1 + 256*3] = *r14;
    data_shared[tid + 1024*2 + 256*3] = *r15;
    data_shared[tid + 1024*3 + 256*3] = *r16;
    __syncthreads();

    ////Level 16 64 256 1024///////////////////////////////////////////////////

    for(int i=1;i<=64;i=i*4)
    {
        int inexd_tmp = 256/i;

        index = (tid&(256 - inexd_tmp))*4 + tid%inexd_tmp;

        *r1  = data_shared[index + 1024*0 + inexd_tmp*0];
        *r2  = data_shared[index + 1024*0 + inexd_tmp*1];
        *r3  = data_shared[index + 1024*0 + inexd_tmp*2];
        *r4  = data_shared[index + 1024*0 + inexd_tmp*3];
        *r5  = data_shared[index + 1024*1 + inexd_tmp*0];
        *r6  = data_shared[index + 1024*1 + inexd_tmp*1];
        *r7  = data_shared[index + 1024*1 + inexd_tmp*2];
        *r8  = data_shared[index + 1024*1 + inexd_tmp*3];
        *r9  = data_shared[index + 1024*2 + inexd_tmp*0];
        *r10 = data_shared[index + 1024*2 + inexd_tmp*1];
        *r11 = data_shared[index + 1024*2 + inexd_tmp*2];
        *r12 = data_shared[index + 1024*2 + inexd_tmp*3];
        *r13 = data_shared[index + 1024*3 + inexd_tmp*0];
        *r14 = data_shared[index + 1024*3 + inexd_tmp*1];
        *r15 = data_shared[index + 1024*3 + inexd_tmp*2];
        *r16 = data_shared[index + 1024*3 + inexd_tmp*3];


        wi_c = -2*FFT_Pi/(16*i);
        index = tid / inexd_tmp;
        StockhamR4_block(r1,  r2,  r3,  r4,  &wi, wi_c, index + i*0);
        StockhamR4_block(r5,  r6,  r7,  r8,  &wi, wi_c, index + i*1);
        StockhamR4_block(r9,  r10, r11, r12, &wi, wi_c, index + i*2);
        StockhamR4_block(r13, r14, r15, r16, &wi, wi_c, index + i*3);

		__syncthreads();

        data_shared[tid + 1024*0 + 256*0] = *r1;
        data_shared[tid + 1024*1 + 256*0] = *r2;
        data_shared[tid + 1024*2 + 256*0] = *r3;
        data_shared[tid + 1024*3 + 256*0] = *r4;
        data_shared[tid + 1024*0 + 256*1] = *r5;
        data_shared[tid + 1024*1 + 256*1] = *r6;
        data_shared[tid + 1024*2 + 256*1] = *r7;
        data_shared[tid + 1024*3 + 256*1] = *r8;
        data_shared[tid + 1024*0 + 256*2] = *r9;
        data_shared[tid + 1024*1 + 256*2] = *r10;
        data_shared[tid + 1024*2 + 256*2] = *r11;
        data_shared[tid + 1024*3 + 256*2] = *r12;
        data_shared[tid + 1024*0 + 256*3] = *r13;
        data_shared[tid + 1024*1 + 256*3] = *r14;
        data_shared[tid + 1024*2 + 256*3] = *r15;
        data_shared[tid + 1024*3 + 256*3] = *r16;
        __syncthreads();

    }

    ////Level 4096//////////////////////////////////////////////////////////

    index=tid*4;

    *r1  = data_shared[index + 1024*0 + 0];
    *r2  = data_shared[index + 1024*0 + 1];
    *r3  = data_shared[index + 1024*0 + 2];
    *r4  = data_shared[index + 1024*0 + 3];
    *r5  = data_shared[index + 1024*1 + 0];
    *r6  = data_shared[index + 1024*1 + 1];
    *r7  = data_shared[index + 1024*1 + 2];
    *r8  = data_shared[index + 1024*1 + 3];
    *r9  = data_shared[index + 1024*2 + 0];
    *r10 = data_shared[index + 1024*2 + 1];
    *r11 = data_shared[index + 1024*2 + 2];
    *r12 = data_shared[index + 1024*2 + 3];
    *r13 = data_shared[index + 1024*3 + 0];
    *r14 = data_shared[index + 1024*3 + 1];
    *r15 = data_shared[index + 1024*3 + 2];
    *r16 = data_shared[index + 1024*3 + 3];

    wi_c  = -2*FFT_Pi/4096;
    index = tid;
    StockhamR4_block(r1,  r2,  r3,  r4,  &wi, wi_c, index + 256*0);
    StockhamR4_block(r5,  r6,  r7,  r8,  &wi, wi_c, index + 256*1);
    StockhamR4_block(r9,  r10, r11, r12, &wi, wi_c, index + 256*2);
    StockhamR4_block(r13, r14, r15, r16, &wi, wi_c, index + 256*3);
	__syncthreads();

    ////end//////////////////////////////////////////////////////////////////
}



//////////////////////////////////////////////////////////////////
////                                                          ////
////  Stockham_FFT    N=4096                                  ////
////                                                          ////
////  all in one kernel , no need of temporary space          ////
////                                                          ////
////                                     Date:                ////
////                                            2016-10-16    ////
//////////////////////////////////////////////////////////////////


extern "C"
__global__ void __launch_bounds__(256) Stockham_FFT_4096_SM(Complex *data_in, int ifft_flag, int ifft_divideN_flag, int fftshift_front, int fftshift_back)
{
    Complex rA1, rA2, rA3, rA4, rA5, rA6, rA7, rA8, rA9, rA10, rA11, rA12, rA13, rA14, rA15, rA16;

    __shared__ Complex data_in_shared_A[4096];

    int bx  = blockIdx.x;
    int tid = threadIdx.x;

    float wi_c;
    Complex wi;

    int index = 0;
    int index_in = bx * 4096 + tid;

    if(fftshift_front==1)
    {
        rA1  = data_in[index_in + 1024*2 + 256*0];
        rA2  = data_in[index_in + 1024*3 + 256*0];
        rA3  = data_in[index_in + 1024*0 + 256*0];
        rA4  = data_in[index_in + 1024*1 + 256*0];
        rA5  = data_in[index_in + 1024*2 + 256*1];
        rA6  = data_in[index_in + 1024*3 + 256*1];
        rA7  = data_in[index_in + 1024*0 + 256*1];
        rA8  = data_in[index_in + 1024*1 + 256*1];
        rA9  = data_in[index_in + 1024*2 + 256*2];
        rA10 = data_in[index_in + 1024*3 + 256*2];
        rA11 = data_in[index_in + 1024*0 + 256*2];
        rA12 = data_in[index_in + 1024*1 + 256*2];
        rA13 = data_in[index_in + 1024*2 + 256*3];
        rA14 = data_in[index_in + 1024*3 + 256*3];
        rA15 = data_in[index_in + 1024*0 + 256*3];
        rA16 = data_in[index_in + 1024*1 + 256*3];
    }
    else
    {
        rA1  = data_in[index_in + 1024*0 + 256*0];
        rA2  = data_in[index_in + 1024*1 + 256*0];
        rA3  = data_in[index_in + 1024*2 + 256*0];
        rA4  = data_in[index_in + 1024*3 + 256*0];
        rA5  = data_in[index_in + 1024*0 + 256*1];
        rA6  = data_in[index_in + 1024*1 + 256*1];
        rA7  = data_in[index_in + 1024*2 + 256*1];
        rA8  = data_in[index_in + 1024*3 + 256*1];
        rA9  = data_in[index_in + 1024*0 + 256*2];
        rA10 = data_in[index_in + 1024*1 + 256*2];
        rA11 = data_in[index_in + 1024*2 + 256*2];
        rA12 = data_in[index_in + 1024*3 + 256*2];
        rA13 = data_in[index_in + 1024*0 + 256*3];
        rA14 = data_in[index_in + 1024*1 + 256*3];
        rA15 = data_in[index_in + 1024*2 + 256*3];
        rA16 = data_in[index_in + 1024*3 + 256*3];
    }
    //Move your code of previous kernel here , if possible.-----------------------------





    //---------------------------------------------------------------------------------

    if(ifft_flag == 1)
    {
        rA1.y  = -rA1.y ;
        rA2.y  = -rA2.y ;
        rA3.y  = -rA3.y ;
        rA4.y  = -rA4.y ;
        rA5.y  = -rA5.y ;
        rA6.y  = -rA6.y ;
        rA7.y  = -rA7.y ;
        rA8.y  = -rA8.y ;
        rA9.y  = -rA9.y ;
        rA10.y = -rA10.y;
        rA11.y = -rA11.y;
        rA12.y = -rA12.y;
        rA13.y = -rA13.y;
        rA14.y = -rA14.y;
        rA15.y = -rA15.y;
        rA16.y = -rA16.y;
    }


    Stockham_4096_block(&rA1, &rA2, &rA3, &rA4, &rA5, &rA6, &rA7, &rA8,
                        &rA9, &rA10, &rA11, &rA12, &rA13, &rA14, &rA15, &rA16,
                        data_in_shared_A);

    if(ifft_flag == 1)
    {
        rA1.y  = -rA1.y ;
        rA2.y  = -rA2.y ;
        rA3.y  = -rA3.y ;
        rA4.y  = -rA4.y ;
        rA5.y  = -rA5.y ;
        rA6.y  = -rA6.y ;
        rA7.y  = -rA7.y ;
        rA8.y  = -rA8.y ;
        rA9.y  = -rA9.y ;
        rA10.y = -rA10.y;
        rA11.y = -rA11.y;
        rA12.y = -rA12.y;
        rA13.y = -rA13.y;
        rA14.y = -rA14.y;
        rA15.y = -rA15.y;
        rA16.y = -rA16.y;
    }

    if( ifft_divideN_flag && ifft_flag )
    {
        rA1.x  = rA1.x  / 4096 ;
        rA2.x  = rA2.x  / 4096 ;
        rA3.x  = rA3.x  / 4096 ;
        rA4.x  = rA4.x  / 4096 ;
        rA5.x  = rA5.x  / 4096 ;
        rA6.x  = rA6.x  / 4096 ;
        rA7.x  = rA7.x  / 4096 ;
        rA8.x  = rA8.x  / 4096 ;
        rA9.x  = rA9.x  / 4096 ;
        rA10.x = rA10.x / 4096 ;
        rA11.x = rA11.x / 4096 ;
        rA12.x = rA12.x / 4096 ;
        rA13.x = rA13.x / 4096 ;
        rA14.x = rA14.x / 4096 ;
        rA15.x = rA15.x / 4096 ;
        rA16.x = rA16.x / 4096 ;

        rA1.y  = rA1.y  / 4096 ;
        rA2.y  = rA2.y  / 4096 ;
        rA3.y  = rA3.y  / 4096 ;
        rA4.y  = rA4.y  / 4096 ;
        rA5.y  = rA5.y  / 4096 ;
        rA6.y  = rA6.y  / 4096 ;
        rA7.y  = rA7.y  / 4096 ;
        rA8.y  = rA8.y  / 4096 ;
        rA9.y  = rA9.y  / 4096 ;
        rA10.y = rA10.y / 4096 ;
        rA11.y = rA11.y / 4096 ;
        rA12.y = rA12.y / 4096 ;
        rA13.y = rA13.y / 4096 ;
        rA14.y = rA14.y / 4096 ;
        rA15.y = rA15.y / 4096 ;
        rA16.y = rA16.y / 4096 ;
    }


    //Move your code of next kernel here , if possible.-----------------------------





    //-----------------------------------------------------------------------------

    if(fftshift_back==1)
    {
        data_in[index_in + 1024*2 + 256*0] = rA1 ;
        data_in[index_in + 1024*3 + 256*0] = rA2 ;
        data_in[index_in + 1024*0 + 256*0] = rA3 ;
        data_in[index_in + 1024*1 + 256*0] = rA4 ;
        data_in[index_in + 1024*2 + 256*1] = rA5 ;
        data_in[index_in + 1024*3 + 256*1] = rA6 ;
        data_in[index_in + 1024*0 + 256*1] = rA7 ;
        data_in[index_in + 1024*1 + 256*1] = rA8 ;
        data_in[index_in + 1024*2 + 256*2] = rA9 ;
        data_in[index_in + 1024*3 + 256*2] = rA10;
        data_in[index_in + 1024*0 + 256*2] = rA11;
        data_in[index_in + 1024*1 + 256*2] = rA12;
        data_in[index_in + 1024*2 + 256*3] = rA13;
        data_in[index_in + 1024*3 + 256*3] = rA14;
        data_in[index_in + 1024*0 + 256*3] = rA15;
        data_in[index_in + 1024*1 + 256*3] = rA16;
    }
    else
    {
        data_in[index_in + 1024*0 + 256*0] = rA1 ;
        data_in[index_in + 1024*1 + 256*0] = rA2 ;
        data_in[index_in + 1024*2 + 256*0] = rA3 ;
        data_in[index_in + 1024*3 + 256*0] = rA4 ;
        data_in[index_in + 1024*0 + 256*1] = rA5 ;
        data_in[index_in + 1024*1 + 256*1] = rA6 ;
        data_in[index_in + 1024*2 + 256*1] = rA7 ;
        data_in[index_in + 1024*3 + 256*1] = rA8 ;
        data_in[index_in + 1024*0 + 256*2] = rA9 ;
        data_in[index_in + 1024*1 + 256*2] = rA10;
        data_in[index_in + 1024*2 + 256*2] = rA11;
        data_in[index_in + 1024*3 + 256*2] = rA12;
        data_in[index_in + 1024*0 + 256*3] = rA13;
        data_in[index_in + 1024*1 + 256*3] = rA14;
        data_in[index_in + 1024*2 + 256*3] = rA15;
        data_in[index_in + 1024*3 + 256*3] = rA16;
    }
}