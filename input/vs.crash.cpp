/*****************************************************************************
 * vs.c: VapourSynth-Vpy input
 *****************************************************************************
 *
 *****************************************************************************/
extern "C"{
#include "stdint.h"
#include "x264.h"
#include "input.h"
}

#include "vs-include\VapourSynth.h"
#include "vs-include\VSHelper.h"
#include "vs-include\VSScript.h"
#include <iostream>
#include <string>
#include <condition_variable>
#include <codecvt>
#include <algorithm>
#include <locale>
#include <sstream>
#include <atomic> 


#define SIMD_WIDTH 8

#ifdef SIMD_WIDTH
#include "emmintrin.h"

template<int SIMDWIDTH>
inline void modify2x264StyleDepth(uint16_t *lp,const int lshift)
{
	
	
	if(SIMDWIDTH==8)
	{
		auto xmm0 = _mm_loadu_si128((__m128i*)lp);
		auto xmm1 = _mm_slli_epi16(xmm0,lshift);
		_mm_stream_si128((__m128i*)lp,xmm1);
	}
	else
	{
		for(int i=0;i<SIMDWIDTH;i++)
		{
			lp[i] = lp[i] << lshift;
		}
	}
}
#endif


typedef std::wstring nstring;
#define NSTRING(x) L##x
std::string nstringToUtf8(const nstring &s) {
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> conversion;
	return conversion.to_bytes(s);
}

std::wstring s2ws(const std::string& s)  
{  
    setlocale(LC_ALL, "chs");   
    const char* _Source = s.c_str();  
    size_t _Dsize = s.size() + 1;  
    wchar_t *_Dest = new wchar_t[_Dsize];  
    wmemset(_Dest, 0, _Dsize);  
    mbstowcs(_Dest,_Source,_Dsize);  
    std::wstring result = _Dest;  
    delete []_Dest;  
    setlocale(LC_ALL, "C");  
    return result;  
}



#define FAIL_IF_ERROR( cond, ... ) FAIL_IF_ERR( cond, "vs", __VA_ARGS__ )

class semaphore
{
	std::mutex mtx;
	std::condition_variable cv;
	std::atomic<int> counter;

public:
	semaphore(int count)
	{
		counter = count;
	}

	void wait()
	{
		std::unique_lock<std::mutex> locker(mtx);
		cv.wait(locker, [this]{return counter>0;});
		counter--;
	}
	void release()
	{
		//mtx.try_lock();
		counter++;
		//mtx.unlock();
		cv.notify_one();
	}
};



typedef struct
{
    const VSAPI *vsapi = nullptr;
	VSScript *se = nullptr;
	VSNodeRef *node = nullptr;
	semaphore* sea;
    int bit_depth;
} vs_hnd_t;

typedef struct
{
    int frameNumber;
} vs_usr_data;



struct FrameWaiter {
	std::condition_variable a;
    std::mutex b;
    const VSFrameRef *r;
};

static void VS_CC frameWaiterCallback(void *userData, const VSFrameRef *frame, int n, VSNodeRef *node, const char *errorMsg) 
{
    FrameWaiter *g = static_cast<FrameWaiter *>(userData);
    std::lock_guard<std::mutex> l(g->b);
    g->r = frame;
    g->a.notify_one();
}


static int open_file( char *psz_filename, hnd_t *p_handle, video_info_t *info, cli_input_opt_t *opt )
{
	vs_hnd_t *h = new vs_hnd_t;
    if( !h )
        return -1;
	
	FILE *fh = x264_fopen(psz_filename, "rb");
	if (!fh)
		return -1;
	int b_regular = x264_is_regular_file(fh);
	fclose(fh);
	FAIL_IF_ERROR(!b_regular, "VS input is incompatible with non-regular file `%s'\n", psz_filename);

	FAIL_IF_ERROR(!vsscript_init(), "Failed to initialize VapourSynth environment\n");
	h->vsapi = vsscript_getVSApi();
	if (!h->vsapi) {
		fprintf(stderr, "Failed to get VapourSynth API pointer\n");
		vsscript_finalize();
		return -1;
	}

	// Should always succeed
	if (vsscript_createScript(&h->se)) {
		fprintf(stderr, "Script environment initialization failed:\n%s\n", vsscript_getError(h->se));
		vsscript_freeScript(h->se);
		vsscript_finalize();
		return -1;
	}

	std::string strfilename = psz_filename;
	nstring scriptFilename = s2ws(strfilename);
	if (vsscript_evaluateFile(&h->se, nstringToUtf8(scriptFilename).c_str(), efSetWorkingDir)) {
		fprintf(stderr, "Script evaluation failed:\n%s\n", vsscript_getError(h->se));
		vsscript_freeScript(h->se);
		vsscript_finalize();
		return -1;
	}

	h->node = vsscript_getOutput(h->se, 0);//outputIndex
	if (!h->node) {
		fprintf(stderr, "Failed to retrieve output node. Invalid index specified?\n");
		vsscript_freeScript(h->se);
		vsscript_finalize();
		return -1;
	}

	const VSCoreInfo *vsInfo = h->vsapi->getCoreInfo(vsscript_getCore(h->se));
    h->sea = new semaphore(vsInfo->numThreads);

	const VSVideoInfo *vi = h->vsapi->getVideoInfo(h->node);
	if (vi->format->colorFamily != cmYUV) {
		fprintf(stderr, "Can only read YUV format clips"); 
		h->vsapi->freeNode(h->node);
		vsscript_freeScript(h->se);
		vsscript_finalize();
        return -1;
    }
	
	if (!isConstantFormat(vi)) {
		fprintf(stderr, "Cannot output clips with varying dimensions\n");
		h->vsapi->freeNode(h->node);
		vsscript_freeScript(h->se);
		vsscript_finalize();
		return -1;
	}

	info->width = vi->width;
	info->height = vi->height;
	info->fps_num = vi->fpsNum;
	info->fps_den = vi->fpsDen;
	info->thread_safe = 1;
	info->num_frames = vi->numFrames;

	if (vi->format->subSamplingW == 1 && vi->format->subSamplingH == 1)
		info->csp = X264_CSP_I420;
    else if (vi->format->subSamplingW == 1 && vi->format->subSamplingH == 0)
        info->csp = X264_CSP_I422;
    else if (vi->format->subSamplingW == 0 && vi->format->subSamplingH == 0)
        info->csp = X264_CSP_I444;


	h->bit_depth = vi->format->bitsPerSample;
	if (h->bit_depth > 8)
	{
		info->csp |= X264_CSP_HIGH_DEPTH;
	}

	*p_handle = (void*)h;
    return 0;
}

static int read_frame( cli_pic_t *pic, hnd_t handle, int i_frame )
{
	vs_hnd_t *h = (vs_hnd_t*)handle;


	const int rgbRemap[] = { 1, 2, 0 };
	
	h->sea->wait();
	FrameWaiter fw;
	std::unique_lock<std::mutex> locker(fw.b);
	h->vsapi->getFrameAsync(i_frame,h->node,frameWaiterCallback,&fw);
	fw.a.wait(locker);
	h->sea->release();
	const VSFrameRef *frame = fw.r;
	
	const VSFormat *fi = h->vsapi->getFrameFormat(frame);
	
	for (int rp = 0; rp < fi->numPlanes; rp++) 
	{
		int p = (fi->colorFamily == cmRGB) ? rgbRemap[rp] : rp;
		int stride = h->vsapi->getStride(frame, p);
	
        const uint8_t *readPtr = h->vsapi->getReadPtr(frame, p);
		int width = h->vsapi->getFrameWidth(frame, p);
        int height = h->vsapi->getFrameHeight(frame, p);
		int rowSize = width * fi->bytesPerSample;
		
		if(rowSize != stride)
		{
			size_t lpdest = 0;
			for (int y = 0; y < height; y++) 
			{
				memcpy(pic->img.plane[rp] + lpdest,readPtr,rowSize);
				lpdest += rowSize;
				readPtr += stride;
			}
		}
		else
		{
			int planeSize = height * rowSize;
			memcpy(pic->img.plane[rp],readPtr,planeSize);
		}

		

		if(h->bit_depth & 7)
		{
			uint16_t *plane = (uint16_t*)pic->img.plane[rp];
			const size_t picPixelCount = width * height;
			const int lshift = 16 - h->bit_depth;
#ifndef SIMD_WIDTH
			for( uint64_t j = 0; j < picPixelCount; j++ )
				plane[j] = plane[j] << lshift;
#else
			uint64_t j = 0;
			
			for( ; j < picPixelCount; j+=SIMD_WIDTH )
			{
				modify2x264StyleDepth<SIMD_WIDTH>(plane+j,lshift);
			}
			if(j != picPixelCount)
			{
				for( j=j-SIMD_WIDTH ; j < picPixelCount; j++ )
					plane[j] = plane[j] << lshift;
				
			}
#endif
		}
    }
	h->vsapi->freeFrame(frame);

    return 0;
}

static int release_frame( cli_pic_t *pic, hnd_t handle )
{
	vs_hnd_t *h = (vs_hnd_t*)handle;
    return 0;
}

static int picture_alloc( cli_pic_t *pic, hnd_t handle, int csp, int width, int height )
{
    return x264_cli_pic_alloc( pic, csp, width, height );
}

static void picture_clean( cli_pic_t *pic, hnd_t handle )
{
	x264_cli_pic_clean(pic);
}

static int close_file( hnd_t handle )
{
	vs_hnd_t *h = (vs_hnd_t*)handle;
	h->vsapi->freeNode(h->node);
	delete h->sea;
	vsscript_freeScript(h->se);
	vsscript_finalize();
	delete h;
    return 0;
}

const cli_input_t vs_input = { open_file, picture_alloc, read_frame, release_frame, picture_clean, close_file };
