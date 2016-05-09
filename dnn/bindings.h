#pragma once

#include <stdio.h>


#ifdef DNN_DLL
	#ifdef DNN_EXPORTS
		#define DNN_API __declspec(dllexport)
	#else
		#define DNN_API __declspec(dllimport)
	#endif
#else		
	#define DNN_API extern 
#endif /* DNN_DLL */

extern "C" {
	
	DNN_API void write_time_series(const double* data, int nrows, int ncols,  const char* label, const char* dst_file);

	DNN_API void run_iaf_network(const char* config, const double* data, int nrows, int ncols, const char* dst_file);
	
}