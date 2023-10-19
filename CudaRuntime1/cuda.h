#pragma once

#include <utils/compilation/CUDA.h>
#include <stdexcept>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace utils::CUDA
	{
	struct error : std::runtime_error { using std::runtime_error::runtime_error; };

	/// <summary>
	/// Prints eventual CUDA errors before terminating the program
	/// </summary>
	inline void throw_if_failed(cudaError_t err)
		{
		if (err != cudaSuccess)
			{
			throw error{cudaGetErrorString(err)};
			}
		}
	}

// Makes sure intellisense recognizes __syncthreads as a defined function
#ifdef __INTELLISENSE__
void __syncthreads() {};
#endif

namespace utils::CUDA
	{
	/// <summary>
	/// Wrapper for CUDA's kernel calls. Allows intellisense to report eventual errors in the kernel call parameters before compile time.
	/// </summary>
	template <auto device_function>
	struct launcher
		{
		size_t blocks_n{1};
		size_t threads_n{1024};
		size_t smem_size{0};

		/// <summary>
		/// Launch the kernel with the given parameters.
		/// </summary>
		template <typename ...Args>
		void operator()(Args&&... args) const noexcept
			{
#ifndef __INTELLISENSE__
			device_function<<<blocks_n, threads_n, smem_size>>>(std::forward<Args>(args)...);
#endif
			}
		};
	}