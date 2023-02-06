#pragma once
#include <vector>
#include <utils/compilation/debug.h>
#include <stdexcept>

#include <thrust/system/cuda/memory.h>
//#include <thrust/system/cuda/pinned_allocator.h>

namespace utils::CUDA::containers
	{
	template <typename T>
	using pinned_allocator = thrust::system::cuda::universal_allocator<T>;

	template <typename T>
	class device_vector;

	/// <summary>
	/// Device side vector abstraction to allow ranged for and iterators usage on the device side.
	/// Can be constructed from pinned memory or returned from a global memory vector manager (see utils::CUDA::vector)
	/// Note: does NOT have ownership of any resource. Ownership is managed exclusively by the host side.
	/// </summary>
	/// <typeparam name="T">The type stored in the vector.</typeparam>
	template <typename T>
	class vector_observer
		{
		template <typename T>
		friend class device_vector;
		public:
			using value_type             = T;
			using size_type              = size_t;
			using reference              = T&;
			using const_reference        = const reference;
			using pointer                = T*;
			using const_pointer          = const pointer;

			__device__ size_t size() const noexcept { return _size; }

			__device__       T& operator[](size_t index)       noexcept { return _arr[index]; }
			__device__ const T& operator[](size_t index) const noexcept { return _arr[index]; }

			__device__       T* begin()       noexcept { return _arr; }
			__device__       T* end  ()       noexcept { return _arr + _size; }
			__device__ const T* begin() const noexcept { return _arr; }
			__device__ const T* end  () const noexcept { return _arr + _size; }

			/// <summary>
			/// Construct from a vector which data is allocated in pinned memory
			/// </summary>
			/// <param name="pinned_vector">A vector which exists in pinned memory</param>
			__host__ vector_observer(std::vector<T, pinned_allocator<T>>& pinned_vector) : _arr{pinned_vector.data()}, _size{pinned_vector.size()} {}

		protected:
			//Used to construct from CUDA::vector, points to global memory
			__host__ vector_observer(T* device_array, size_t size) : _arr{device_array}, _size{size} {}

			T* _arr;
			size_t _size;
		};

	/// <summary>
	/// Host side RAII manager for a vector in device global memory.
	/// </summary>
	/// <typeparam name="T">The type stored in the vector.</typeparam>
	template <typename T>
	class device_vector
		{
		public:
			using value_type             = T;
			using size_type              = size_t;
			using reference              = T&;
			using const_reference        = const reference;
			using pointer                = T*;
			using const_pointer          = const pointer;

			//TODO copy constructor copying data in CUDA

			/// <summary>
			/// Allocates a vector that only exists on the device's global memory.
			/// </summary>
			/// <param name="size">The size of the vector.</param>
			device_vector(size_t size) : _size{size}
				{
				utils::CUDA::throw_if_failed(cudaMalloc((void**)&arr_ptr, size * sizeof(T)));
				}

			/// <summary>
			/// Allocates a vector that only exists on the device's global memory and initializes it with the values of the given host-side vector.
			/// </summary>
			/// <param name="vec">The host side vector to take data from.</param>
			device_vector(const std::vector<T>& vec) : device_vector{vec.size()} { from(vec); }

			/// <summary>
			/// Retrieve a device_vector to use in kernels which wraps the globally allocated memory.
			/// </summary>
			/// <returns></returns>
			      vector_observer<T> get_vector_observer()       noexcept { return {arr_ptr, size()}; }
			const vector_observer<T> get_vector_observer() const noexcept { return {arr_ptr, size()}; }

			/// <summary>
			/// Copies data from host to device. Assumes both have the same size (checked in debug mode, unchecked in release)
			/// </summary>
			void from(const std::vector<T>& vec) utils_if_release(noexcept)
				{
				if constexpr (utils::compilation::debug)
					{
					if (vec.size() != size()) { throw std::out_of_range{"Trying to copy vector from CPU to GPU, but sizes don't match."}; }
					}
				utils::CUDA::throw_if_failed(cudaMemcpy(arr_ptr, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice));
				}

			/// <summary>
			/// Copies data from device to host. Assumes both have the same size (checked in debug mode, unchecked in release)
			/// </summary>
			void to(std::vector<T>& vec) const utils_if_release(noexcept)
				{
				if constexpr (utils::compilation::debug)
					{
					if (vec.size() != size()) { throw std::out_of_range{"Trying to copy vector from CPU to GPU, but sizes don't match."}; }
					}
				utils::CUDA::throw_if_failed(cudaMemcpy(vec.data(), arr_ptr, vec.size() * sizeof(T), cudaMemcpyDeviceToHost));
				}

			size_t size() const noexcept { return _size; }

			~device_vector() { if (arr_ptr) { utils::CUDA::throw_if_failed(cudaFree(arr_ptr)); } }

		protected:
			T* arr_ptr{nullptr};

			const size_t _size{0};
		};


	/// <summary>
	/// Abstraction for a vector in shared memory.
	/// </summary>
	/// <typeparam name="T">The type stored in the vector.</typeparam>
	template <typename T>
	class shared_vector
		{
		public:
			using value_type             = T;
			using size_type              = size_t;
			using reference              = T&;
			using const_reference        = const reference;
			using pointer                = T*;
			using const_pointer          = const pointer;

			/// <summary>
			/// Initializes with an array to allocated shared memory space.
			/// Note: smem available must be equal to size * sizeof(T)!!!
			/// </summary>
			/// <param name="data">Address of shared memory</param>
			/// <param name="block_size">Size of the shared vector</param>
			__device__ shared_vector(T* data, vector_observer<T>& source, size_t size) : _arr{data}, source{source}, _size{size}, _to_end{size}, _necessary_iterations{source.size() / size}
				{}
			__device__ size_t size() const noexcept { return _size; }

			__device__       T& operator[](size_t index)       noexcept { return _arr[index]; }
			__device__ const T& operator[](size_t index) const noexcept { return _arr[index]; }

			__device__ T* begin() { return _arr; }
			__device__ T* end() { return _arr + _to_end; }
			__device__ const T* begin() const noexcept { return _arr; }
			__device__ const T* end()   const noexcept { return _arr + _to_end; }

			/// <summary>
			/// Loads the source vector in chunks to shared memory and calls the passed callable after each chunk is loaded.
			/// Assumes to be called by all threads in the block, and assumes that the shared vector size is equal to the amount of threads in the block.
			/// Will take care about synchronizing all threads after loading a chunk and after the callable is run.
			/// </summary>
			template <typename T>
			__device__ void load(size_t thread_id, T callable)
				{
				while (load_next(thread_id))
					{
					__syncthreads();
					callable();
					__syncthreads();
					}
				if (load_last(thread_id))
					{
					__syncthreads();
					callable();
					__syncthreads();
					}
				}

		protected:
			T* _arr;
			vector_observer<T>& source;
			const size_t _size;
			size_t _to_end;

			size_t iteration{0};
			const size_t _necessary_iterations;

			/// <summary>
			/// Loads part of a device-side vector into the shared vector. 
			/// Multiple calls will load the next parts of the source vector until it has all been loaded.
			/// Note: Assumes it is being called by all the threads in a block!!!
			/// Node: Synchronization does NOT happen internally. Use __synchthreads in the caller.
			/// </summary>
			/// <param name="source">The vector to take data from</param>
			/// <param name="thread_id">The id of the current thread</param>
			/// <returns>true until the whole vector is loaded, or if the last elements count is smaller than the shared vector size. Use load_last in that case.</returns>
			__device__ bool load_next(size_t thread_id)
				{
				size_t index{(iteration * size()) + thread_id};

				(*this)[thread_id] = source[index];

				iteration++;
				return iteration < necessary_iterations(); //Same for all threads
				}

			__device__ size_t necessary_iterations() const { return _necessary_iterations; }

			/// <summary>
			/// Call after load_next returns false. Loads eventual elements left in the source vector which hadn't been loaded in shared memory. 
			/// Updates the end() iterator so that it reaches the last element loaded from the source, instead of the last shared vector index.
			/// Note: Assumes it is being called by all the threads in a block!!!
			/// Node: Synchronization does NOT happen internally. Use __synchthreads in the caller.
			/// </summary>
			/// <param name="thread_id"></param>
			/// <returns>true if some elements were loaded, false otherwise.</returns>
			__device__ bool load_last(size_t thread_id)
				{
				size_t index{(iteration * size()) + thread_id};

				if (_to_end = source.size() % size()) //Same for all threads
					{
					bool exists = index < source.size();
					if (exists) //Different across threads
						{
						(*this)[thread_id] = source[index];
						}
					return exists;
					}
				return false;
				}
		};
	}