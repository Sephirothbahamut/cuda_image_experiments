
#include <utils/math/vec2.h>
#include <utils/compilation/CUDA.h>
#include <utils/containers/matrix_dyn.h>

#include "cuda.h"
#include "vector.cuh"

namespace utils::CUDA::containers
	{
	template <typename T, utils::containers::matrix_memory MEMORY_LAYOUT>
	class device_matrix;

	template <typename T, utils::containers::matrix_memory MEMORY_LAYOUT = utils::containers::matrix_memory::width_first>
	class matrix_observer : public vector_observer<T>
		{
		template <typename T, utils::containers::matrix_memory MEMORY_LAYOUT>
		friend class device_matrix;
		public:
			using value_type             = vector_observer<T>::value_type     ;
			using size_type              = vector_observer<T>::size_type      ;
			using reference              = vector_observer<T>::reference      ;
			using const_reference        = vector_observer<T>::const_reference;
			using pointer                = vector_observer<T>::pointer        ;
			using const_pointer          = vector_observer<T>::const_pointer  ;
			using coords_type = utils::math::vec2s;
			inline static constexpr utils::containers::matrix_memory memory_layout{MEMORY_LAYOUT};
			
			__device__ const size_type   width () const noexcept { return _sizes.x; }
			__device__ const size_type   height() const noexcept { return _sizes.y; }
			__device__ const coords_type sizes () const noexcept { return _sizes; }

			__device__       T& operator[](coords_type coords)       noexcept { return operator[](get_index(coords)); }
			__device__ const T& operator[](coords_type coords) const noexcept { return operator[](get_index(coords)); }
			
			__device__ size_type get_index(coords_type coords) const noexcept { get_index(coords.x, coords.y); }
			__device__ size_type get_index(size_type x, size_type y) const noexcept
				{
				if constexpr (memory_layout == utils::containers::matrix_memory::width_first) { return x + (y * _sizes.x); }
				else { return y + (x * _sizes.y); }
				}
			__device__ size_type   get_x     (size_type index) const noexcept { if constexpr (memory_layout == utils::containers::matrix_memory::width_first) { return index % width(); } else { return index / height(); } }
			__device__ size_type   get_y     (size_type index) const noexcept { if constexpr (memory_layout == utils::containers::matrix_memory::width_first) { return index / width(); } else { return index % height(); } }
			__device__ coords_type get_coords(size_type index) const noexcept { return { get_x(index), get_y(index) }; }

		protected:
			__host__ matrix_observer(T* device_array, coords_type sizes) : vector_observer<T>{device_array}, _sizes{sizes} {}

			coords_type _sizes;
		};
	
	template <typename T, utils::containers::matrix_memory MEMORY_LAYOUT = utils::containers::matrix_memory::width_first>
	class device_matrix : public device_vector<T>
		{
		public:
			using value_type             = device_vector<T>::value_type     ;
			using size_type              = device_vector<T>::size_type      ;
			using reference              = device_vector<T>::reference      ;
			using const_reference        = device_vector<T>::const_reference;
			using pointer                = device_vector<T>::pointer        ;
			using const_pointer          = device_vector<T>::const_pointer  ;
			using coords_type = utils::math::vec2s;
			inline static constexpr utils::containers::matrix_memory memory_layout{MEMORY_LAYOUT};

			device_matrix(coords_type sizes) : _sizes{sizes}, device_vector<T>{sizes.x * sizes.y} {}
			
			const size_type   width () const noexcept { return _sizes.x; }
			const size_type   height() const noexcept { return _sizes.y; }
			const coords_type sizes () const noexcept { return _sizes; }

			      matrix_observer<T> get_matrix_observer()       noexcept { return {device_vector<T>::arr_ptr, sizes()}; }
			const matrix_observer<T> get_matrix_observer() const noexcept { return {device_vector<T>::arr_ptr, sizes()}; }

			device_matrix(const utils::containers::matrix_dyn<T>& mat) : device_matrix{mat.size()} { from(mat); }

			void from(const utils::containers::matrix_dyn<T>& mat) utils_if_release(noexcept)
				{
				if constexpr (utils::compilation::debug)
					{
					if (mat.sizes() != sizes()) { throw std::out_of_range{"Trying to copy vector from CPU to GPU, but sizes don't match."}; }
					}
				utils::CUDA::throw_if_failed(cudaMemcpy(device_vector<T>::arr_ptr, mat.data(), mat.size() * sizeof(T), cudaMemcpyHostToDevice));
				}

			void to(utils::containers::matrix_dyn<T>& mat) const utils_if_release(noexcept)
				{
				if constexpr (utils::compilation::debug)
					{
					if (mat.sizes() != sizes()) { throw std::out_of_range{"Trying to copy vector from CPU to GPU, but sizes don't match."}; }
					}
				utils::CUDA::throw_if_failed(cudaMemcpy(mat.data(), device_vector<T>::arr_ptr, mat.size() * sizeof(T), cudaMemcpyDeviceToHost));
				}

		protected:
			coords_type _sizes{0, 0};
		};

	template <typename T, utils::containers::matrix_memory MEMORY_LAYOUT = utils::containers::matrix_memory::width_first>
	class shared_matrix : shared_vector<T>
		{
		public:
			using value_type             = shared_vector<T>::value_type     ;
			using size_type              = shared_vector<T>::size_type      ;
			using reference              = shared_vector<T>::reference      ;
			using const_reference        = shared_vector<T>::const_reference;
			using pointer                = shared_vector<T>::pointer        ;
			using const_pointer          = shared_vector<T>::const_pointer  ;
			using coords_type = utils::math::vec2s;
			inline static constexpr utils::containers::matrix_memory memory_layout{MEMORY_LAYOUT};

			__device__ shared_matrix(T* data, vector_observer<T>& source, size_t size) : shared_vector<T>{data, source, size} {}
			__device__ const size_type   width () const noexcept { return _sizes.x; }
			__device__ const size_type   height() const noexcept { return _sizes.y; }
			__device__ const coords_type sizes () const noexcept { return _sizes; }

		protected:
			const coords_type _sizes;
		};


	}