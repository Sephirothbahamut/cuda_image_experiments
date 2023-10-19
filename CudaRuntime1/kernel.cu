
#include "Header.cuh"
//#include "cuda.h"

//#include <utils/math/geometry/shapes.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"

#include <limits>

#ifdef __NVCC__
#define utils_cuda_available __device__ __host__
#define utils_is_CUDA
#define utils_if_CUDA(x) x
#else
#define utils_cuda_available
#define utils_if_CUDA(x)
#endif

template <typename T, T FULL_VALUE>
struct type_based_numeric_range;

namespace concepts
	{
	template <typename T>
	concept type_based_numeric_range = std::same_as<T, ::type_based_numeric_range<typename T::value_type, T::full_value>>;
	}

template <typename T, T FULL_VALUE = std::floating_point<T> ? static_cast<T>(1.) : std::numeric_limits<T>::max() >
struct type_based_numeric_range
	{
	using value_type = T;
	inline static constexpr const T full_value{FULL_VALUE};

	template <concepts::type_based_numeric_range other>
	utils_cuda_available
	static other::value_type cast_to(value_type value) noexcept
		{
		if constexpr (std::same_as<value_type, typename other::value_type>)
			{
			if constexpr (full_value == other::full_value) { return value; }
			}

		using tmp_t = std::conditional_t<(std::numeric_limits<value_type>::max() > std::numeric_limits<typename other::value_type>::max()), value_type, typename other::value_type>;
		tmp_t tmp{(static_cast<tmp_t>(value) / static_cast<tmp_t>(full_value)) * static_cast<tmp_t>(other::full_value)};
		return static_cast<other::value_type>(tmp);
		}
	};



template <typename T>
struct my_struct;

namespace concepts
	{
	template <typename T>
	concept my_struct = std::same_as<T, ::my_struct<typename T::value_type>>;
	}

template <typename T>
struct my_struct
	{
	using value_type = T;
	using range = type_based_numeric_range<T>;
	T value;

	utils_cuda_available constexpr my_struct() = default;
	utils_cuda_available constexpr my_struct(T value) : value{value} {};

	template <concepts::my_struct other_t>
	utils_cuda_available
	constexpr my_struct(const other_t& other) noexcept
		{
		value = other_t::range::template cast_to<range>(other.value);
		}
	};

/*
__global__ void kernel(float val)
	{
	//ok
	auto v{type_based_numeric_range<float, 1.f>::cast_to<type_based_numeric_range<float, 360.f>>(val)};
	printf("\nFull values: \"%f\", \"%f\"\n", type_based_numeric_range<float, 1.f>::full_value, type_based_numeric_range<float, 360.f>::full_value);
	printf("\nInput: \"%f\", cast result: \"%f\"\n", val, v);

	//doesn't compile
	my_struct<float> mystr_float{val};
	my_struct<int> mystr_int{mystr_float};

	printf("\nInput: \"%f\", cast result: \"%i\"\n", mystr_float.value, mystr_int.value);
	}
/*/
void kernel(float val)
	{
	//ok
	auto v{type_based_numeric_range<float, 1.f>::cast_to<type_based_numeric_range<float, 360.f>>(val)};
	printf("\nFull values: \"%f\", \"%f\"\n", type_based_numeric_range<float, 1.f>::full_value, type_based_numeric_range<float, 360.f>::full_value);
	printf("\nInput: \"%f\", cast result: \"%f\"\n", val, v);

	//doesn't compile
	my_struct<float> mystr_float{val};
	my_struct<int> mystr_int{mystr_float};

	printf("\nInput: \"%f\", cast result: \"%i\"\n", mystr_float.value, mystr_int.value);
	}
/**/

void test_func()
	{
	kernel/*<<<1, 3, 0>>>*/(.5f);

	cudaDeviceSynchronize();

	char c; std::cin >> c;
	}