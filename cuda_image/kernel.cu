
//#include <iostream>
//
//#include <utils/compilation/CUDA.h>
//#include <utils/oop/disable_move_copy.h>
//#include <utils/graphics/colour.h>
//
//#include <utils/math/geometry/shapes.h>
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include "matrix.cuh"
//#include "cuda.h"
//
//#include <SFML/Graphics.hpp>
//
//class output_manager
//	{
//	public:
//		class indenter;
//		friend class indenter;
//		class indenter
//			{
//			friend class output_manager;
//			public:
//				indenter& operator=(indenter&& move)
//					{
//					release();
//					valid = move.valid;
//					move.valid = false;
//					}
//
//				void release() noexcept 
//					{
//					if(valid) { output_manager._out(); }
//					valid = false;
//					}
//
//				~indenter() { if (valid) { output_manager._out(); } }
//
//			private:
//				indenter(output_manager& output_manager) : output_manager{output_manager} { output_manager._in(); }
//				output_manager& output_manager;
//				bool valid{true};
//			};
//
//		void operator()(const std::string& string) const noexcept
//			{
//			std::cout << indent_str << string << std::endl;
//			}
//
//		[[nodiscard]] indenter in() { return indenter{*this}; }
//
//	private:
//		std::string indent_str{};
//		
//		void _in () { indent_str += "    "; }
//		void _out() { indent_str.resize(indent_str.size() - 4); }
//	};
//output_manager cout;
//
////enum class side { ll, up, rr, dw, no };
////
////namespace cutils
////	{
////	__global__ void draw_border(utils::CUDA::containers::device_vector<rgba_f> in, utils::CUDA::containers::device_vector<rgba_f> out, vec2s size, size_t thickness)
////		{
////		size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};
////		size_t x{index % size.x};
////		size_t y{index / size.x};
////
////		bool is_ll{x <           thickness};
////		bool is_rr{x >= size.x - thickness};
////		bool is_up{y <           thickness};
////		bool is_dw{y >= size.y - thickness};
////
////		if (is_ll && is_up) { is_ll = x          <= y         ; is_up = y          <= x         ; }
////		if (is_ll && is_dw) { is_ll = x + 1      <= size.y - y; is_dw = size.y - y <= x + 1     ; }
////		if (is_rr && is_up) { is_rr = y + 1      >= size.x - x; is_up = y          <  size.x - x; }
////		if (is_rr && is_dw) { is_rr = size.x - x <= size.y - y; is_dw = size.y - y <= size.x - x; }
////
////		side side = is_ll ? side::ll : is_rr ? side::rr : is_up ? side::up : is_dw ? side::dw : side::no;
////		if (side == side::no) { out[index] = {0, 0, 0, 0}; return; }
////
////		float from_ll{x + .5f};
////		float from_rr{size.x - x - .5f};
////		float from_up{y + .5f};
////		float from_dw{size.y - y - .5f};
////
////		float from_edge{is_ll ? from_ll : is_rr ? from_rr : is_up ? from_up : is_dw ? from_dw : 0.f};
////		float t{from_edge / thickness};
////
////		float v{0.f};
////
////		if (t < .5f)
////			{
////			float inner_t{(t * 2.f)};
////			v = sqrt(1 - pow(inner_t - 1, 2));
////			}
////		else
////			{
////			float inner_t{((t - .5f) * 2.f)};
////			v = sqrt(1 - pow(inner_t, 2));
////			}
////
////
////		out[index].r = v;
////		out[index].g = v;
////		out[index].b = v;
////		out[index].a = 1.f;
////		}
////	}
////
////template <auto function, typename ...Args>
////void launch_vec(utils::CUDA::vector<rgba_f>& in, utils::CUDA::vector<rgba_f>& out, Args&&... args)
////	{
////	auto indent{cout.in()};
////	cout("calculating kernel sizes");
////	size_t pixels_per_block{256};
////	size_t threads_n{in.size() < pixels_per_block ? in.size() : pixels_per_block};
////	size_t blocks_n{in.size() / pixels_per_block + ((in.size() % pixels_per_block) ? 1 : 0)};
////	size_t smem_size{threads_n * sizeof(rgba_f)};
////
////	cout("run kernel");
////	utils::CUDA::launcher<function>{blocks_n, threads_n, smem_size}(in.get_device_vector(), out.get_device_vector(), std::forward<Args>(args)...);
////	}
//
//namespace utils::CUDA
//	{
//	namespace details
//		{
//		template <auto blend_operation, utils::graphics::colour::concepts::colour T>
//		inline static __global__ void blend(containers::matrix_observer<T>& a, const containers::matrix_observer<T>& b, const utils::math::vec2s offset = {0, 0})
//			{
//			const size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};
//			bool exists{index < b.size()};
//			if (!exists) { return; }
//
//			const auto coords_b{b.get_coords(index)};
//			const auto coords_a{coords_b + offset};
//
//			if (coords_a.x >= a.sizes().x || coords_a.y >= a.sizes().y) { return; }
//
//			blend_operation(a[coords_a], coords_b);
//			}
//		}
//
//	template <utils::graphics::colour::concepts::colour T>
//	class device_image : public utils::CUDA::containers::device_matrix<T>
//		{
//		public:
//			using coords_type = utils::CUDA::containers::device_matrix<T>::coords_type;
//			using utils::CUDA::containers::device_matrix<T>::device_matrix;
//
//			device_image& blend_normal(const device_image& other, const utils::math::vec2s offset = {0, 0})
//				{
//				size_t pixels_per_block{256};
//				size_t threads_n{other.size() < pixels_per_block ?   other.size() : pixels_per_block};
//				size_t blocks_n {other.size() / pixels_per_block + ((other.size() % pixels_per_block) ? 1 : 0)};
//				size_t smem_size{threads_n * 2 * sizeof(T)};
//				utils::CUDA::launcher<details::blend<blend_operation_normal, T>>{blocks_n, threads_n, smem_size}(this->get_matrix_observer(), other.get_matrix_observer(), offset);
//				}
//
//			//TODO gpu copy constructor
//			//inline static device_image blend_normal(const device_image& a, const device_image& b, const utils::math::vec2s offset = {0, 0})
//			//	{
//			//	device_image ret{a};
//			//	}
//
//		private:
//			inline static __device__ void blend_operation_normal(T& a, const T& b) noexcept
//				{
//				if constexpr (T::has_alpha)
//					{
//					for (size_t i = 0; i < T::static_size - 1; i++)
//						{
//						a[i] = a[i] * a.a + b[i] * b.a;
//						}
//					}
//				else { a = b; }
//				}
//		};
//	}
//
//
//
//void mainz()
//	{
//	utils::CUDA::device_image<utils::graphics::colour::rgba_f> blank{utils::math::vec2u{816, 1100}};
//	utils::CUDA::device_image<utils::graphics::colour::rgba_f> rect {utils::math::vec2u{200,  100}};
//
//	blank.blend_normal(rect, {10, 20});
//
//	utils::containers::matrix_dyn<utils::graphics::colour::rgba_f> cpu_image{blank.sizes()};
//	sf::Image sf_image;
//	sf_image.create(blank.sizes().x, blank.sizes().y);
//	for (size_t y{0}; y < blank.sizes().y; y++)
//		{
//		for (size_t x{0}; x < blank.sizes().x; x++)
//			{
//			const auto& src{cpu_image[{x, y}]};
//			sf::Color sfc{static_cast<uint8_t>(src.r * 255.f), static_cast<uint8_t>(src.g * 255.f), static_cast<uint8_t>(src.b * 255.f), static_cast<uint8_t>(src.a * 255.f)};
//			sf_image.setPixel(x, y, sfc);
//			}
//		}
//	
//	cout("save to file");
//	sf_image.saveToFile("out.png");
//	}

