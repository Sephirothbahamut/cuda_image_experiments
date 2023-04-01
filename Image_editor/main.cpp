#include <vector>
#include <ranges>
#include <execution>
#include <algorithm>

#include <SFML/Graphics.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <utils/math/vec2.h>
#include <utils/math/vec3.h>
#include <utils/index_range.h>
#include <utils/math/easings.h>
#include <utils/graphics/colour.h>
#include <utils/matrix_interface.h>
#include <utils/math/geometry/shapes.h>
#include <utils/math/geometry/interactions.h>

sf::Color SFMLify_rgba_f(const utils::graphics::colour::rgba_f& colour)
	{
	// turn 0-1 to 0-255
	uint8_t col_r{static_cast<uint8_t>(colour.r * 255.f)};
	uint8_t col_g{static_cast<uint8_t>(colour.g * 255.f)};
	uint8_t col_b{static_cast<uint8_t>(colour.b * 255.f)};
	uint8_t col_a{static_cast<uint8_t>(colour.a * 255.f)};

	return {col_r, col_g, col_b, col_a};
	}

template <typename T>
class image
	{
	public:
		using value_type = T;
		image(utils::math::vec2s sizes) : vec(sizes.x * sizes.y), mat{sizes, vec} {}

		std::vector<T> vec;
		utils::matrix_wrapper<std::vector<T>> mat;

		void save_to_file(const std::string& fname) const noexcept
			requires utils::graphics::colour::concepts::colour<T>
			{
			sf::Image sfimage;
			sfimage.create(mat.width(), mat.height());

			for (size_t y = 0; y < mat.height(); y++)
				{
				for (size_t x = 0; x < mat.width(); x++)
					{
					if constexpr (std::floating_point<typename T::value_type>)
						{
						sfimage.setPixel(x, y, SFMLify_rgba_f(mat[{x, y}]));
						}
					else
						{
						utils::graphics::colour::rgba_u colour_8bit{mat[{x, y}]};
						sfimage.setPixel(x, y, sf::Color{colour_8bit.r, colour_8bit.g, colour_8bit.b, colour_8bit.a});
						}
					}
				}
			sfimage.saveToFile(fname);
			}

	private:
	};

template <typename T>
using image_mat = decltype(image<T>::mat);

namespace concepts
	{
	template <typename T>
	concept image = std::same_as<T, ::image<typename T::value_type>>;
	}


template <typename callback_t, typename ...Args>
void foreach(callback_t callback, utils::math::vec2s sizes, Args&... args)
	{
	std::ranges::iota_view indices{size_t{0}, sizes.x * sizes.y};
	std::for_each(std::execution::par, indices.begin(), indices.end(), [&](size_t index)
		{
		utils::math::vec2s coords{index % sizes.x, index / sizes.x};
		callback(index, coords, args...);
		});
	}

namespace funcs
	{
	void position(size_t index, utils::math::vec2s coords, image_mat<utils::math::vec2f>& out)
		{
		out[index].x = static_cast<float>(coords.x) + .5f;
		out[index].y = static_cast<float>(coords.y) + .5f;
		}

	template <utils::math::geometry::concepts::shape shape_t>
	void closest_point(size_t index, utils::math::vec2s coords, image_mat<utils::math::vec2f>& out, image_mat<utils::math::vec2f>& from, const shape_t& shape)
		{
		out[index] = shape.closest_point_to(from[index]);
		}

	template <typename T>
	void difference(size_t index, utils::math::vec2s coords, image_mat<T>& out, image_mat<T>& a, image_mat<T>& b)
		{
		out[index] = a[index] - b[index];
		}
	}

int main()
	{
	using namespace utils::output;
	utils::math::vec2s sizes{744, 1039};

	//image<utils::math::vec2f> positions{sizes};
	//foreach(&funcs::position, sizes, positions.mat);
	//
	//const utils::math::geometry::segment& segment{{10, 5}, {128, 89}};
	//image<utils::math::vec2f> closest{sizes};
	//foreach([](size_t index, utils::math::vec2s coords, image_mat<utils::math::vec2f>& out, image_mat<utils::math::vec2f>& from, const auto& shape)
	//	{
	//	out[index] = shape.closest_point_to(from[index]);
	//	}, sizes, closest.mat, positions.mat, segment);
	//
	//image<utils::math::vec2f> distance2d{sizes};
	//foreach([](size_t index, utils::math::vec2s coords, image_mat<utils::math::vec2f>& out, image_mat<utils::math::vec2f>& a, image_mat<utils::math::vec2f>& b)
	//	{
	//	out[index] = b[index] - a[index];
	//	}, sizes, distance2d.mat, closest.mat, positions.mat);
	//
	//image<float> distance{sizes};
	//foreach([](size_t index, utils::math::vec2s coords, image_mat<float>& out, image_mat<utils::math::vec2f>& distance2d)
	//	{
	//	out[index] = distance2d[index].length;
	//	}, sizes, distance.mat, distance2d.mat);
	//
	//
	//image<utils::graphics::colour::rgba_f> image{sizes};
	//
	//foreach([](size_t index, utils::math::vec2s coords, image_mat<float>& a, utils::matrix_wrapper<std::vector<utils::graphics::colour::rgba_f>>& b)
	//	{
	//	b[index].r = a[index] / 50.f;
	//	b[index].g = a[index] / 50.f;
	//	b[index].b = a[index] / 50.f;
	//	b[index].a = 1.f;
	//	}, sizes, distance.mat, image.mat);


	image<utils::graphics::colour::rgba_f> image{sizes};

	float outline_thickness                { 36.f};
	float border_outer_radius_and_thickness{  8.f};
	float border_inner_radius              {  8.f};
	float border_inner_top_thickness       {border_inner_radius};
	float border_inner_bottom_thickness    {border_inner_radius * .5f};
	float bottom_height                    {256.f}; 

	utils::math::geometry::aabb rect_outline{.ll{0.f}, .up{0.f}, .rr{static_cast<float>(sizes.x)}, .dw{static_cast<float>(sizes.y)}};

	utils::math::geometry::aabb rect_border_outer{rect_outline};
	rect_border_outer.pos () += outline_thickness;
	rect_border_outer.size() -= outline_thickness * 2.f;

	utils::math::geometry::aabb rect_border_inner {rect_border_outer};
	rect_border_inner.pos () += border_outer_radius_and_thickness;
	rect_border_inner.size() -= border_outer_radius_and_thickness * 2.f;

	utils::math::geometry::aabb border_inner_top_rect{rect_border_inner};
	border_inner_top_rect.height() -= bottom_height;

	utils::math::geometry::aabb border_inner_bottom_rect{rect_border_inner};
	border_inner_bottom_rect.height() = bottom_height;
	border_inner_bottom_rect.y() = border_inner_top_rect.dw;

	utils::math::vec2f name_box_inner_size{512.f, 48.f};
	float name_box_from_top               {32.f};
	float name_box_border_radius          {12.f};
	float name_box_border_thickness{name_box_border_radius * .5f};
	utils::math::vec2f name_box_size{name_box_inner_size + utils::math::vec2f{name_box_border_thickness * 2.f, name_box_border_thickness * 2.f}};

	utils::math::vec2f name_box_center{static_cast<float>(sizes.x) / 2.f, border_inner_top_rect.up + name_box_from_top + (name_box_size.y / 2.f)};

	auto name_box{utils::math::geometry::aabb::from_possize(name_box_center - (name_box_size / 2.f), name_box_size)};

	// direction of the light
	utils::math::vec3f light_dir{1.f, .5f, 3.f};
	light_dir.normalize_self();
	float light_intensity{1.1f}; //intensity > 1 makes the scene max colour even with slight angles
	auto light_source{light_dir * light_intensity};



	foreach([&](size_t index, utils::math::vec2s coords, image_mat<utils::graphics::colour::rgba_f>& image)
		{
		utils::math::vec2f pos{static_cast<float>(coords.x) + .5f, static_cast<float>(coords.y) + .5f};

		//dark outline rounded
		if (!rect_border_outer.contains(pos))
			{
			auto outline_dist{rect_border_outer.distance_min(pos)};
			auto colour{1 - utils::math::clamp(outline_dist - outline_thickness, 0.f, 1.f)};
		
			image[index].r = 0.f;
			image[index].g = 0.f;
			image[index].b = 0.f;
			image[index].a = colour;
			return;
			}

		//normal
		auto normal{[&]
			{
			auto f{[pos](const utils::math::geometry::aabb& rect, float radius, float thickness, bool ascending)
				{
				utils::math::geometry::closest_point_and_distance_t closest_dist{rect.closest_point_and_distance(pos)};
				float proportion{closest_dist.distance / radius};
				utils::math::vec2f to_closest_edge{ascending ? closest_dist.position - pos : pos - closest_dist.position};

				if (closest_dist.distance > thickness) { to_closest_edge = {}; }
				else
					{
					if (ascending)
						{
						to_closest_edge.length = 0.f + utils::math::easing::ease<utils::math::easing::linear, utils::math::easing::in >(proportion);
						}
					else
						{
						to_closest_edge.length = 1.f - utils::math::easing::ease<utils::math::easing::linear, utils::math::easing::out>(proportion);
						}
					}

				float z{std::sqrt(1.f - (to_closest_edge.x * to_closest_edge.x) - (to_closest_edge.y * to_closest_edge.y))};
				return utils::math::vec3f{to_closest_edge.x, to_closest_edge.y, z};
				}};

			if (name_box.contains(pos))
				{
				return f(name_box, name_box_border_radius, name_box_border_thickness, false);
				}
			else if (border_inner_top_rect.contains(pos))
				{
				return f(border_inner_top_rect, border_inner_radius, border_inner_top_thickness, true);
				}
			else if (border_inner_bottom_rect.contains(pos))
				{
				return f(border_inner_bottom_rect, border_inner_radius, border_inner_bottom_thickness, true);
				}
			else
				{
				return f(rect_border_outer, border_outer_radius_and_thickness, border_outer_radius_and_thickness, false);
				}
			}()};

		utils::graphics::colour::rgba_f normal_visualizable
			{
			(normal.x + 1.f) / 2.f,
			(normal.y + 1.f) / 2.f,
			(normal.z + 1.f) / 2.f,
			1.f
			};
		image[index] = normal_visualizable;

		//lightmap
		float intensity{[normal, light_source]
			{
			float tmp{normal <utils::math::operators::dot> light_source};
			return std::clamp(tmp, 0.f, 1.f); // could do some fancy software-HDR instead
			}()};


		image[index].r = intensity;
		image[index].g = intensity;
		image[index].b = intensity;
		image[index].a = 1.f;
		}, sizes, image.mat);


	image.save_to_file("hello.png");
	}