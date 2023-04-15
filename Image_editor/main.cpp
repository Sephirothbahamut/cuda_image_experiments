#include <vector>
#include <ranges>
#include <random>
#include <execution>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <utils/math/geometry/voronoi/voronoi.h>

#include "text.h"
#include "image.h"
#include "voronoi.h"
#include "noise.h"

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

void mainz();

int main()
	{
	try { mainz(); }
	catch (const std::exception e) { std::cout << e.what(); }
	return 0;
	}





namespace effects
	{
	class cracked_magma
		{
		public:
			cracked_magma() noexcept
				{
				generator_cellular.SetNoiseType               (FastNoiseLite::NoiseType_Cellular                );
				generator_cellular.SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_Euclidean);
				generator_cellular.SetCellularReturnType      (FastNoiseLite::CellularReturnType_Distance2Sub   );
				generator_cellular.SetFractalType             (FastNoiseLite::FractalType_FBm);
				generator_cellular.SetFractalOctaves          (2);

				generator_value_cubic.SetNoiseType(FastNoiseLite::NoiseType_ValueCubic);
				generator_value_cubic.SetFractalType(FastNoiseLite::FractalType_Ridged  );

				generator_offset_x.SetNoiseType  (FastNoiseLite::NoiseType_OpenSimplex2S);
				generator_offset_x.SetFractalType(FastNoiseLite::FractalType_FBm);
				generator_offset_x.SetFrequency  (0.03f);
				generator_offset_x.SetSeed       (0);

				generator_offset_y.SetNoiseType  (FastNoiseLite::NoiseType_OpenSimplex2S);
				generator_offset_y.SetFractalType(FastNoiseLite::FractalType_FBm        );
				generator_offset_y.SetFrequency  (0.03f);
				generator_offset_y.SetSeed       (3);

				generator_offset_y.SetNoiseType  (FastNoiseLite::NoiseType_OpenSimplex2S);
				generator_offset_y.SetFractalType(FastNoiseLite::FractalType_FBm        );
				generator_offset_y.SetFrequency  (0.03f);
				generator_offset_y.SetSeed       (6);

				generator_ghost.SetNoiseType     (FastNoiseLite::NoiseType_Perlin);
				generator_ghost.SetFractalType   (FastNoiseLite::FractalType_FBm);
				generator_ghost.SetFrequency     (0.005f);
				generator_ghost.SetFractalOctaves(6);
				}
			utils::graphics::colour::rgba_f operator()(utils::math::vec2f pos) const noexcept
				{
				using namespace utils::math::angle::literals;

				utils::math::vec2f noise_offset{generator_offset_x.GetNoise(pos.x, pos.y), generator_offset_y.GetNoise(pos.x, pos.y)};
				utils::math::vec2f offset_pos {pos + noise_offset * generator_offset_z.GetNoise(pos.x, pos.y) * 16.f};
				
				float cellular_value{utils::math::map(-0.9f, 0.1f, 0.f, 1.f, generator_cellular.GetNoise(offset_pos.x, offset_pos.y))};

				float fissure_depth{utils::math::map(0.f, .2f, 1.f, 0.f, cellular_value)};
				
				float ghost{utils::math::map(-.5f, .5f, 0.f, 1.f, generator_ghost.GetNoise(pos.x, pos.y))};
				
				float value{utils::math::clamp(fissure_depth * ghost, 0.f, 1.f)};
				float cubic_value{generator_value_cubic.GetNoise(pos.x * 1.f, pos.y * 1.f)};
				
				auto hue{utils::math::lerp(0_deg, 50_deg, cubic_value)};
				
				utils::graphics::colour::hsv<float, false> hsv{.h{hue}, .s{1.f}, .v{1.f}};
				auto rgb{hsv.rgb()};

				return {hsv.rgb(), value * value};
				}

		private:
			FastNoiseLite generator_cellular;

			FastNoiseLite generator_value_cubic;

			FastNoiseLite generator_offset_x;
			FastNoiseLite generator_offset_y;
			FastNoiseLite generator_offset_z;

			FastNoiseLite generator_ghost;
		};
	class water
		{
		public:
			water() noexcept
				{
				generator_cellular.SetNoiseType               (FastNoiseLite::NoiseType_Cellular                  );
				generator_cellular.SetCellularDistanceFunction(FastNoiseLite::CellularDistanceFunction_EuclideanSq);
				generator_cellular.SetCellularReturnType      (FastNoiseLite::CellularReturnType_Distance         );
				generator_cellular.SetFractalType      (FastNoiseLite::FractalType_FBm);
				generator_cellular.SetFractalOctaves   (3   );
				generator_cellular.SetFractalLacunarity(1.6f);
				
				generator_ghost.SetNoiseType     (FastNoiseLite::NoiseType_Perlin);
				generator_ghost.SetFractalType   (FastNoiseLite::FractalType_FBm);
				generator_ghost.SetFrequency     (0.005f);
				generator_ghost.SetFractalOctaves(6);
				
				generator_caustics.SetNoiseType        (FastNoiseLite::NoiseType_OpenSimplex2S);
				generator_caustics.SetFractalType      (FastNoiseLite::FractalType_Ridged);
				generator_caustics.SetFractalOctaves   (4   );
				generator_caustics.SetFractalLacunarity(1.6f);
				generator_caustics.SetFractalGain      ( .3f);
				}
			utils::graphics::colour::rgba_f operator()(utils::math::vec2f pos) const noexcept
				{
				using namespace utils::math::angle::literals;

				//float cellular_value{(1.f + generator_cellular.GetNoise(pos.x, pos.y)) * 2.f};
				
				float caustics{utils::math::map(-.5f, .9f, 0.f, 1.f, generator_caustics.GetNoise(pos.x, pos.y))};
				caustics *= caustics;

				float ghost{utils::math::map(-.5f, .5f, .4f, 1.f, generator_ghost.GetNoise(pos.x, pos.y))};
				
				//float intensity{utils::math::clamp(caustics * ghost, 0.f, 1.f)};
				float intensity{caustics * ghost};
				float clamped_intensity{utils::math::clamp(intensity, 0.f, 1.f)};

				float hue = utils::math::map(0.f, 1.f, 240.f/*blue*/, 180.f/*cyan*/, clamped_intensity);
				utils::math::angle::deg hue_deg{hue};

				float saturation{1.f - intensity};
				float value{utils::math::clamp(.5f + (intensity / 2.f), 0.f, 1.f)};

				utils::graphics::colour::hsv<float, false> hsv{.h{hue_deg}, .s{saturation}, .v{value += caustics}};
				auto rgb{hsv.rgb()};
				return {hsv.rgb(), 1.f};
				}

		private:
			FastNoiseLite generator_cellular;

			FastNoiseLite generator_caustics;

			FastNoiseLite generator_ghost;
		};
	}






void mainz()
	{
	std::random_device random_device;
	std::mt19937 random_generator{random_device()};

	utils::MS::graphics::initializer MS_graphics_initializer;

	using namespace utils::output;
	utils::math::vec2s sizes{744, 1039};
	auto image_rect{utils::math::rect<float>::from_possize(utils::math::vec2f{0.f, 0.f}, utils::math::vec2f{static_cast<float>(sizes.x), static_cast<float>(sizes.y)})};

	std::wstring name{L"Hello world!"};

	text::manager text_manager;
	auto format{text_manager.create_text_format(text::format::create_info
		{
		.size{24.f}
		})};

	::image<utils::graphics::colour::rgba_f> image{sizes};
	::image<utils::graphics::colour::rgba_f> bright_image{sizes};
	sf::Image source; 
	if (!source.loadFromFile("source.png")) 
		{
		if (!source.loadFromFile("source.jpg"))
			{
			return;
			}
		};

#pragma region edges
	float outline_thickness                { 36.f};
	float border_outer_radius_and_thickness{ 16.f};
	float border_inner_radius              { 16.f};
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

	utils::math::geometry::aabb rect_hole_top{border_inner_top_rect};
	rect_hole_top.ll += border_inner_top_thickness;
	rect_hole_top.up += border_inner_top_thickness;
	rect_hole_top.rr -= border_inner_top_thickness;
	rect_hole_top.dw -= border_inner_top_thickness;
	utils::math::geometry::aabb rect_hole_bottom{border_inner_bottom_rect};
	rect_hole_bottom.ll += border_inner_bottom_thickness;
	rect_hole_bottom.up += border_inner_bottom_thickness;
	rect_hole_bottom.rr -= border_inner_bottom_thickness;
	rect_hole_bottom.dw -= border_inner_bottom_thickness;

#pragma endregion edges

#pragma region name
	utils::math::vec2f name_box_inner_size{512.f, 48.f};
	float name_box_from_top               {32.f};
	float name_box_border_radius          {12.f};
	float name_box_border_thickness{name_box_border_radius * .5f};
	utils::math::vec2f name_box_size{name_box_inner_size + utils::math::vec2f{name_box_border_thickness * 2.f, name_box_border_thickness * 2.f}};

	utils::math::vec2f name_box_center{static_cast<float>(sizes.x) / 2.f, border_inner_top_rect.up + name_box_from_top + (name_box_size.y / 2.f)};

	auto name_box{utils::math::geometry::aabb::from_possize(name_box_center - (name_box_size / 2.f), name_box_size)};

	auto image_name{text_manager.create_text_image(name, format, name_box.size())};
	utils::math::rect<size_t> image_name_rect{utils::math::rect<size_t>::from_possize
		(
		utils::math::vec2s
			{
			static_cast<size_t>(name_box_center.x) - (image_name.mat.sizes().x / 2), 
			static_cast<size_t>(name_box_center.y) - (image_name.mat.sizes().y / 2)
			},
		image_name.mat.sizes() - utils::math::vec2s{1, 1}
		)};
#pragma endregion name


#pragma region light
	// direction of the light
	utils::math::vec3f light_dir{1.f, .5f, 3.f};
	light_dir.normalize_self();
	float light_intensity{1.1f}; //intensity > 1 makes the scene max colour even with slight angles
	auto light_source{light_dir * light_intensity};
#pragma endregion light

	effects::cracked_magma cracked_magma;
	effects::water         water;

	foreach([&](size_t index, utils::math::vec2s coords, image_mat<utils::graphics::colour::rgba_f>& image)
		{

		utils::math::vec2f pos{static_cast<float>(coords.x) + .5f, static_cast<float>(coords.y) + .5f};
		utils::math::vec2f normalized_pos{pos / std::min(image.width(), image.height())};

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

		bool is_frame{!rect_hole_top.contains(pos) && !rect_hole_bottom.contains(pos)};
		bool is_textbox{name_box.contains(pos) || rect_hole_bottom.contains(pos)};

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

		//background
		auto water_background = water(pos);
		image[index] = water_background;
		/*//picture in frame
		if (rect_hole_top.contains(pos))
			{
			image[index] = {0.f};
			auto source_pos{coords - rect_hole_top.ul()};
			if(source_pos.x < source.getSize().x && source_pos.y < source.getSize().y)
				{
				auto sf_col{source.getPixel(source_pos.x, source_pos.y)};
				image[index] = rgba_fify_SFML(sf_col);
				}
			}
		
		/*///full art
		if(coords.x < source.getSize().x && coords.y < source.getSize().y)
			{
			auto sf_col{source.getPixel(coords.x, coords.y)};
			image[index] = rgba_fify_SFML(sf_col);
			}/**/

		if (is_frame)
			{
			for (size_t i{0}; i < 4; i++)
				{
				image[index][i] = utils::math::lerp(image[index][i], water_background[i], .3f);
				}
			}
		if (is_textbox)
			{
			auto hsv{image[index].hsv()};
			hsv.s /= 2.f;
			hsv.v /= 2.f;

			image[index] = hsv.rgb();

			for (size_t i{0}; i < 4; i++)
				{
				image[index][i] = utils::math::lerp(image[index][i], water_background[i], .3f);
				}

			hsv = image[index].hsv();
			hsv.s /= 1.2f;
			hsv.v /= 4.f;

			image[index] = hsv.rgb();
			}

		//text
		if (image_name_rect.contains(coords))
			{
			auto image_name_coords{coords - image_name_rect.ul()};
			image[index].r = utils::math::clamp(image[index].r + image_name.mat[image_name_coords].r, 0.f, 1.f);
			image[index].g = utils::math::clamp(image[index].g + image_name.mat[image_name_coords].g, 0.f, 1.f);
			image[index].b = utils::math::clamp(image[index].b + image_name.mat[image_name_coords].b, 0.f, 1.f);
			}


		//lightmap
		float intensity{[normal, light_source]
			{
			float tmp{normal <utils::math::operators::dot> light_source};
			return std::clamp(tmp, 0.f, 1.f); // could do some fancy software-HDR instead
			}()};


		image[index].r *= intensity;
		image[index].g *= intensity;
		image[index].b *= intensity;
		image[index].a = 1.f;

		bright_image.mat[index].r = image[index].r > 1.f ? image[index].r : 0.f;
		bright_image.mat[index].g = image[index].g > 1.f ? image[index].g : 0.f;
		bright_image.mat[index].b = image[index].b > 1.f ? image[index].b : 0.f;

		image[index].r = image[index].r > 1.f ? 1.f : image[index].r;
		image[index].g = image[index].g > 1.f ? 1.f : image[index].g;
		image[index].b = image[index].b > 1.f ? 1.f : image[index].b;

		}, sizes, image.mat);

	//hdr and bloom
	foreach([&](size_t index, utils::math::vec2s coords, image_mat<utils::graphics::colour::rgba_f>& image)
		{
		utils::math::vec2f pos{static_cast<float>(coords.x) + .5f, static_cast<float>(coords.y) + .5f};
		utils::math::vec2f normalized_pos{pos / std::min(image.width(), image.height())};

		}, sizes, image.mat);


	image.save_to_file("hello.png");
	}
