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
				generator_cellular.SetSeed                    (255);

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
				utils::math::vec2f offset_pos {pos + noise_offset * generator_offset_z.GetNoise(pos.x, pos.y) * 12.f};
				
				float cellular_value{utils::math::clamp(utils::math::map(-0.9f, -0.7f, 1.f, 0.f, generator_cellular.GetNoise(offset_pos.x, offset_pos.y)), 0.f, 1.f)};

				float fissure_depth{cellular_value};
				
				float ghost{utils::math::map(-.5f, .5f, 0.f, 1.f, generator_ghost.GetNoise(pos.x, pos.y))};
				
				float value{fissure_depth * 2.f};
				float cubic_value{generator_value_cubic.GetNoise(pos.x * 1.f, pos.y * 1.f)};
				
				auto hue{utils::math::lerp(0_deg, 50_deg, cubic_value)};
				
				utils::graphics::colour::hsv<float, false> hsv{.h{hue}, .s{1.f}, .v{1.f}};
				auto rgb{hsv.rgb()};

				return {rgb.r, rgb.g, rgb.b, 1.2f};
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
	
	class shape
		{
		public:
			template <utils::math::geometry::concepts::shape shape_t>
			static utils::math::vec3f normal(utils::math::vec2f pos, const shape_t& shape, float radius, float thickness, bool ascending)
				{
				utils::math::geometry::closest_point_and_distance_t closest_dist{shape.closest_point_and_distance(pos)};
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
				}
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

#pragma region reference_measurements
	float outline_thickness                { 36.f};
	float border_outer_radius_and_thickness{ 16.f};
	float border_inner_radius              { 16.f};
	float border_inner_image_thickness     {border_inner_radius};
	float border_inner_text_thickness      {border_inner_radius * .5f};

	utils::math::vec2f padding_text_name{16.f, 0.f};
	utils::math::vec2f padding_text_desc{ 8.f, 8.f};
#pragma endregion reference_measurements

#pragma region text
	std::wstring text_name{L"Fiery Phoenix"};
	std::wstring text_desc{L"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."};
	
	text::manager text_manager;
	auto text_format_name{text_manager.create_text_format(text::format::create_info
		{
		.size{36.f},
		.weight{DWRITE_FONT_WEIGHT_BOLD}
		})};
	auto text_format_desc{text_manager.create_text_format(text::format::create_info
		{
		.size{16.f}
		})};

	float max_text_width{sizes.x - ((outline_thickness + border_outer_radius_and_thickness + border_inner_text_thickness) * 2.f)};
	auto image_text_name{text_manager.create_text_image(text_name, text_format_name, utils::graphics::colour::rgba_f{1.f, 0.84f, 0.f}, {max_text_width - (padding_text_name.x * 2.f),  64.f})};
	//auto image_text_name{text_manager.create_text_image(text_name, text_format_name, utils::graphics::colour::rgba_f{0.f, 0.6f, 1.f}, {max_text_width - (padding_text_name.x * 2.f),  64.f})};
	auto image_text_desc{text_manager.create_text_image(text_desc, text_format_desc, utils::graphics::colour::base::white, {max_text_width - (padding_text_desc.x * 2.f), 512.f})};

	auto image_normal_text_name{normal_from_height(image_text_name)};


#pragma endregion text

#pragma region edges
	float bottom_height = static_cast<float>(image_text_desc.mat.height());

	utils::math::geometry::aabb rect_outline{.ll{0.f}, .up{0.f}, .rr{static_cast<float>(sizes.x)}, .dw{static_cast<float>(sizes.y)}};

	utils::math::geometry::aabb shape_border_outer_begin{rect_outline};
	shape_border_outer_begin.pos () += outline_thickness;
	shape_border_outer_begin.size() -= outline_thickness * 2.f;

	utils::math::geometry::aabb shape_border_outer_end{shape_border_outer_begin};
	shape_border_outer_end.pos () += border_outer_radius_and_thickness;
	shape_border_outer_end.size() -= border_outer_radius_and_thickness * 2.f;
	
	utils::math::geometry::aabb shape_border_inner_text_begin
		{
		.ll{shape_border_outer_end.ll},
		.up{shape_border_outer_end.dw - (border_inner_text_thickness + padding_text_desc.y + image_text_desc.mat.height() + padding_text_desc.y + border_inner_text_thickness)},
		.rr{shape_border_outer_end.rr},
		.dw{shape_border_outer_end.dw}
		};
	utils::math::geometry::aabb shape_border_inner_text_end{shape_border_inner_text_begin};
	shape_border_inner_text_end.pos () += border_inner_text_thickness;
	shape_border_inner_text_end.size() -= border_inner_text_thickness * 2.f;

	utils::math::geometry::polygon shape_border_inner_image_begin
		{
			{shape_border_outer_end.ll, shape_border_outer_end.up + padding_text_name.y + image_text_name.mat.height() + padding_text_name.y},
			{shape_border_outer_end.ll + padding_text_name.x + image_text_name.mat.width() + padding_text_name.x, shape_border_outer_end.up + padding_text_name.y + image_text_name.mat.height() + padding_text_name.y},
			{shape_border_outer_end.ll + padding_text_name.x + image_text_name.mat.width() + padding_text_name.x + padding_text_name.x, shape_border_outer_end.up},
			{shape_border_outer_end.ur()},
			{shape_border_outer_end.rr, shape_border_inner_text_begin.up},
			{shape_border_outer_end.ll, shape_border_inner_text_begin.up}
		};
	utils::math::geometry::polygon shape_border_inner_image_end
		{
			{shape_border_inner_image_begin.get_vertices()[0] + utils::math::vec2f{ border_inner_image_thickness,  border_inner_image_thickness}},
			{shape_border_inner_image_begin.get_vertices()[1] + utils::math::vec2f{ border_inner_image_thickness,  border_inner_image_thickness}},
			{shape_border_inner_image_begin.get_vertices()[2] + utils::math::vec2f{ border_inner_image_thickness,  border_inner_image_thickness}},

			{shape_border_inner_image_begin.get_vertices()[3] + utils::math::vec2f{-border_inner_image_thickness,  border_inner_image_thickness}},
			{shape_border_inner_image_begin.get_vertices()[4] + utils::math::vec2f{-border_inner_image_thickness, -border_inner_image_thickness}},
			{shape_border_inner_image_begin.get_vertices()[5] + utils::math::vec2f{ border_inner_image_thickness, -border_inner_image_thickness}}
		};
#pragma endregion edges

#pragma region locations
	utils::math::vec2s coords_image_text_name{static_cast<size_t>(shape_border_outer_end     .ll + padding_text_name.x), static_cast<size_t>(shape_border_outer_end     .up + padding_text_name.y)};
	utils::math::vec2s coords_image_text_desc{static_cast<size_t>(shape_border_inner_text_end.ll + padding_text_desc.x), static_cast<size_t>(shape_border_inner_text_end.up + padding_text_desc.y)};
#pragma endregion locations

#pragma region light
	// direction of the light
	utils::math::vec3f light_dir{1.f, .5f, 3.f};
	light_dir.normalize_self();
	float light_intensity{1.0f}; //intensity > 1 makes the scene max colour even with slight angles
	auto light_source{light_dir * light_intensity};
#pragma endregion light

#pragma region effects
	effects::cracked_magma cracked_magma;
	effects::water         water;
#pragma endregion effects

	foreach([&](size_t index, utils::math::vec2s coords, image_mat<utils::graphics::colour::rgba_f>& image)
		{
		utils::math::vec2f pos{static_cast<float>(coords.x) + .5f, static_cast<float>(coords.y) + .5f};
		utils::math::vec2f normalized_pos{pos / std::min(image.width(), image.height())};

		bool is_outline{rect_outline.contains(pos) && !shape_border_outer_begin.contains(pos)};

		bool is_border_outer{shape_border_outer_begin.contains(pos) && !shape_border_outer_end.contains(pos)};

		bool is_border_body {shape_border_outer_end.contains(pos) && !shape_border_inner_image_begin.contains(pos) && !shape_border_inner_text_begin.contains(pos)};

		bool is_border_inner_image{shape_border_inner_image_begin.contains(pos) && !shape_border_inner_image_end.contains(pos)};
		bool is_border_inner_text {shape_border_inner_text_begin .contains(pos) && !shape_border_inner_text_end .contains(pos)};

		bool is_border_inner{is_border_inner_image || is_border_inner_text};

		bool is_border{is_border_outer || is_border_body || is_border_inner};
		
		bool is_image  {shape_border_inner_image_end.contains(pos)};
		bool is_textbox{shape_border_inner_text_end.contains(pos)};

		//dark outline rounded
		if (is_outline)
			{
			auto outline_dist{shape_border_outer_begin.distance_min(pos)};
			auto colour{1 - utils::math::clamp(outline_dist - outline_thickness, 0.f, 1.f)};
		
			image[index].r = 0.f;
			image[index].g = 0.f;
			image[index].b = 0.f;
			image[index].a = colour;
			return;
			}

		//normal
		utils::math::vec3f normal{0.f, 0.f, 1.f};

		if (is_border_outer)
			{
			normal = effects::shape::normal(pos, shape_border_outer_begin, border_outer_radius_and_thickness, border_outer_radius_and_thickness, false);
			}
		else if (is_border_inner_image)
			{
			normal = effects::shape::normal(pos, shape_border_inner_image_begin, border_inner_radius, border_inner_image_thickness, true);
			}
		else if (is_border_inner_text)
			{
			normal = effects::shape::normal(pos, shape_border_inner_text_begin, border_inner_text_thickness, border_inner_text_thickness, true);
			}

		//background
		auto background{cracked_magma(pos)};
		
		image[index] = background;

		if (is_image)
			{
			image[index].r /= 2.f;
			image[index].g /= 2.f;
			image[index].b /= 2.f;
			}

		if(coords.x < source.getSize().x && coords.y < source.getSize().y)
			{
			auto sf_col{source.getPixel(coords.x, coords.y)};
			image[index] = image[index].blend(rgba_fify_SFML(sf_col));
			}

		if (is_border)
			{
			for (size_t i{0}; i < 4; i++)
				{
				image[index][i] = utils::math::lerp(image[index][i], background[i], .5f);
				}
			}

		if (is_textbox)
			{
			image[index].r /= 3.f;
			image[index].g /= 3.f;
			image[index].b /= 3.f;
			}

		//texts
		if (coords.x > coords_image_text_name.x && coords.y > coords_image_text_name.y)
			{
			utils::math::vec2s tmp_coords{coords - coords_image_text_name};
			if (image_text_name.mat.is_valid_index(tmp_coords))
				{
				image[coords] = image[coords].blend(image_text_name.mat[tmp_coords]);

				normal += image_normal_text_name.mat[tmp_coords];
				if (normal.z > 1.f) { normal.z = 1.f; }
				}
			}
		if (coords.x > coords_image_text_desc.x && coords.y > coords_image_text_desc.y)
			{
			utils::math::vec2s tmp_coords{coords - coords_image_text_desc};
			if (image_text_desc.mat.is_valid_index(tmp_coords))
				{
				image[coords] = image[coords].blend(image_text_desc.mat[tmp_coords]);
				}
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

		//for phoenix, make right side wing overlap the border
		if (coords.x > shape_border_inner_image_end.get_vertices()[0].x && coords.y < shape_border_inner_image_end.get_vertices()[4].y)
			{
			auto sf_col{source.getPixel(coords.x, coords.y)};
			image[index] = image[index].blend(rgba_fify_SFML(sf_col));
			}




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
