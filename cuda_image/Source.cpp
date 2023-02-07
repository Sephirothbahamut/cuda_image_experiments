#include <iostream>
#include <algorithm>
#include <execution>

#include <utils/math/vec2.h>
#include <utils/math/vec3.h>
#include <utils/index_range.h>
#include <utils/math/easings.h>
#include <utils/graphics/colour.h>
#include <utils/math/geometry/shapes.h>
#include <utils/oop/disable_move_copy.h>
#include <utils/containers/matrix_dyn.h>
#include <utils/math/geometry/interactions.h>

#include <SFML/Graphics.hpp>

utils::math::vec2f vec_to_closest_edge(const utils::math::vec2f coords, const utils::math::geometry::polygon& shape)
	{
	float min{std::numeric_limits<float>::infinity()};
	utils::math::geometry::segment closest_edge;

	for (const auto& edge : shape.get_edges())
		{
		float from_edge{utils::math::geometry::distance(edge, coords)};
		if (from_edge < min) { min = from_edge; closest_edge = edge; }
		}

	utils::math::geometry::point closest_point{closest_edge.closest_point(coords)};
	return closest_point - coords;
	}
utils::math::vec2f vec_to_closest_edge(const utils::math::vec2f coords, const utils::math::geometry::aabb& shape)
	{
	float min{std::numeric_limits<float>::infinity()};
	utils::math::geometry::segment closest_edge;

	if(true)
		{
		utils::math::geometry::segment edge{shape.ul(), shape.ur()};
		float from_edge{utils::math::geometry::distance(edge, coords)};
		if (from_edge < min) { min = from_edge; closest_edge = edge; }
		}
	if (true)
		{
		utils::math::geometry::segment edge{shape.ur(), shape.dr()};
		float from_edge{utils::math::geometry::distance(edge, coords)};
		if (from_edge < min) { min = from_edge; closest_edge = edge; }
		}
	if (true)
		{
		utils::math::geometry::segment edge{shape.dr(), shape.dl()};
		float from_edge{utils::math::geometry::distance(edge, coords)};
		if (from_edge < min) { min = from_edge; closest_edge = edge; }
		}
	if (true)
		{
		utils::math::geometry::segment edge{shape.dl(), shape.ul()};
		float from_edge{utils::math::geometry::distance(edge, coords)};
		if (from_edge < min) { min = from_edge; closest_edge = edge; }
		}

	utils::math::geometry::point closest_point{closest_edge.closest_point(coords)};
	return closest_point - coords;
	}

template <utils::math::geometry::shape_discrete shape_a_t, utils::math::geometry::shape_discrete shape_b_t>
float normalized_distance_mask(const utils::math::vec2f coords, const shape_a_t& shape_a, const shape_b_t& shape_b)
	{
	float dist_a{utils::math::geometry::distance(coords, shape_a)};
	float dist_b{utils::math::geometry::distance(coords, shape_b)};

	float tot_dist{dist_a + dist_b};
	float proportional_a{dist_a / tot_dist};
	float proportional_b{dist_b / tot_dist};

	return proportional_a;
	}
template <utils::math::geometry::shape_discrete shape_a_t>
float draw_shape(const utils::math::vec2f coords, const shape_a_t& shape)
	{
	float dist{utils::math::geometry::distance(coords, shape)};

	return std::max(1.f - dist, 0.f);
	}

sf::Color SFMLify_rgba_f(const utils::graphics::colour::rgba_f& colour)
	{
	// turn 0-1 to 0-255
	uint8_t col_r{static_cast<uint8_t>(colour.r * 255.f)};
	uint8_t col_g{static_cast<uint8_t>(colour.g * 255.f)};
	uint8_t col_b{static_cast<uint8_t>(colour.b * 255.f)};
	uint8_t col_a{static_cast<uint8_t>(colour.a * 255.f)};

	return {col_r, col_g, col_b, col_a};
	}

int main()
	{
	utils::math::vec2s image_size{816, 1100};

	// SFML image only to save to file
	sf::Image sf_albedo; sf_albedo.create(image_size.x, image_size.y);
	sf::Image sf_normal; sf_normal.create(image_size.x, image_size.y);
	sf::Image sf_light ; sf_light .create(image_size.x, image_size.y);
	sf::Image sf_img   ; sf_img   .create(image_size.x, image_size.y);

	// matrix only for parallel for_each since it doesn't work with ranges::iota :(
	// not used for actual storage
	utils::containers::matrix_dyn<utils::graphics::colour::rgba_f> matrix{image_size};

	// some shapes to draw stuff with
	utils::math::geometry::circle circle_a{.center{image_size / 2}, .radius{128}};
	utils::math::geometry::circle circle_b{.center{image_size / 2}, .radius{512}};
	utils::math::geometry::circle circle_d{.center{circle_a.center + 16}, .radius{64}};

	utils::math::geometry::segment sa{.a{20, 20}, .b{20, image_size.y - 40}};
	utils::math::geometry::segment sb{.a{image_size.x - 40, 20}, .b{image_size.x - 40, image_size.y - 40}};
	utils::math::geometry::segment sc{.a{sa.a}, .b{sb.b}};
	utils::math::geometry::segment sd{.a{sb.a}, .b{sa.b}};

	utils::math::geometry::point pa{20, 20};
	utils::math::geometry::point pb{image_size.x - 40, image_size.x - 40};

	const auto& shape_a{circle_a};
	const auto& shape_b{sc};
	
	// pixels at the edge to create a normal map from for lighting
	float edge_thickness{64.f};

	// direction of the light
	utils::math::vec3f light_dir{1.f, .5f, 3.f};
	light_dir.normalize_self();

	// pixel "shader" but I hate shaders languages so I'll do it in C++ lol
	std::for_each(std::execution::par, matrix.begin(), matrix.end(), [&](const auto& element)
		{
		size_t index{static_cast<size_t>(&element - matrix.data())}; //workaround indicized parallel foreach
		auto coords{matrix.get_coords(index)};

		// albedo
		float gradient{normalized_distance_mask(coords, shape_a, shape_b)};
		float shape_a_value{draw_shape(coords, shape_a)};
		float shape_b_value{draw_shape(coords, shape_b)};

		utils::graphics::colour::rgba_f albedo{gradient, shape_a_value, shape_b_value, 1.f};
		sf_albedo.setPixel(coords.x, coords.y, SFMLify_rgba_f(albedo));

		// calc normal vector
		auto vec2{vec_to_closest_edge(coords, utils::math::geometry::aabb{0.f, 0.f, static_cast<float>(image_size.x), static_cast<float>(image_size.y)})};

		if (vec2.length > edge_thickness) { vec2 = {}; }
		else
			{
			vec2.length = utils::math::easing::ease<utils::math::easing::circ, utils::math::easing::in_out>(vec2.length / edge_thickness);
			}
		
		utils::math::vec3f normal{vec2.x, vec2.y, std::sqrt(1.f - (vec2.x * vec2.x) - (vec2.y * vec2.y))};
		utils::graphics::colour::rgba_f normal_visualizable
			{
			(normal.x + 1.f) / 2.f,
			(normal.y + 1.f) / 2.f,
			(normal.z + 1.f) / 2.f,
			1.f
			};
		sf_normal.setPixel(coords.x, coords.y, SFMLify_rgba_f({normal_visualizable}));

		// lightmap
		float light{normal <utils::math::operators::dot> light_dir};
		light = std::clamp(light, 0.f, 1.f); // could do some fancy software-HDR instead

		sf_light.setPixel(coords.x, coords.y, SFMLify_rgba_f(light));

		// apply light to albedo
		utils::graphics::colour::rgba_f colour{albedo * light};
		colour.a = 1.f;

		// SFML-ify
		sf_img.setPixel(coords.x, coords.y, SFMLify_rgba_f(colour));
		});

	sf_albedo.saveToFile("0_albedo.png");
	sf_normal.saveToFile("1_normal.png");
	sf_light .saveToFile("2_light.png" );
	sf_img   .saveToFile("3_img.png"   );
	}