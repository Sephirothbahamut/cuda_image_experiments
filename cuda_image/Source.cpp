#include <iostream>
#include <algorithm>
#include <execution>

#include <utils/index_range.h>
#include <utils/graphics/colour.h>
#include <utils/math/geometry/shapes.h>
#include <utils/oop/disable_move_copy.h>
#include <utils/containers/matrix_dyn.h>
#include <utils/math/geometry/interactions.h>

#include <SFML/Graphics.hpp>

enum class side { ll, up, rr, dw, no };

void draw_border(const size_t index, const utils::containers::matrix_dyn<utils::graphics::colour::rgba_f>& in, utils::containers::matrix_dyn<utils::graphics::colour::rgba_f>& out, const size_t thickness)
	{
	//size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};
	size_t x{in.get_x(index)};
	size_t y{in.get_x(index)};

	auto size{in.sizes()};

	bool is_ll{x <           thickness};
	bool is_rr{x >= size.x - thickness};
	bool is_up{y <           thickness};
	bool is_dw{y >= size.y - thickness};

	if (is_ll && is_up) { is_ll = x          <= y         ; is_up = y          <= x         ; }
	if (is_ll && is_dw) { is_ll = x + 1      <= size.y - y; is_dw = size.y - y <= x + 1     ; }
	if (is_rr && is_up) { is_rr = y + 1      >= size.x - x; is_up = y          <  size.x - x; }
	if (is_rr && is_dw) { is_rr = size.x - x <= size.y - y; is_dw = size.y - y <= size.x - x; }

	side side = is_ll ? side::ll : is_rr ? side::rr : is_up ? side::up : is_dw ? side::dw : side::no;
	if (side == side::no) { out[index] = {0, 0, 0, 0}; return; }

	float from_ll{x + .5f};
	float from_rr{size.x - x - .5f};
	float from_up{y + .5f};
	float from_dw{size.y - y - .5f};

	float from_edge{is_ll ? from_ll : is_rr ? from_rr : is_up ? from_up : is_dw ? from_dw : 0.f};
	float t{from_edge / thickness};

	float v{0.f};

	if (t < .5f)
		{
		float inner_t{(t * 2.f)};
		v = sqrt(1 - pow(inner_t - 1, 2));
		}
	else
		{
		float inner_t{((t - .5f) * 2.f)};
		v = sqrt(1 - pow(inner_t, 2));
		}


	out[index].r = v;
	out[index].g = v;
	out[index].b = v;
	out[index].a = 1.f;
	}

template <utils::math::geometry::shape_discrete shape_a_t, utils::math::geometry::shape_discrete shape_b_t>
void normalized_distance_mask(const size_t index, utils::containers::matrix_dyn<float>& out, const shape_a_t& shape_a, const shape_b_t& shape_b)
	{
	size_t x{out.get_x(index)};
	size_t y{out.get_x(index)};

	utils::math::vec2f world_coords{out.get_coords(index)};

	float dist_a{utils::math::geometry::distance(world_coords, shape_a)};
	float dist_b{utils::math::geometry::distance(world_coords, shape_b)};

	float tot_dist{dist_a + dist_b};
	float proportional_a{dist_a / tot_dist};
	float proportional_b{dist_b / tot_dist};

	out[index] = proportional_a;
	}
template <utils::math::geometry::shape_discrete shape_a_t>
void draw_shape(const size_t index, utils::containers::matrix_dyn<float>& out, const shape_a_t& shape)
	{
	size_t x{out.get_x(index)};
	size_t y{out.get_x(index)};

	utils::math::vec2f world_coords{out.get_coords(index)};

	float dist{utils::math::geometry::distance(world_coords, shape)};

	out[index] = std::max(1.f - dist, out[index]);
	}

int main()
	{
	utils::math::vec2s image_size{816, 1100};
	sf::Image sfimg; sfimg.create(image_size.x, image_size.y);

	utils::containers::matrix_dyn<utils::graphics::colour::rgba_f> a{image_size};
	utils::containers::matrix_dyn<utils::graphics::colour::rgba_f> b{image_size};
	utils::containers::matrix_dyn<float> mask{image_size};
	utils::containers::matrix_dyn<float> shape_a_mask{image_size};
	utils::containers::matrix_dyn<float> shape_b_mask{image_size};

	std::ranges::iota_view indices{a.begin(), a.end()};

	utils::math::geometry::circle circle_a{.center{image_size / 2}, .radius{128}};
	utils::math::geometry::circle circle_b{.center{image_size / 2}, .radius{512}};
	utils::math::geometry::circle circle_d{.center{circle_a.center + 16}, .radius{64}};

	utils::math::geometry::segment sa{.a{20, 20}, .b{20, image_size.x - 40}};
	utils::math::geometry::segment sb{.a{image_size.x - 40, 20}, .b{image_size.x - 40, image_size.x - 40}};
	utils::math::geometry::segment sc{.a{sa.a}, .b{sb.b}};
	utils::math::geometry::segment sd{.a{sb.a}, .b{sa.b}};

	utils::math::geometry::point pa{20, 20};
	utils::math::geometry::point pb{image_size.x - 40, image_size.x - 40};

	const auto& shape_a{circle_a};
	const auto& shape_b{sd};

	std::for_each(std::execution::par, a.begin(), a.end(), [&](const auto& element)
		{
		size_t index{static_cast<size_t>(&element - a.data())};
		//draw_border(index, a, b, 5);
		normalized_distance_mask(index, mask, shape_a, shape_b);
		draw_shape(index, shape_a_mask, shape_a);
		draw_shape(index, shape_b_mask, shape_b);
		});

	std::for_each(std::execution::par, a.begin(), a.end(), [&](const auto& element)
		{
		size_t index{static_cast<size_t>(&element - a.data())};

		auto coords{b.get_coords(index)};
		
		const float& pixel_v{mask        [index]};
		const float& pixel_a{shape_a_mask[index]};
		const float& pixel_b{shape_b_mask[index]};

		uint8_t col_r{static_cast<uint8_t>(pixel_a * 255.f)};
		uint8_t col_g{static_cast<uint8_t>(pixel_b * 255.f)};
		uint8_t col_b{static_cast<uint8_t>(pixel_v * 255.f)};
		uint8_t col_a{static_cast<uint8_t>(255.f)};

		sf::Color sfcol{col_r, col_g, col_b, col_a};
		sfimg.setPixel(coords.x, coords.y, sfcol);
		});

	sfimg.saveToFile("out.png");
	}