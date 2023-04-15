#pragma once

#include <vector>
#include <execution>

#include <utils/math/vec2.h>
#include <utils/math/vec3.h>
#include <utils/index_range.h>
#include <utils/math/easings.h>
#include <utils/graphics/colour.h>
#include <utils/matrix_interface.h>
#include <utils/math/geometry/shapes.h>
#include <utils/math/geometry/interactions.h>

#include <SFML/Graphics.hpp>

sf::Color SFMLify_rgba_f(const utils::graphics::colour::rgba_f& colour)
	{
	// turn 0-1 to 0-255
	uint8_t col_r{static_cast<uint8_t>(colour.r * 255.f)};
	uint8_t col_g{static_cast<uint8_t>(colour.g * 255.f)};
	uint8_t col_b{static_cast<uint8_t>(colour.b * 255.f)};
	uint8_t col_a{static_cast<uint8_t>(colour.a * 255.f)};

	return {col_r, col_g, col_b, col_a};
	}
utils::graphics::colour::rgba_f rgba_fify_SFML(const sf::Color& colour)
	{
	// turn 0-1 to 0-255
	float col_r{static_cast<float>(colour.r) / 255.f};
	float col_g{static_cast<float>(colour.g) / 255.f};
	float col_b{static_cast<float>(colour.b) / 255.f};
	float col_a{static_cast<float>(colour.a) / 255.f};

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