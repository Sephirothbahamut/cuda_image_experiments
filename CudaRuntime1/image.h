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

		static image<T> from_file(const std::filesystem::path& file_path)
			requires std::same_as<T, utils::graphics::colour::rgba_f>
			{
			sf::Image sf_image;
			sf_image.loadFromFile(file_path.string());

			image<T> image{{sf_image.getSize().x, sf_image.getSize().y}};

			for (size_t y{0}; y < image.mat.height(); y++)
				{
				for (size_t x{0}; x < image.mat.width(); x++)
					{
					image.mat[{x, y}] = rgba_fify_SFML(sf_image.getPixel(x, y));
					}
				}
			return image;
			}

		std::vector<T> vec;
		utils::matrix_wrapper<std::vector<T>> mat;

		void save_to_file(const std::string& fname) const noexcept
			{
			sf::Image sfimage;
			sfimage.create(mat.width(), mat.height());

			for (size_t y = 0; y < mat.height(); y++)
				{
				for (size_t x = 0; x < mat.width(); x++)
					{
					sfimage.setPixel(x, y, eval_pixel(utils::math::vec2s{x, y}));
					}
				}
			sfimage.saveToFile(fname);
			}

	private:
		sf::Color eval_pixel(utils::math::vec2s coords) const noexcept
			{
			auto value{mat[coords]};

			if constexpr (utils::graphics::colour::concepts::colour<T>)
				{
				if constexpr (std::floating_point<typename T::value_type>)
					{
					return SFMLify_rgba_f(value);
					}
				else
					{
					utils::graphics::colour::rgba_u colour_8bit{value};
					return sf::Color{colour_8bit.r, colour_8bit.g, colour_8bit.b, colour_8bit.a};
					}
				}
			else if constexpr (std::same_as<T, utils::math::vec3f>)
				{
				return SFMLify_rgba_f(utils::graphics::colour::rgba_f{value.x, value.y, value.z});
				}
			else if constexpr (std::same_as<T, float>)
				{
				return SFMLify_rgba_f({value});
				}
			else if constexpr (std::same_as<T, bool>)
				{
				return sf::Color{value ? 255u : 0u, value ? 255u : 0u, value ? 255u : 0u, 255u};
				}
			else { std::unreachable(); }
			}
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

template <typename T>
image<utils::math::vec3f> normal_from_height(const image<T>& heightmap)
	{
	image<utils::math::vec3f> ret{heightmap.mat.sizes()};

	foreach([&](size_t index, utils::math::vec2s coords)
		{
		auto get_height{[&](utils::math::vec2i64 signed_coords)
			{
			if (signed_coords.x < 0) { signed_coords.x = 0; } else if (signed_coords.x >= heightmap.mat.width ()) { signed_coords.x = heightmap.mat.width () - 1; }
			if (signed_coords.y < 0) { signed_coords.y = 0; } else if (signed_coords.y >= heightmap.mat.height()) { signed_coords.y = heightmap.mat.height() - 1; }
			
			utils::math::vec2s coords{static_cast<size_t>(signed_coords.x), static_cast<size_t>(signed_coords.y)};

			float height{0.f};

			if constexpr (std::same_as<T, float>) { height = heightmap.mat[coords]; }
			else if constexpr (utils::math::concepts::vec<T>)
				{
				for (size_t i{0}; i < T::static_size; i++) { height += heightmap.mat[coords][i]; }
				height /= static_cast<float>(T::static_size);
				}
			else if constexpr (/*utils::graphics::concepts::rgb<T>*/true)
				{
				height = heightmap.mat[coords].a;
				}
			return height;
			}};

		float tl{get_height({static_cast<int64_t>(coords.x) - 1, static_cast<int64_t>(coords.y) - 1})}; // top left
		float  l{get_height({static_cast<int64_t>(coords.x) - 1, static_cast<int64_t>(coords.y)    })}; // left
		float bl{get_height({static_cast<int64_t>(coords.x) - 1, static_cast<int64_t>(coords.y) + 1})}; // bottom left
		float  t{get_height({static_cast<int64_t>(coords.x)    , static_cast<int64_t>(coords.y) - 1})}; // top
		float  b{get_height({static_cast<int64_t>(coords.x)    , static_cast<int64_t>(coords.y) + 1})}; // bottom
		float tr{get_height({static_cast<int64_t>(coords.x) + 1, static_cast<int64_t>(coords.y) - 1})}; // top right
		float  r{get_height({static_cast<int64_t>(coords.x) + 1, static_cast<int64_t>(coords.y)    })}; // right
		float br{get_height({static_cast<int64_t>(coords.x) + 1, static_cast<int64_t>(coords.y) + 1})}; // bottom right

		tl = tl * .2f + l * .4f + t * .4f;
		bl = bl * .2f + l * .4f + b * .4f;
		tr = tr * .2f + r * .4f + t * .4f;
		br = br * .2f + r * .4f + b * .4f;

		// sobel filter
		const float dX = (tr + 2.f * r + br) - (tl + 2.f * l + bl);
		const float dY = (bl + 2.f * b + br) - (tl + 2.f * t + tr);
		const float dZ = 1.f / 1.f;

		utils::math::vec3f n{dX, dY, dZ};
		n.normalize_self();
		ret.mat[coords] = n;
		}, heightmap.mat.sizes());

	return ret;
	}