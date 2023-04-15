// The MIT License
// Copyright © 2013 Inigo Quilez
// https://www.youtube.com/c/InigoQuilez
// https://iquilezles.org/
// Permission is hereby granted,  free of charge,  to any person obtaining a copy of this software and associated documentation files (the "Software"),  to deal in the Software without restriction,  including without limitation the rights to use,  copy,  modify,  merge,  publish,  distribute,  sublicense,  and/or sell copies of the Software,  and to permit persons to whom the Software is furnished to do so,  subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",  WITHOUT WARRANTY OF ANY KIND,  EXPRESS OR IMPLIED,  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,  DAMAGES OR OTHER LIABILITY,  WHETHER IN AN ACTION OF CONTRACT,  TORT OR OTHERWISE,  ARISING FROM,  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// I've not seen anybody out there computing correct cell interior distances for Voronoi
// patterns yet. That's why they cannot shade the cell interior correctly,  and why you've
// never seen cell boundaries rendered correctly. 
//
// However,  here's how you do mathematically correct distances (note the equidistant and non
// degenerated grey isolines inside the cells) and hence edges (in yellow):
//
// https://iquilezles.org/articles/voronoilines
//
// More Voronoi shaders:
//
// Exact edges:  https://www.shadertoy.com/view/ldl3W8
// Hierarchical: https://www.shadertoy.com/view/Xll3zX
// Smooth:       https://www.shadertoy.com/view/ldB3zc
// Voronoise:    https://www.shadertoy.com/view/Xd23Dh

#include <utils/math/vec2.h>
#include <utils/math/vec3.h>
#include <utils/matrix_interface.h>
#include <utils/graphics/colour.h>

utils::math::vec2f hash2(utils::math::vec2f p)
	{
	utils::math::vec2f tmp{utils::math::vec2f::dot(p, utils::math::vec2f{127.1, 311.7}), utils::math::vec2f::dot(p, utils::math::vec2f{269.5f, 183.3f})};
	
	tmp.for_each<[](float& value) 
		{
		float tmp{std::sin(value) * 43758.5453f};
		value = tmp - std::floor(tmp);
		}>();
	return tmp;
	}

struct voronoi_ret_t
	{
	utils::math::vec2f to_center;
	float edge_distance;
	};

voronoi_ret_t voronoi(utils::math::vec2f coords)
	{
	utils::math::vec2f n{coords.for_each_to_new<[](float value) { return         std::floor(value); }>()};
	utils::math::vec2f f{coords.for_each_to_new<[](float value) { return value - std::floor(value); }>()};

	//----------------------------------
	// first pass: regular voronoi
	//----------------------------------
	utils::math::vec2f mg,  mr;

	float md = 8.0;
	for (int j = -1; j <= 1; j++)
		{
		for (int i = -1; i <= 1; i++)
			{
			utils::math::vec2f g = utils::math::vec2f(float(i), float(j));
			utils::math::vec2f o = hash2(n + g);
			utils::math::vec2f r = g + o - f;
			float d = utils::math::vec2f::dot(r, r);

			if (d < md)
				{
				md = d;
				mr = r;
				mg = g;
				}
			}
		}

	//----------------------------------
	// second pass: distance to borders
	//----------------------------------
	md = 8.0;
	for (int j = -2; j <= 2; j++)
		{
		for (int i = -2; i <= 2; i++)
			{
			utils::math::vec2f g = mg + utils::math::vec2f(float(i), float(j));
			utils::math::vec2f o = hash2(n + g);

			utils::math::vec2f r = g + o - f;

			if (utils::math::vec2f::dot(mr - r, mr - r) > 0.00001f)
				{
				md = std::min(md, utils::math::vec2f::dot((mr + r) * 0.5f, (r - mr).normalize()));
				}
			}
		}

	return voronoi_ret_t{.to_center{mr.x, mr.y}, .edge_distance{md}};
	}