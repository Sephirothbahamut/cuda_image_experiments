#pragma once

#include <string>
#include <unordered_map>

#include <utils\MS\graphics\d2d.h>

#include "image.h"

namespace text
	{
	using format = utils::MS::graphics::dw::text_format;

	class manager
		{
		public:
			manager() :
				dw_factory         {},
				d3d_device         {},
				dxgi_device        {d3d_device},
				d2d_factory        {},
				d2d_device         {d2d_factory, dxgi_device},
				d2d_device_context {d2d_device},
				wic_imaging_factory{},
				white_brush        {d2d_device, utils::graphics::colour::rgba_f{utils::graphics::colour::base::white}}
				{}

			format create_text_format(const format::create_info& create_info) noexcept
				{
				return utils::MS::graphics::dw::text_format{dw_factory, create_info};
				}

			::image<utils::graphics::colour::rgba_f> create_text_image(const std::wstring& string, const format& format, utils::math::vec2f max_size)
				{
				utils::MS::graphics::dw::text_layout layout{dw_factory, string, format, max_size};
				utils::math::vec2f size{layout.get_size()};
				utils::math::vec2u usize{size.x, size.y};
				size_t bytes_per_pixel{4};

				//utils::MS::graphics::d2d::bitmap gpu_bitmap{d2d_device_context, usize};
				//
				//utils::MS::graphics::details::com_ptr<IWICBitmap> bitmap;
				//utils::MS::graphics::details::throw_if_failed(wic_imaging_factory->CreateBitmap
				//	(
				//	usize.x,
				//	usize.y, 
				//	GUID_WICPixelFormat32bppPBGRA,
				//	WICBitmapCacheOnLoad, 
				//	bitmap.address_of()
				//	));
				//utils::MS::graphics::details::com_ptr<ID2D1RenderTarget> bitmap_gpu;
				//utils::MS::graphics::details::throw_if_failed(d2d_factory->CreateWicBitmapRenderTarget
				//	(
				//	bitmap.get(),
				//	D2D1_RENDER_TARGET_PROPERTIES
				//		{
				//		.type{D2D1_RENDER_TARGET_TYPE_DEFAULT},
				//		.pixelFormat
				//			{
				//			.format{DXGI_FORMAT_B8G8R8A8_UNORM},
				//			.alphaMode{D2D1_ALPHA_MODE_PREMULTIPLIED}
				//			},
				//		.dpiX{0.f},
				//		.dpiY{0.f},
				//		.usage{D2D1_RENDER_TARGET_USAGE_GDI_COMPATIBLE}
				//		},
				//	bitmap_gpu.address_of()
				//	));

				utils::MS::graphics::details::com_ptr<ID2D1Bitmap1> bitmap_gpu;
				utils::MS::graphics::details::throw_if_failed(d2d_device_context->CreateBitmap
					(
					D2D1_SIZE_U{.width{usize.x}, .height{usize.y}},
					nullptr,
					usize.x * 4,
					D2D1_BITMAP_PROPERTIES1
						{
						.pixelFormat
							{
							.format{DXGI_FORMAT_B8G8R8A8_UNORM},
							.alphaMode{D2D1_ALPHA_MODE_PREMULTIPLIED}
							},
						.bitmapOptions{D2D1_BITMAP_OPTIONS_TARGET},
						},
					bitmap_gpu.address_of()
					));
				
				d2d_device_context->SetTarget(bitmap_gpu.get());
				
				d2d_device_context->SetTextAntialiasMode(D2D1_TEXT_ANTIALIAS_MODE_GRAYSCALE);
				
				d2d_device_context->BeginDraw();
				d2d_device_context->Clear(D2D1_COLOR_F{.r{0.f}, .g{0.f}, .b{0.f}, .a{0.f}});
				d2d_device_context->DrawTextLayout({0.f, 0.f}, layout.get(), white_brush.get(), D2D1_DRAW_TEXT_OPTIONS_CLIP);
				d2d_device_context->EndDraw();
				
				utils::MS::graphics::details::com_ptr<ID2D1Bitmap1> bitmap_cpu;
				utils::MS::graphics::details::throw_if_failed(d2d_device_context->CreateBitmap
					(
					D2D1_SIZE_U{.width{usize.x}, .height{usize.y}},
					nullptr,
					0u,//usize.x * 4,
					D2D1_BITMAP_PROPERTIES1
					{
					.pixelFormat
						{
						.format{DXGI_FORMAT_B8G8R8A8_UNORM},
						.alphaMode{D2D1_ALPHA_MODE_PREMULTIPLIED}
						},
					.bitmapOptions{D2D1_BITMAP_OPTIONS_CANNOT_DRAW | D2D1_BITMAP_OPTIONS_CPU_READ},
					},
					bitmap_cpu.address_of()
					));
				bitmap_cpu->CopyFromBitmap(nullptr, bitmap_gpu.get(), nullptr);

				utils::MS::graphics::wic::save_to_file(wic_imaging_factory, d2d_device, bitmap_gpu, "gpu.png");
				utils::MS::graphics::wic::save_to_file(wic_imaging_factory, d2d_device, bitmap_cpu, "cpu.png");


				D2D1_MAPPED_RECT mapped;
				utils::MS::graphics::details::throw_if_failed(bitmap_cpu->Map(D2D1_MAP_OPTIONS_READ, &mapped));
				
				image<utils::graphics::colour::rgba_f> ret{usize};
				
				for (size_t i{0}; i < ret.mat.size(); i++)
					{
					size_t bits_index{i * 4};
					utils::graphics::colour::rgba_u pixel_uint{mapped.bits[bits_index], mapped.bits[bits_index + 1], mapped.bits[bits_index + 2], mapped.bits[bits_index + 3]};
					utils::graphics::colour::rgba_f pixel_float{pixel_uint};
					ret.mat[i] = pixel_float;
					}

				return ret;
				}

		private:
			utils::MS::graphics::dw  ::factory         dw_factory;
			utils::MS::graphics::d3d ::device          d3d_device;
			utils::MS::graphics::dxgi::device          dxgi_device;
			utils::MS::graphics::d2d ::factory         d2d_factory;
			utils::MS::graphics::d2d ::device          d2d_device{d2d_factory, dxgi_device};
			utils::MS::graphics::d2d ::device_context  d2d_device_context{d2d_device};
			utils::MS::graphics::wic ::imaging_factory wic_imaging_factory;
			utils::MS::graphics::d2d ::solid_brush     white_brush;
		};
	}