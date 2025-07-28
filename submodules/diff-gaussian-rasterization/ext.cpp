/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depth_to_normal", &depthToNormal);        // 深度图转法线图
  m.def("SSAO", &SSAO);                           // 屏幕空间环境光遮蔽
  m.def("SSR", &SSR);                             // 屏幕空间反射
  m.def("SSR_BACKWARD", &SSR_BACKWARD);           // SSR反向传播
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);     // 完整高斯渲染
  m.def("lite_rasterize_gaussians", &LiteRasterizeGaussiansCUDA); // 轻量级渲染
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA); // 反向传播
  m.def("mark_visible", &markVisible);            // 可见性标记
}
