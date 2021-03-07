__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;
__kernel void default_function_kernel2(__read_only image2d_t mali_conv2d_nchw_winograd_U, __read_only image2d_t mali_conv2d_nchw_winograd_V, __write_only image2d_t mali_conv2d_nchw_winograd_M) {
  

    #define A mali_conv2d_nchw_winograd_U 
    #define B mali_conv2d_nchw_winograd_V 
    #define C mali_conv2d_nchw_winograd_M 
    const int nbatch=4*4;
    const int nm_pack4=get_image_dim(mali_conv2d_nchw_winograd_U).x/4/4;
    const int nk_pack4=get_image_dim(mali_conv2d_nchw_winograd_U).y;
    const int nn_pack4=get_image_dim(mali_conv2d_nchw_winograd_V).x;

    const int n_pack4 = get_global_id(0);
  const int m_pack4 = get_global_id(1);
  const int batch = get_global_id(2);
    //if (m_pack4==0&&batch==0)
    //printf("mali_conv2d_nchw_winograd_M %d,%d....",get_image_dim(mali_conv2d_nchw_winograd_M).x,get_image_dim(mali_conv2d_nchw_winograd_M).y);
    //return;

  if (batch >= nbatch || n_pack4 >= nn_pack4 || m_pack4 >= nm_pack4) return;

  const int n = n_pack4 << 2;

  float4 a0, a1, a2, a3;
  float4 b0, b1, b2, b3;
  float4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;

  for (short k_pack4 = 0; k_pack4 < nk_pack4; k_pack4 += 1) {
    const int k = k_pack4 << 2;

    a0 = read_imagef(A, sampler, (int2)(m_pack4 + batch * nm_pack4, k + 0));
    a1 = read_imagef(A, sampler, (int2)(m_pack4 + batch * nm_pack4, k + 1));
    a2 = read_imagef(A, sampler, (int2)(m_pack4 + batch * nm_pack4, k + 2));
    a3 = read_imagef(A, sampler, (int2)(m_pack4 + batch * nm_pack4, k + 3));

    b0 = read_imagef(B, sampler, (int2)(n + 0, batch + k_pack4 * nbatch));
    b1 = read_imagef(B, sampler, (int2)(n + 1, batch + k_pack4 * nbatch));
    b2 = read_imagef(B, sampler, (int2)(n + 2, batch + k_pack4 * nbatch));
    b3 = read_imagef(B, sampler, (int2)(n + 3, batch + k_pack4 * nbatch));

    c0 = mad(b0.x, a0, c0);
    c0 = mad(b0.y, a1, c0);
    c0 = mad(b0.z, a2, c0);
    c0 = mad(b0.w, a3, c0);

    c1 = mad(b1.x, a0, c1);
    c1 = mad(b1.y, a1, c1);
    c1 = mad(b1.z, a2, c1);
    c1 = mad(b1.w, a3, c1);

    c2 = mad(b2.x, a0, c2);
    c2 = mad(b2.y, a1, c2);
    c2 = mad(b2.z, a2, c2);
    c2 = mad(b2.w, a3, c2);

    c3 = mad(b3.x, a0, c3);
    c3 = mad(b3.y, a1, c3);
    c3 = mad(b3.z, a2, c3);
    c3 = mad(b3.w, a3, c3);
  }

  write_imagef(C, (int2)(n + 0, batch + m_pack4 * nbatch), c0);
  write_imagef(C, (int2)(n + 1, batch + m_pack4 * nbatch), c1);
  write_imagef(C, (int2)(n + 2, batch + m_pack4 * nbatch), c2);
  write_imagef(C, (int2)(n + 3, batch + m_pack4 * nbatch), c3);
    }
    __kernel void default_function_kernel0(__read_only image2d_t filter, __write_only image2d_t mali_conv2d_nchw_winograd_U) {


 #define filter_mat mali_conv2d_nchw_winograd_U 
      const int nk_pack4 = get_image_dim(filter).x/3/3 ;
      const int nc = (get_image_dim(filter).y);

       const int k_pack4 = get_global_id(0);
  const int c = get_global_id(1);
    //if (k_pack4==0 && c==0)
     //printf("mali_conv2d_nchw_winograd_U %d,%d",get_image_dim(mali_conv2d_nchw_winograd_U).x,get_image_dim(mali_conv2d_nchw_winograd_U).y);
     //return;
  if (k_pack4 >= nk_pack4 || c >= nc) return;

  float4 g00 = read_imagef(filter, sampler, (int2)(0 + k_pack4 * 9, c));
  float4 g01 = read_imagef(filter, sampler, (int2)(1 + k_pack4 * 9, c));
  float4 g02 = read_imagef(filter, sampler, (int2)(2 + k_pack4 * 9, c));
  float4 g10 = read_imagef(filter, sampler, (int2)(3 + k_pack4 * 9, c));
  float4 g11 = read_imagef(filter, sampler, (int2)(4 + k_pack4 * 9, c));
  float4 g12 = read_imagef(filter, sampler, (int2)(5 + k_pack4 * 9, c));
  float4 g20 = read_imagef(filter, sampler, (int2)(6 + k_pack4 * 9, c));
  float4 g21 = read_imagef(filter, sampler, (int2)(7 + k_pack4 * 9, c));
  float4 g22 = read_imagef(filter, sampler, (int2)(8 + k_pack4 * 9, c));

  float4 z00 = g00;
  float4 z01 = (g00 + g01 + g02) * 0.5f;
  float4 z02 = (g00 - g01 + g02) * 0.5f;
  float4 z03 = g02;
  float4 z10 = (g00 + g10 + g20) * 0.5f;
  float4 z11 = (g00 + g01 + g02 + g10 + g11 + g12 + g20 + g21 + g22) * 0.25f;
  float4 z12 = (g00 - g01 + g02 + g10 - g11 + g12 + g20 - g21 + g22) * 0.25f;
  float4 z13 = (g02 + g12 + g22) * 0.5f;
  float4 z20 = (g00 - g10 + g20) * 0.5f;
  float4 z21 = (g00 + g01 + g02 - g10 - g11 - g12 + g20 + g21 + g22) * 0.25f;
  float4 z22 = (g00 - g01 + g02 - g10 + g11 - g12 + g20 - g21 + g22) * 0.25f;
  float4 z23 = (g02 - g12 + g22) * 0.5f;
  float4 z30 = g20;
  float4 z31 = (g20 + g21 + g22) * 0.5f;
  float4 z32 = (g20 - g21 + g22) * 0.5f;
  float4 z33 = g22;

  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x0, c), z00);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x1, c), z01);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x2, c), z02);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x3, c), z03);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x4, c), z10);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x5, c), z11);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x6, c), z12);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x7, c), z13);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x8, c), z20);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x9, c), z21);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xa, c), z22);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xb, c), z23);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xc, c), z30);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xd, c), z31);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xe, c), z32);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xf, c), z33);
}

    __kernel void default_function_kernel1(__read_only image2d_t data, __write_only image2d_t mali_conv2d_nchw_winograd_V) {

        
    #define input data
    #define input_mat mali_conv2d_nchw_winograd_V 
    const int nc_pack4 = (get_image_dim(data).y/get_image_dim(data).x);
    const int nh = get_image_dim(data).x;
    const int nh_pack_wino = (nh+2-1)/2;
    const int nw_pack_wino = nh_pack_wino;


    const int w_pack_wino = get_global_id(0);
  const int h_pack_wino = get_global_id(1);
  const int c_pack4 = get_global_id(2);
//if (w_pack_wino==0&&h_pack_wino==0&&c_pack4==0)
//printf("mali_conv2d_nchw_winograd_V %d,%d",get_image_dim(mali_conv2d_nchw_winograd_V).x,get_image_dim(mali_conv2d_nchw_winograd_V).y);
    //return;

  if (w_pack_wino >= nw_pack_wino || h_pack_wino >= nh_pack_wino || c_pack4 >= nc_pack4) return;

  const int w = (w_pack_wino << 1) - 1;
  const int h = (h_pack_wino << 1) - 1;

  float4 d00 = h >= 0 ? read_imagef(input, sampler, (int2)((w + 0), (h + 0) + c_pack4 * nh)) : 0;
  float4 d01 = h >= 0 ? read_imagef(input, sampler, (int2)((w + 1), (h + 0) + c_pack4 * nh)) : 0;
  float4 d02 = h >= 0 ? read_imagef(input, sampler, (int2)((w + 2), (h + 0) + c_pack4 * nh)) : 0;
  float4 d03 = h >= 0 ? read_imagef(input, sampler, (int2)((w + 3), (h + 0) + c_pack4 * nh)) : 0;
  float4 d10 = read_imagef(input, sampler, (int2)((w + 0), (h + 1) + c_pack4 * nh));
  float4 d11 = read_imagef(input, sampler, (int2)((w + 1), (h + 1) + c_pack4 * nh));
  float4 d12 = read_imagef(input, sampler, (int2)((w + 2), (h + 1) + c_pack4 * nh));
  float4 d13 = read_imagef(input, sampler, (int2)((w + 3), (h + 1) + c_pack4 * nh));
  float4 d20 = h + 2 < nh ? read_imagef(input, sampler, (int2)((w + 0), (h + 2) + c_pack4 * nh)) : 0;
  float4 d21 = h + 2 < nh ? read_imagef(input, sampler, (int2)((w + 1), (h + 2) + c_pack4 * nh)) : 0;
  float4 d22 = h + 2 < nh ? read_imagef(input, sampler, (int2)((w + 2), (h + 2) + c_pack4 * nh)) : 0;
  float4 d23 = h + 2 < nh ? read_imagef(input, sampler, (int2)((w + 3), (h + 2) + c_pack4 * nh)) : 0;
  float4 d30 = h + 3 < nh ? read_imagef(input, sampler, (int2)((w + 0), (h + 3) + c_pack4 * nh)) : 0;
  float4 d31 = h + 3 < nh ? read_imagef(input, sampler, (int2)((w + 1), (h + 3) + c_pack4 * nh)) : 0;
  float4 d32 = h + 3 < nh ? read_imagef(input, sampler, (int2)((w + 2), (h + 3) + c_pack4 * nh)) : 0;
  float4 d33 = h + 3 < nh ? read_imagef(input, sampler, (int2)((w + 3), (h + 3) + c_pack4 * nh)) : 0;

  float4 z00 = d00 - d02 - d20 + d22;
  float4 z01 = d01 + d02 - d21 - d22;
  float4 z02 = -d01 + d02 + d21 - d22;
  float4 z03 = d01 - d03 - d21 + d23;
  float4 z10 = d10 - d12 + d20 - d22;
  float4 z11 = d11 + d12 + d21 + d22;
  float4 z12 = -d11 + d12 - d21 + d22;
  float4 z13 = d11 - d13 + d21 - d23;
  float4 z20 = -d10 + d12 + d20 - d22;
  float4 z21 = -d11 - d12 + d21 + d22;
  float4 z22 = d11 - d12 - d21 + d22;
  float4 z23 = -d11 + d13 + d21 - d23;
  float4 z30 = d10 - d12 - d30 + d32;
  float4 z31 = d11 + d12 - d31 - d32;
  float4 z32 = -d11 + d12 + d31 - d32;
  float4 z33 = d11 - d13 - d31 + d33;

  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x0 + c_pack4 * 16), z00);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x1 + c_pack4 * 16), z01);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x2 + c_pack4 * 16), z02);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x3 + c_pack4 * 16), z03);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x4 + c_pack4 * 16), z10);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x5 + c_pack4 * 16), z11);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x6 + c_pack4 * 16), z12);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x7 + c_pack4 * 16), z13);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x8 + c_pack4 * 16), z20);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x9 + c_pack4 * 16), z21);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xa + c_pack4 * 16), z22);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xb + c_pack4 * 16), z23);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xc + c_pack4 * 16), z30);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xd + c_pack4 * 16), z31);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xe + c_pack4 * 16), z32);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xf + c_pack4 * 16), z33);
}

    __kernel void default_function_kernel3(__read_only image2d_t mali_conv2d_nchw_winograd_M, __write_only image2d_t mali_conv2d_nchw_winograd_output) {

//printf("mali_conv2d_nchw_winograd_output %d,%d",get_image_dim(mali_conv2d_nchw_winograd_output).x,get_image_dim(mali_conv2d_nchw_winograd_output).y);
   // return;
        #define output_mat mali_conv2d_nchw_winograd_M 
        #define output mali_conv2d_nchw_winograd_output 
        const int nk_pack4 = (get_image_dim(output).y/get_image_dim(output).x);
        const int nh = get_image_dim(output).x;
        const int nh_pack_wino = (nh+2-1)/2;
        const int nw_pack_wino = nh_pack_wino;


  const int w_pack_wino = get_global_id(0);
  const int h_pack_wino = get_global_id(1);
  const int k_pack4 = get_global_id(2);

  if (w_pack_wino >= nw_pack_wino || h_pack_wino >= nh_pack_wino || k_pack4 >= nk_pack4) return;

  const int w = w_pack_wino << 1;
  const int h = h_pack_wino << 1;

  float4 d00 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x0 + k_pack4 * 16));
  float4 d01 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x1 + k_pack4 * 16));
  float4 d02 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x2 + k_pack4 * 16));
  float4 d03 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x3 + k_pack4 * 16));
  float4 d10 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x4 + k_pack4 * 16));
  float4 d11 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x5 + k_pack4 * 16));
  float4 d12 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x6 + k_pack4 * 16));
  float4 d13 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x7 + k_pack4 * 16));
  float4 d20 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x8 + k_pack4 * 16));
  float4 d21 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x9 + k_pack4 * 16));
  float4 d22 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xa + k_pack4 * 16));
  float4 d23 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xb + k_pack4 * 16));
  float4 d30 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xc + k_pack4 * 16));
  float4 d31 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xd + k_pack4 * 16));
  float4 d32 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xe + k_pack4 * 16));
  float4 d33 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xf + k_pack4 * 16));

  float4 z00 = d00 + d10 + d20 + d01 + d11 + d21 + d02 + d12 + d22;
  float4 z01 = d01 + d11 + d21 - d02 - d12 - d22 - d03 - d13 - d23;
  float4 z10 = d10 - d20 - d30 + d11 - d21 - d31 + d12 - d22 - d32;
  float4 z11 = d11 - d21 - d31 - d12 + d22 + d32 - d13 + d23 + d33;

  write_imagef(output, (int2)((w + 0), (h + 0) + k_pack4 * nh), z00);
  write_imagef(output, (int2)((w + 1), (h + 0) + k_pack4 * nh), z01);
  write_imagef(output, (int2)((w + 0), (h + 1) + k_pack4 * nh), z10);
  write_imagef(output, (int2)((w + 1), (h + 1) + k_pack4 * nh), z11);
}

    
