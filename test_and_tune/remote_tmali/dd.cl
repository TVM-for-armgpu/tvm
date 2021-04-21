__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP| CLK_FILTER_NEAREST;
__kernel void default_function_kernel2(__read_only image2d_t mali_conv2d_nchw_winograd_M, __write_only image2d_t mali_conv2d_nchw_winograd_output) {


        #define output_mat mali_conv2d_nchw_winograd_M 
        #define output mali_conv2d_nchw_winograd_output 

        const int nh_pack_wino = (get_image_dim(output).x+2-1)/2;
        const int nw_pack_wino = nh_pack_wino;
        const int nk_pack4 = (get_image_dim(output).y/get_image_dim(output).x);
        const int nh = get_image_dim(output).x;

    const int w_pack_wino = get_global_id(0);
    const int h_pack_wino = get_global_id(1);
    const int k_pack4 = get_global_id(2);
    //if (w_pack_wino==0 && h_pack_wino==0 && k_pack4==0 ){
    //  printf("mali_conv2d_nchw_winograd_output %d,%d,%d,%d\n",nh_pack_wino,nw_pack_wino,nk_pack4,nh);
    //}
    // return;
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

    float4 z00 = d00 + d01 + d02 + d10 + d11 + d12 + d20 + d21 + d22;
    float4 z01 = -d01 + d02 +d03 - d11 + d12 + d13 - d21 + d22 + d23;
    float4 z10 = -d10 - d11 - d12  + d20 + d21 + d22 + d30 + d31 + d32;
    float4 z11 = d11 - d12 - d13 - d21 + d22 + d23 - d31 + d32 + d33;

    write_imagef(output, (int2)((w + 0), (h + 0) + k_pack4 * nh), z00);
    write_imagef(output, (int2)((w + 1), (h + 0) + k_pack4 * nh), z01);
    write_imagef(output, (int2)((w + 0), (h + 1) + k_pack4 * nh), z10);
    write_imagef(output, (int2)((w + 1), (h + 1) + k_pack4 * nh), z11);
    }
    __kernel void default_function_kernel0(__read_only image2d_t placeholder, __write_only image2d_t mali_conv2d_nchw_winograd_V) {

        
     #define input placeholder 
    #define input_mat mali_conv2d_nchw_winograd_V 

    const int nw_pack_wino = (get_image_dim(input).x+2-1)/2;
    const int nh_pack_wino = nw_pack_wino;
    const int nc_pack4 = (get_image_dim(input).y/get_image_dim(input).x);
    const int nh = get_image_dim(input).x;


      const int w_pack_wino = get_global_id(0);
    const int h_pack_wino = get_global_id(1);
    const int c_pack4 = get_global_id(2);

    //if (w_pack_wino==0&&h_pack_wino==0&&c_pack4==0)
    //{printf("mali_conv2d_nchw_winograd_V %d,%d %d %d\n",nw_pack_wino,nh_pack_wino,nc_pack4,nh);}
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
        //if (w_pack_wino==0&&h_pack_wino==0&&c_pack4==0){
        //    printf("%f,%f,%f,%f  %f,%f,%f,%f   %f,%f,%f,%f  %f,%f,%f,%f\n",d00.x,d01.x,d02.x,d03.x,d10.x,d11.x,d12.x,d13.x,d20.x,d21.x,d22.x,d23.x,d30.x,d31.x,d32.x,d33.x);
        //    printf("%f,%f,%f,%f  %f,%f,%f,%f   %f,%f,%f,%f  %f,%f,%f,%f\n",d00.y,d01.y,d02.y,d03.y,d10.y,d11.y,d12.y,d13.y,d20.y,d21.y,d22.y,d23.y,d30.y,d31.y,d32.y,d33.y);
        //    printf("%f,%f,%f,%f  %f,%f,%f,%f   %f,%f,%f,%f  %f,%f,%f,%f\n",d00.z,d01.z,d02.z,d03.z,d10.z,d11.z,d12.z,d13.z,d20.z,d21.z,d22.z,d23.z,d30.z,d31.z,d32.z,d33.z);
        //    printf("%f,%f,%f,%f  %f,%f,%f,%f   %f,%f,%f,%f  %f,%f,%f,%f\n",d00.w,d01.w,d02.w,d03.w,d10.w,d11.w,d12.w,d13.w,d20.w,d21.w,d22.w,d23.w,d30.w,d31.w,d32.w,d33.w);
        //}
    float4 z00 = d00 - d02 - d20 + d22;
    float4 z01 = -d01 + d02 + d21 - d22;
    float4 z02 = d01 + d02 - d21 - d22;
    float4 z03 = -d01 + d03 + d21 - d23;

    float4 z10 = -d10 + d12 + d20 - d22;
    float4 z11 = d11 - d12 - d21 + d22;
    float4 z12 = -d11 - d12 + d21 + d22;
    float4 z13 = d11 - d13 - d21 + d23;

    float4 z20 = d10 - d12 + d20 - d22;
    float4 z21 = -d11 + d12 - d21 + d22;
    float4 z22 = d11 + d12 + d21 + d22;
    float4 z23 = -d11 + d13 - d21 + d23;

    float4 z30 = -d10 + d12 + d30 - d32;
    float4 z31 = d11 - d12 - d31 + d32;
    float4 z32 = -d11 - d12 + d31 + d32;
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

    __kernel void default_function_kernel1(__read_only image2d_t mali_conv2d_nchw_winograd_U, __read_only image2d_t mali_conv2d_nchw_winograd_V, __write_only image2d_t mali_conv2d_nchw_winograd_M) {
  

    #define A mali_conv2d_nchw_winograd_U 
    #define B mali_conv2d_nchw_winograd_V 
    #define C mali_conv2d_nchw_winograd_M 

    const int nbatch=4*4;
    const int nm_pack4=get_image_dim(mali_conv2d_nchw_winograd_U).x/4/4;
    const int nk_pack4=get_image_dim(mali_conv2d_nchw_winograd_U).y/4;
    const int nn_pack4=(get_image_dim(mali_conv2d_nchw_winograd_V).x+3)/4;

    const int n_pack4 = get_global_id(0);
    const int m_pack4 = get_global_id(1);
    const int batch = get_global_id(2);
    //if (n_pack4==0&&m_pack4==0&&batch==0)
    //printf("mali_conv2d_nchw_winograd_M %d,%d.%d...\n",nm_pack4,nk_pack4,nn_pack4);
    //return;

    if (batch >= nbatch || n_pack4 >= nn_pack4 || m_pack4 >= nm_pack4) return;

    const int n = n_pack4 << 2;

    float4 a0, a1, a2, a3;
    float4 b0, b1, b2, b3;
    float4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    for (short k_pack4 = 0; k_pack4 < nk_pack4; k_pack4 += 1) {
        const int k = k_pack4 << 2;

        a0 = read_imagef(A, sampler, (int2)(m_pack4 * nbatch + batch, k + 0));
        a1 = read_imagef(A, sampler, (int2)(m_pack4 * nbatch + batch, k + 1));
        a2 = read_imagef(A, sampler, (int2)(m_pack4 * nbatch + batch, k + 2));
        a3 = read_imagef(A, sampler, (int2)(m_pack4 * nbatch + batch, k + 3));

        b0 = read_imagef(B, sampler, (int2)(n + 0, batch + k_pack4 * nbatch));
        b1 = read_imagef(B, sampler, (int2)(n + 1, batch + k_pack4 * nbatch));
        b2 = read_imagef(B, sampler, (int2)(n + 2, batch + k_pack4 * nbatch));
        b3 = read_imagef(B, sampler, (int2)(n + 3, batch + k_pack4 * nbatch));

       // if (m_pack4 == 1 && n_pack4 == 0 && batch == 0) {
       //     float4 d00 = a0, d01 = a1, d02 = a2, d03 = a3, d10 = b0, d11 = b1, d12 = b2, d13 = b3;
       //     printf("%.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f   %.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f\n", d00.x, d01.x, d02.x, d03.x, d10.x, d11.x, d12.x, d13.x);
       //     printf("%.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f   %.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f\n", d00.y, d01.y, d02.y, d03.y, d10.y, d11.y, d12.y, d13.y);
       //     printf("%.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f   %.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f\n", d00.z, d01.z, d02.z, d03.z, d10.z, d11.z, d12.z, d13.z);
       //     printf("%.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f   %.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f\n\n", d00.w, d01.w, d02.w, d03.w, d10.w, d11.w, d12.w, d13.w);
       // }


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
    //if (m_pack4 == 1 && n_pack4 == 0 && batch == 0) {
    //    float4 d00 = c0, d01 = c1, d02 = c2, d03 = c3;
    //    printf("%.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f   %.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f\n", d00.x, d01.x, d02.x, d03.x);
    //    printf("%.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f   %.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f\n", d00.y, d01.y, d02.y, d03.y);
    //    printf("%.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f   %.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f\n", d00.z, d01.z, d02.z, d03.z);
    //    printf("%.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f   %.1f,%.1f,%.1f,%.1f  %.1f,%.1f,%.1f,%.1f\n\n", d00.w, d01.w, d02.w, d03.w);
    //}
    }
    
