__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;
__kernel void default_function_kernel0(__read_only image2d_t A, __read_only image2d_t W, __write_only image2d_t B) {

    //const int group_id0 = get_group_id(0);
    //const int group_id1 = get_group_id(1);
    //const int group_id2 = get_group_id(2);
    //
    //const int local_id0 = get_local_id(0);
    //const int local_id1 = get_local_id(1);
    //const int local_id2 = get_local_id(2);

      float4 B_local[4];
  float A_local[16];
  float4 W_local[4];
  const int g_14654870462301441827 = ((((((((int)get_group_id(2))*409600)+(((int)get_local_id(2))*6400))+(((int)get_group_id(1))*1280))+(((int)get_local_id(1))*320))+(((int)get_group_id(0))*16))+(((int)get_local_id(0))*8));
  vstore4(((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, (float*)B_local + 0);
  vstore4(((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, (float*)B_local + 4);
  vstore4(((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, (float*)B_local + 8);
  vstore4(((float4)(0.000000e+00f, 0.000000e+00f, 0.000000e+00f, 0.000000e+00f)), 0, (float*)B_local + 12);
  for (int rc_outer = 0; rc_outer < 64; ++rc_outer) {
    const int g_2412082800274090786 = (((rc_outer*2048)+(((int)get_group_id(2))*256))+(((int)get_local_id(2))*4));
    const int g_11908412642207760570 = (((((rc_outer*6400)+(((int)get_group_id(1))*1280))+(((int)get_local_id(1))*320))+(((int)get_group_id(0))*16))+(((int)get_local_id(0))*8));
    int4 _1 = (int4)(g_11908412642207760570, (g_11908412642207760570 + 1), (g_11908412642207760570 + 2), (g_11908412642207760570 + 3));
    vstore4(((float4)(read_imagef(A,sampler,(int2)((_1.s0 % 6400), (_1.s0 / 6400))).x,read_imagef(A,sampler,(int2)((_1.s1 % 6400), (_1.s1 / 6400))).x,read_imagef(A,sampler,(int2)((_1.s2 % 6400), (_1.s2 / 6400))).x,read_imagef(A,sampler,(int2)((_1.s3 % 6400), (_1.s3 / 6400))).x)), 0, A_local + 0);
    int4 _2 = (int4)((g_11908412642207760570 + 4), ((g_11908412642207760570 + 4) + 1), ((g_11908412642207760570 + 4) + 2), ((g_11908412642207760570 + 4) + 3));
    vstore4(((float4)(read_imagef(A,sampler,(int2)((_2.s0 % 6400), (_2.s0 / 6400))).x,read_imagef(A,sampler,(int2)((_2.s1 % 6400), (_2.s1 / 6400))).x,read_imagef(A,sampler,(int2)((_2.s2 % 6400), (_2.s2 / 6400))).x,read_imagef(A,sampler,(int2)((_2.s3 % 6400), (_2.s3 / 6400))).x)), 0, A_local + 4);
    int4 _3 = (int4)((g_11908412642207760570 + 160), ((g_11908412642207760570 + 160) + 1), ((g_11908412642207760570 + 160) + 2), ((g_11908412642207760570 + 160) + 3));
    vstore4(((float4)(read_imagef(A,sampler,(int2)((_3.s0 % 6400), (_3.s0 / 6400))).x,read_imagef(A,sampler,(int2)((_3.s1 % 6400), (_3.s1 / 6400))).x,read_imagef(A,sampler,(int2)((_3.s2 % 6400), (_3.s2 / 6400))).x,read_imagef(A,sampler,(int2)((_3.s3 % 6400), (_3.s3 / 6400))).x)), 0, A_local + 8);
    int4 _4 = (int4)((g_11908412642207760570 + 164), ((g_11908412642207760570 + 164) + 1), ((g_11908412642207760570 + 164) + 2), ((g_11908412642207760570 + 164) + 3));
    vstore4(((float4)(read_imagef(A,sampler,(int2)((_4.s0 % 6400), (_4.s0 / 6400))).x,read_imagef(A,sampler,(int2)((_4.s1 % 6400), (_4.s1 / 6400))).x,read_imagef(A,sampler,(int2)((_4.s2 % 6400), (_4.s2 / 6400))).x,read_imagef(A,sampler,(int2)((_4.s3 % 6400), (_4.s3 / 6400))).x)), 0, A_local + 12);
    int4 _5 = (int4)(g_2412082800274090786, (g_2412082800274090786 + 1), (g_2412082800274090786 + 2), (g_2412082800274090786 + 3));
    vstore4(((float4)(read_imagef(W,sampler,(int2)((_5.s0 % 512), (_5.s0 / 512))).x,read_imagef(W,sampler,(int2)((_5.s1 % 512), (_5.s1 / 512))).x,read_imagef(W,sampler,(int2)((_5.s2 % 512), (_5.s2 / 512))).x,read_imagef(W,sampler,(int2)((_5.s3 % 512), (_5.s3 / 512))).x)), 0, (float*)W_local + 0);
    int4 _6 = (int4)((g_2412082800274090786 + 512), ((g_2412082800274090786 + 512) + 1), ((g_2412082800274090786 + 512) + 2), ((g_2412082800274090786 + 512) + 3));
    vstore4(((float4)(read_imagef(W,sampler,(int2)((_6.s0 % 512), (_6.s0 / 512))).x,read_imagef(W,sampler,(int2)((_6.s1 % 512), (_6.s1 / 512))).x,read_imagef(W,sampler,(int2)((_6.s2 % 512), (_6.s2 / 512))).x,read_imagef(W,sampler,(int2)((_6.s3 % 512), (_6.s3 / 512))).x)), 0, (float*)W_local + 4);
    int4 _7 = (int4)((g_2412082800274090786 + 1024), ((g_2412082800274090786 + 1024) + 1), ((g_2412082800274090786 + 1024) + 2), ((g_2412082800274090786 + 1024) + 3));
    vstore4(((float4)(read_imagef(W,sampler,(int2)((_7.s0 % 512), (_7.s0 / 512))).x,read_imagef(W,sampler,(int2)((_7.s1 % 512), (_7.s1 / 512))).x,read_imagef(W,sampler,(int2)((_7.s2 % 512), (_7.s2 / 512))).x,read_imagef(W,sampler,(int2)((_7.s3 % 512), (_7.s3 / 512))).x)), 0, (float*)W_local + 8);
    int4 _8 = (int4)((g_2412082800274090786 + 1536), ((g_2412082800274090786 + 1536) + 1), ((g_2412082800274090786 + 1536) + 2), ((g_2412082800274090786 + 1536) + 3));
    vstore4(((float4)(read_imagef(W,sampler,(int2)((_8.s0 % 512), (_8.s0 / 512))).x,read_imagef(W,sampler,(int2)((_8.s1 % 512), (_8.s1 / 512))).x,read_imagef(W,sampler,(int2)((_8.s2 % 512), (_8.s2 / 512))).x,read_imagef(W,sampler,(int2)((_8.s3 % 512), (_8.s3 / 512))).x)), 0, (float*)W_local + 12);
    vstore4((vload4(0, (float*)B_local + 0) + (((float4*)(A_local))[0].x * vload4(0, (float*)W_local + 0))), 0, (float*)B_local + 0);
    vstore4((vload4(0, (float*)B_local + 4) + (((float4*)(A_local))[1].x * vload4(0, (float*)W_local + 0))), 0, (float*)B_local + 4);
    vstore4((vload4(0, (float*)B_local + 8) + (((float4*)(A_local))[2].x * vload4(0, (float*)W_local + 0))), 0, (float*)B_local + 8);
    vstore4((vload4(0, (float*)B_local + 12) + (((float4*)(A_local))[3].x * vload4(0, (float*)W_local + 0))), 0, (float*)B_local + 12);
    vstore4((vload4(0, (float*)B_local + 0) + (((float4*)(A_local))[0].y * vload4(0, (float*)W_local + 4))), 0, (float*)B_local + 0);
    vstore4((vload4(0, (float*)B_local + 4) + (((float4*)(A_local))[1].y * vload4(0, (float*)W_local + 4))), 0, (float*)B_local + 4);
    vstore4((vload4(0, (float*)B_local + 8) + (((float4*)(A_local))[2].y * vload4(0, (float*)W_local + 4))), 0, (float*)B_local + 8);
    vstore4((vload4(0, (float*)B_local + 12) + (((float4*)(A_local))[3].y * vload4(0, (float*)W_local + 4))), 0, (float*)B_local + 12);
    vstore4((vload4(0, (float*)B_local + 0) + (((float4*)(A_local))[0].z * vload4(0, (float*)W_local + 8))), 0, (float*)B_local + 0);
    vstore4((vload4(0, (float*)B_local + 4) + (((float4*)(A_local))[1].z * vload4(0, (float*)W_local + 8))), 0, (float*)B_local + 4);
    vstore4((vload4(0, (float*)B_local + 8) + (((float4*)(A_local))[2].z * vload4(0, (float*)W_local + 8))), 0, (float*)B_local + 8);
    vstore4((vload4(0, (float*)B_local + 12) + (((float4*)(A_local))[3].z * vload4(0, (float*)W_local + 8))), 0, (float*)B_local + 12);
    vstore4((vload4(0, (float*)B_local + 0) + (((float4*)(A_local))[0].w * vload4(0, (float*)W_local + 12))), 0, (float*)B_local + 0);
    vstore4((vload4(0, (float*)B_local + 4) + (((float4*)(A_local))[1].w * vload4(0, (float*)W_local + 12))), 0, (float*)B_local + 4);
    vstore4((vload4(0, (float*)B_local + 8) + (((float4*)(A_local))[2].w * vload4(0, (float*)W_local + 12))), 0, (float*)B_local + 8);
    vstore4((vload4(0, (float*)B_local + 12) + (((float4*)(A_local))[3].w * vload4(0, (float*)W_local + 12))), 0, (float*)B_local + 12);
  }
    int4 _9 = (int4)(g_14654870462301441827, (g_14654870462301441827 + 1), (g_14654870462301441827 + 2), (g_14654870462301441827 + 3));
    float4 _10 = vload4(0, (float*)B_local + 0);
    write_imagef(B,(int2)((_9.s0 % 6400), (_9.s0 / 6400)) , (float4)(_10.s0));
    write_imagef(B,(int2)((_9.s1 % 6400), (_9.s1 / 6400)) , (float4)(_10.s1));
    write_imagef(B,(int2)((_9.s2 % 6400), (_9.s2 / 6400)) , (float4)(_10.s2));
    write_imagef(B,(int2)((_9.s3 % 6400), (_9.s3 / 6400)) , (float4)(_10.s3));
    int4 _11 = (int4)((g_14654870462301441827 + 4), ((g_14654870462301441827 + 4) + 1), ((g_14654870462301441827 + 4) + 2), ((g_14654870462301441827 + 4) + 3));
    float4 _12 = vload4(0, (float*)B_local + 4);
    write_imagef(B,(int2)((_11.s0 % 6400), (_11.s0 / 6400)) , (float4)(_12.s0));
    write_imagef(B,(int2)((_11.s1 % 6400), (_11.s1 / 6400)) , (float4)(_12.s1));
    write_imagef(B,(int2)((_11.s2 % 6400), (_11.s2 / 6400)) , (float4)(_12.s2));
    write_imagef(B,(int2)((_11.s3 % 6400), (_11.s3 / 6400)) , (float4)(_12.s3));
    int4 _13 = (int4)((g_14654870462301441827 + 160), ((g_14654870462301441827 + 160) + 1), ((g_14654870462301441827 + 160) + 2), ((g_14654870462301441827 + 160) + 3));
    float4 _14 = vload4(0, (float*)B_local + 8);
    write_imagef(B,(int2)((_13.s0 % 6400), (_13.s0 / 6400)) , (float4)(_14.s0));
    write_imagef(B,(int2)((_13.s1 % 6400), (_13.s1 / 6400)) , (float4)(_14.s1));
    write_imagef(B,(int2)((_13.s2 % 6400), (_13.s2 / 6400)) , (float4)(_14.s2));
    write_imagef(B,(int2)((_13.s3 % 6400), (_13.s3 / 6400)) , (float4)(_14.s3));
    int4 _15 = (int4)((g_14654870462301441827 + 164), ((g_14654870462301441827 + 164) + 1), ((g_14654870462301441827 + 164) + 2), ((g_14654870462301441827 + 164) + 3));
    float4 _16 = vload4(0, (float*)B_local + 12);
    write_imagef(B,(int2)((_15.s0 % 6400), (_15.s0 / 6400)) , (float4)(_16.s0));
    write_imagef(B,(int2)((_15.s1 % 6400), (_15.s1 / 6400)) , (float4)(_16.s1));
    write_imagef(B,(int2)((_15.s2 % 6400), (_15.s2 / 6400)) , (float4)(_16.s2));
    write_imagef(B,(int2)((_15.s3 % 6400), (_15.s3 / 6400)) , (float4)(_16.s3));
}

