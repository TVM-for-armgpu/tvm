def conv2d_NCHWc_io_op_common(data, kernel, stride, padding, dilation, layout, out_layout, use_rf, out_dtype):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    """Compute conv2d with NCHWc layout."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        ic_chunk_group, oc_chunk, kernel_height, kernel_width, _, oc_bn = get_const_tuple(
            kernel.shape
        )
        #assert kernel_height==1 and kernel_width==1, "only for conv1x1"
        if ic_chunk == 0:
            ic_chunk=1
        if oc_chunk == 0:
            ic_chunk=1
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        _, num_filter, kernel_height, kernel_width = get_const_tuple(
            kernel.shape)
        oc_bn = 4
        ic_bn = 4
        if in_channel == 0:
            in_channel=1
        if num_filter == 0:
            num_filter=1
        ic_chunk = (in_channel+3)//ic_bn
        oc_chunk = (num_filter+3)//oc_bn
        in_channel = ic_chunk * ic_bn

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, ic_chunk, ih, iw, ic_bn)
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (
                in_channel,
                oc_chunk,
                kernel_height,
                kernel_width,
                1,
                oc_bn,
            )
            kernel = tvm.te.placeholder(
                kshape, kernel.dtype, name="kernel_vec")
        else:
            data, kernel = _pack_data(data, kernel)


    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )
    assert (#kernel_height == 1 and kernel_width == 1 and
            (dilation_h == 1 or dilation_h == 0) and
            (dilation_w == 1 or dilation_w == 0)), " only conv 1x1 support"

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = padding if isinstance(
        padding, (tuple, list)) else (padding, padding, padding, padding)
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)
    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD and kernel_height > 1:
        data_pad = nn.pad(data, pad_before, pad_after, name="data_pad_3x3")
    else:
        data_pad = data
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    def get_factors_for_rf():
        for factor in range(4, 8):
            if (ic_chunk//factor)*factor == ic_chunk:
                return factor
        return 1
    rf_n= get_factors_for_rf()
    ic_rf_thread = (ic_chunk//rf_n)*ic_bn
    if ic_rf_thread != in_channel and use_rf:
        assert n==1,'only batch==1 is support'
        rf_oshape = (n, oc_chunk, out_height, out_width*rf_n, oc_bn)

        irf = te.reduce_axis((0, rf_n), name="irf")
        ic = te.reduce_axis((0, ic_rf_thread), name="ic")
        kh = te.reduce_axis((0, kernel_height), name="kh")
        kw = te.reduce_axis((0, kernel_width), name="kw")

        idxdiv = tvm.tir.indexdiv
        idxmod = tvm.tir.indexmod

        conv_out_s1 = te.compute(
            rf_oshape,
        lambda n_, oc_chunk, oh, ow, oc_block: op_mad.mad(
            op_mad.mymul(
                data_pad[
                    n_,
                    idxdiv((ow%rf_n)*ic_rf_thread +ic, ic_bn),
                    oh * HSTR + kh * dilation_h,
                    ow//rf_n * WSTR + kw * dilation_w,
                    idxmod(ic, ic_bn),
                ].astype(data.dtype)
                , kernel[(ow%rf_n)*ic_rf_thread +ic,oc_chunk, kh,
                        kw, 0, oc_block]
            ),
            axis=[ic, kh, kw],
            ),
            name="conv2d_NCHWc",
            tag="conv2d_NCHWc",
        )
        #conv_out=conv_out_s1
        conv_out = te.compute(
            oshape,
            lambda n_, oc_chunk_, oh, ow, oc_block: te.sum(
                    conv_out_s1[
                        n_,
                        oc_chunk_,
                        oh,
                        ow//rf_n+irf,
                        oc_block,
                    ].astype(data.dtype),
                    axis=[irf],
                ),

            name="conv2d_NCHWc",
            tag="conv2d_NCHWc_rf",
        )
    else:
        ic = te.reduce_axis((0, in_channel), name="ic")
        kh = te.reduce_axis((0, kernel_height), name="kh")
        kw = te.reduce_axis((0, kernel_width), name="kw")

        idxdiv = tvm.tir.indexdiv
        idxmod = tvm.tir.indexmod

        conv_out = te.compute(
            oshape,
            lambda n, oc_chunk, oh, ow, oc_block: op_mad.mad(
                op_mad.mymul(
                    data_pad[
                        n,
                        idxdiv(ic, ic_bn),
                        oh * HSTR + kh * dilation_h,
                        ow * WSTR + kw * dilation_w,
                        idxmod(ic, ic_bn),
                    ].astype(data.dtype)
                    , kernel[ic,oc_chunk, kh,
                            kw, 0, oc_block]
                ),
                axis=[ic, kh, kw],
            ),
            name="conv2d_NCHWc",
            tag="conv2d_NCHWc",
        )
    conv_out.dtype = out_dtype
    flops = n * oc_chunk * out_height * out_width * oc_bn * kernel_height * kernel_width * ic_chunk * ic_bn * 2
    return conv_out, flops
