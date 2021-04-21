import pandas as pd
import json

mace_all_op = []


def get_pad_size(fs, pad):
    #oihw
    if 'VALID' == pad:
        return [0, 0, 0, 0]
    elif 'SAME' == pad:
        l,u = (fs[2] - 1) // 2, (fs[3] - 1)//2
        return [l, u, l, u]
    else:
        pad = eval(pad)
        if isinstance(pad, list) and len(pad) == 2:
            pad = [pad[0], pad[1], pad[0], pad[1]]
            return pad
        else:
            assert 0


#output N = (W − F + 2P )/S+1
def get_input_size(fs, os, pad, stride):
    ins = [1,os[1],os[2],fs[1]]
    ins[1] = (ins[1]) * stride[0] + (
        fs[2] - 1) - pad[0] - pad[2] - 1 * (stride[0] > 1)
    ins[2] = (ins[2]) * stride[1] + (
        fs[3] - 1) - pad[1] - pad[3] - 1 * (stride[1] > 1)

    assert ins[1] == ins[2]
    return ins


def get_shape_same_tvm(fs, os, stride, pad):
    pad_size = get_pad_size(fs, pad)
    ins = get_input_size(fs, os, pad_size, stride)
    s_tvm_shape = ins + fs
    s_tvm_shape.append(stride)
    s_tvm_shape.append(pad_size)

    s_tvm_shape.pop(3)
    return s_tvm_shape

def get_shape_im_repre(fs, os, stride, pad, op_type):
    #special case===============begin=================
    if os[1] == 111:
        pad='[3,3]'
        os[1], os[2] = [112, 112]
    if os[1] in [55,25,26, 11] and op_type == 'wino2x2':
        pad = '[1,1]'
    #special case===============end=================
    pad_size = get_pad_size(fs, pad)
    s_im_rep_shape = os + fs
    s_im_rep_shape.append(stride)
    s_im_rep_shape.append(pad_size)
    s_im_rep_shape.pop(3)
    #un_pack cin
    if s_im_rep_shape[4] == 3:
        s_im_rep_shape[4] = 4
    #handle depthwise
    if op_type == 'dw':
        s_im_rep_shape[3] = s_im_rep_shape[4]



    ins = get_input_size(fs, os, pad_size, stride)
    s_tvm_shape = ins + fs
    s_tvm_shape.append(stride)
    s_tvm_shape.append(pad_size)
    s_tvm_shape.pop(3)
    #un_pack cin
    if s_tvm_shape[4] == 3:
        s_tvm_shape[4] = 4
    #handle depthwise
    if op_type == 'dw':
        s_tvm_shape[3]=s_tvm_shape[4]
    #input-styled shape in tvm for log and debug
    s_im_rep_shape.append(s_tvm_shape)
    return s_im_rep_shape


def load_tvm_shape(filename):
    all_rec = []
    with open(filename) as fp:
        for l in fp:
            all_rec.append(json.loads(l))
    return all_rec


def load_default_tvm_shape(filename):
    all_rec = []
    with open(filename) as fp:
        for l in fp:
            depthwise_tc_rev = json.loads(l)
            df_shape = depthwise_tc_rev["input"][2]
            #df_shape[1][1][0], df_shape[1][1][1] = df_shape[1][1][1], df_shape[1][
            #    1][0]
            depthwise_shape = df_shape[0][1] + df_shape[1][1]
            depthwise_shape.append(df_shape[2])
            depthwise_shape.append(df_shape[3])
            depthwise_shape.pop(1)
            depthwise_tc_rev["input"][2] = depthwise_shape
            all_rec.append(depthwise_tc_rev)
    return all_rec


#O=(I-K+2P)/S+1
def get_tvm_output_size(shape: list) -> list:
    _, H, W, CO, CI, KH, KW, stride, pad = shape
    OH = (H-KH+pad[0]+pad[2])//stride[0]+1
    OW = (W - KW + pad[1] + pad[3]) // stride[1] + 1
    out_shape = [1, OH, OW, CO, CI, KH, KW, stride, pad]
    return out_shape


def find_shape_cost_from_tvm(tvm_costs, shape, mace_index) -> float:
    shape_tvm = shape.pop(-1)
    for tvm_l in tvm_costs:
        if get_tvm_output_size(tvm_l["input"][2]) == shape:
            return tvm_l["result"][0][0], tvm_l["input"][1]
    print(shape_tvm, f'not found in tvm result for {mace_index} op')
    return 1e10, tvm_l["input"][1]


def match_mace_and_tvm(mace_op_fn, tvm_fn, col1):
    df = pd.read_csv(mace_op_fn, sep=',')
    df.insert(df.shape[1], col1, 1e10)
    df.insert(df.shape[1], col1+"_op_name", '')
    print(df.head(5))
    matched = 0
    if 'default' in col1:
        tvm_costs = load_default_tvm_shape(tvm_fn)
    else:
        tvm_costs = load_tvm_shape(tvm_fn)
    for index, row in df.iterrows():
        fs, os, stride, pad = row['filter_shape'], row['output_shape'], row[
            ' Stride '], row['   Pad ']
        pad = pad.strip()
        os = eval(os)
        fs = eval(fs)
        stride = eval(stride)
        mace_shape = get_shape_im_repre(fs, os, stride, pad, row['tips'])
        cost, opname = find_shape_cost_from_tvm(tvm_costs, mace_shape, index)
        if cost != 1e10:
            matched = matched+1
        df.loc[index, col1] = cost
        df.loc[index, col1 + "_op_name"] = opname
        #print(mace_shape, cost)
        #break
    print(df.head(5))
    print(f"total {matched} op matched successfully in {len(tvm_costs)} tvm op")
    df.to_csv('data_df.csv')


if __name__ == "__main__":
    mace_630 = 'op_mace_adreno640.csv'
    tvm_630 = '__640.log'
    match_mace_and_tvm(mace_630, tvm_630, 'improved_tvm_cost')
    #match_mace_and_tvm(mace_630, tvm_630, 'default_tvm_cost')
