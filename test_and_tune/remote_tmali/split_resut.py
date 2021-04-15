import os
import json
import sys
import tarfile
import glob
import csv
import shutil
import logging
import numpy as np
import conv_shape_tunning
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s')
out_name_format = '__exp_{}-{}.log'


def convert_tupleshape_list(shape, need_rev=True):
    if isinstance(shape, tuple):
        shape = list(shape)

    if len(shape) == 11:
        shape.pop(1)
    if len(shape) == 10:
        shape.pop(-1)
    for ind, sp in enumerate(shape):
        if isinstance(sp, tuple):
            shape[ind] = list(sp)
    if need_rev:
        shape[3], shape[4] = shape[4], shape[3]
    return shape

def in_check_point(shape:str, log_file:str,need_rev=True) -> str:
    import json
    shape=convert_tupleshape_list(shape, need_rev)
    if not os.path.exists(log_file):
        return ''
    with open(log_file) as fp:
        for i, sp_hist in enumerate(fp):
            sp_hist_json = json.loads(sp_hist)
            if sp_hist_json["input"][2] == shape:
                return sp_hist
    return ''

def split_for_network(logfile:str, convshape:dict):
    sys.path.append('remote_tmali')
    sys.path.append('./')

    tasks=[]
    log_base=os.path.basename(logfile)
    device=log_base.split('.')[0].split('_')[-1]
    for net, shapes in convshape.items():
        out_name = out_name_format.format(net ,device)
        with open(out_name, 'a') as fp:
            for ith, shape in enumerate(shapes):
                record_sp = in_check_point(shape, logfile)
                #assert record_sp!='','not found'+str(shape)+" "+device
                if record_sp == '':
                    logging.debug('not found' + str(shape) + " " + device + " " + net)
                    continue
                fp.write(record_sp)

def conv1x1(device):
    logfile = 'best_'+device + ".topinhw4c1.log"
    a = os.path.dirname(sys.argv[0])
    b = os.getcwd()
    if len(a) < len(b):
        a, b = b, a
    prefix = a[len(b) + len(a) - len(b) :]
    prefix = '.' if prefix == '' else prefix
    split_for_network(prefix + '/' + logfile, conv_shape_tunning.convshape)


def conv3x3(device):
    logfile = 'best_'+device + '.wino3x3.log'
    #logfile = '_from_rd_'+device + '.log'
    a = os.path.dirname(sys.argv[0])
    b = os.getcwd()
    if len(a) < len(b):
        a, b = b, a
    prefix = a[len(b) + len(a) - len(b):]
    prefix = '.' if prefix == '' else prefix
    split_for_network(prefix + '/' + logfile, conv_shape_tunning.conv3x3shape)

def convdepthwise(device):
    logfile = device + '.depthwise_rp.log'
    a = os.path.dirname(sys.argv[0])
    b = os.getcwd()
    if len(a) < len(b):
        a, b = b, a
    prefix = a[len(b) + len(a) - len(b):]
    prefix = '.' if prefix == '' else prefix
    split_for_network(prefix + '/' + logfile, conv_shape_tunning.conv_depthwise)

def for_multi_device():
    #clean previous results
    files = glob.glob("__exp_*")
    for f in files:
        os.remove(f)
    devices=['Adreno630','Adreno640',"MaliG76"]
    for device in devices:
        conv1x1(device)
        conv3x3(device)
        convdepthwise(device)


def merge_multi_log(log_list:list, bi_convshape:dict):
    log_l_l = []
    for logfile in log_list:
        log_l_l.append([])
        with open(logfile) as fp:
            for log_shape_one in fp:
                log_l_l[-1].append(json.loads(log_shape_one))
    def find_best_from_shape_log(shape_hist, log):
        for i, sp_hist_json in enumerate(log):
            if sp_hist_json["input"][2] == shape_hist["input"][2]:
                if sp_hist_json["result"][0][0] < shape_hist["result"][0][0]:
                    return sp_hist_json, i
                else:
                    return shape_hist, i
        return shape_hist, -1
    best_shape_hist=[]
    iconv=0
    for net, convshapes in bi_convshape.items():
        for shape in convshapes:
            shape=convert_tupleshape_list(shape)
            log_shape = {"input":["","",shape],"result":[[1e10]]}
            best_i=0
            for i, log_one_hists in enumerate(log_l_l):
                log_shape1, ind = find_best_from_shape_log(log_shape, log_one_hists)
                if log_shape1["result"][0][0] != log_shape["result"][0][0]:
                    best_i=i
                log_shape=log_shape1
            #if best_i != 2:
            #    logging.debug(f"find best i{iconv}th {shape} perf from {best_i} {log_shape['result'][0][0]} vs {log_l_l[2][ind]['result'][0][0]}")
            iconv=iconv+1
            best_shape_hist.append(log_shape)
    best_name = os.path.dirname(log_list[0])+'best_'+os.path.basename(log_list[0])
    with open(best_name, 'w') as fp:
        for log_shape in best_shape_hist:
            fp.write(json.dumps(log_shape) + '\n')


def extract_depthsize_from_e2etune(logf):
    all_depthwise =[]
    with open(logf) as fp:
        for l in fp:
            depthwise_tc_rev = json.loads(l)
            if "depthwise_conv2d_NCHWc" not in depthwise_tc_rev["input"][1]:
                continue
            df_shape = depthwise_tc_rev["input"][2]
            depthwise_shape = df_shape[0][1] + df_shape[1][1]
            df_shape[1][1][0], df_shape[1][1][1] = df_shape[1][1][1], df_shape[1][1][
                0]
            depthwise_shape.append(df_shape[2])
            depthwise_shape.append(df_shape[3])
            depthwise_shape.pop(1)

            depthwise_tc_rev["input"][2] = depthwise_shape
            all_depthwise.append(depthwise_tc_rev)
    return all_depthwise

def append_depthwise_to_log(filename='', device=''):
    if filename:
        logfile = f'climgfloatr32-{device}-mobilenet.log'
        depthsise_shape = extract_depthsize_from_e2etune(logfile)
        all_shape_logfile = filename
        with open(all_shape_logfile) as fp:
            for l in fp:
                if "depthwise_conv2d_NCHWc" in json.loads(l)["input"][1]:
                    return
        with open(all_shape_logfile, 'a') as fp:
            for one_depth in depthsise_shape:
                fp.write(json.dumps(one_depth) + '\n')
        return
    devices = ['Adreno630', 'Adreno640', "MaliG76"]
    for device in devices:
        logfile = f'climgfloatr32-{device}-mobilenet.log'
        depthsise_shape = extract_depthsize_from_e2etune(logfile)
        all_shape_logfile = device + ".topinhw4c1.log"
        with open(all_shape_logfile) as fp:
            for l in fp:
                if "depthwise_conv2d_NCHWc" in json.loads(l)["input"][1]:
                    return
        with open(all_shape_logfile, 'a') as fp:
            for one_depth in depthsise_shape:
                fp.write(json.dumps(one_depth)+'\n')


def select_best_for_multi_devices():
    devices = ['Adreno630', 'Adreno640', "MaliG76"]
    for device in devices:
        logfile = "{device}.topinhw4c1.log".format(device=device)
        logfile1 = logfile+'1'
        rc_log_name='{device}.topinhw4c1_rp.log'.format(device=device)
        merge_multi_log([logfile, logfile1, rc_log_name], conv_shape_tunning.convshape)
    for device in devices:
        logfile = "{device}.wino3x3.log".format(device=device)
        rc_log_name = '_from_rd_{device}.log'.format(
            device=device)
        merge_multi_log([logfile, rc_log_name], conv_shape_tunning.conv3x3shape)


def tar_all_files():
    tar = tarfile.open("gpu_exp_data.tar", "w")
    devices = ['Adreno630', 'Adreno640', "MaliG76"]
    for net, _ in conv_shape_tunning.convshape.items():
        for device in devices:
            out_name = out_name_format.format(net, device)
            tar.add(out_name)
    tar.close()


def restore_shape_for_even(src_log_format, dst_log_format, shapedict):
    devices = ['Adreno640', 'Adreno630', "MaliG76"]

    def extract_tune_rec(shape, all_shapes_res, log_name):
        new_shape = list(shape)
        new_shape[4]=new_shape[1]
        new_shape[2]=(1+new_shape[2])//2*2
        new_shape[3]=(1+new_shape[3])//2*2
        sp=in_check_point(new_shape,log_name)
        assert sp != '',f"not found {shape} for {device}"
        sp_sp = json.loads(sp)
        sp_sp["input"][2][1]=shape[2]
        sp_sp["input"][2][2]=shape[3]
        all_shapes_res.append(json.dumps(sp_sp))
    for device in devices:
        log_name=src_log_format.format(device=device)
        all_shapes_res=[]
        #normal conv
        for net, convshapes in shapedict.items():
            for i, shape in enumerate(convshapes):
                extract_tune_rec(shape, all_shapes_res, log_name)
        out_name=dst_log_format.format(device=device)
        with open(out_name, 'w') as fp:
            for f in all_shapes_res:
                fp.write(f+'\n')

def restore_shape():
    conv1x1_log_name_format='{device}.topinhw4c1-m_packed.log'
    dw_log_name_format='{device}.depthwise_packed.log'
    out_log_name_format='{device}.topinhw4c1_rp.log'
    outdw_log_name_format='{device}.depthwise_rp.log'
    restore_shape_for_even(conv1x1_log_name_format,out_log_name_format,conv_shape_tunning.convshape)
    restore_shape_for_even(dw_log_name_format,outdw_log_name_format,conv_shape_tunning.conv_depthwise)

def convert_tunelog_to_topi_log():
    devices = ['Adreno640', 'Adreno630', "MaliG76"]
    log_name_format='{device}.topi_nchw4c.log'
    out_logfile_format = "{device}.topinhw4c1_packed_m.log"
    ori_log_format='{device}.topinhw4c1_packed1.log'
    for device in devices:
        log_name=log_name_format.format(device=device)
        out_name=out_logfile_format.format(device=device)
        ori_name=ori_log_format.format(device=device)
        all_shapes=[]
        with open(ori_name) as fp:
            for f in fp:
                shape_template=json.loads(f)
                #new_shape=shape_js['input'][2]
                #new_shape[1]=(1+new_shape[1])//2*2
                #new_shape[2]=(1+new_shape[2])//2*2
                #sp=in_check_point(new_shape,log_name,need_rev=False)
                #if sp != '':
                #    sp=json.loads(sp)
                #    shape_js['result'][0][0]=sp['result'][0][0]
                #    all_shapes.append(json.dumps(shape_js))
        with open(log_name) as fp:
            for f in fp:
                be_convert_shape = json.loads(f)
                shape_template['input'][2]=be_convert_shape['input'][2]
                all_shapes.append(json.dumps(shape_template))
        with open(out_name,'w') as fp:
            for packed_shape in all_shapes:
                fp.write(packed_shape+'\n')

def unpack_to_original_shape():
    devices = ['Adreno640', 'Adreno630', "MaliG76"]
    for device in devices:
        log_name_format='{device}.topi_nchw4c.log'.format(device=device)
        pi_logfile_format = "{device}.topinhw4c1_packed.log".format(device=device)
        out_logfile_format = "{device}.topinhw4c1-m_packed.log".format(device=device)
        all_shape_dict={}
        with open(log_name_format) as fp:
            for f in fp:
                sp = json.loads(f)
                all_shape_dict[str(sp['input'][2])]=json.dumps(sp)
        with open(pi_logfile_format) as fp:
            for f in fp:
                sp = json.loads(f)
                sp_k = str(sp['input'][2])
                if sp_k not in all_shape_dict:
                    all_shape_dict[sp_k]=json.dumps(sp)
                #else:
                #logging.debug('already exist',sp_k)
        with open(out_logfile_format, 'w') as fp:
            for spk, spv in all_shape_dict.items():
                fp.write(spv+'\n')

def show_every_step_perf():
    devices = ['Adreno640','MaliG76']
    features=['default','layout_nchw4c','buffer+image','common_sub_expr','texture_address_primitive','cl_half_float_storage']
    for device in devices:
        logf = f'{device}.maliexp.log'
        all_shape_conf=[]
        with open(logf) as fp:
            for ith, f in enumerate(fp):
                if ith in [4,5,6,7]:continue
                sp=json.loads(f)
                all_shape_conf.append([features[(23-ith)//4], sp["input"][2],sp["result"][0][0]])
            all_shape_conf.reverse()
            n_all_shape_conf=[]
            for i in range(4):
                for j in range(len(all_shape_conf)//4):
                    n_all_shape_conf.append(all_shape_conf[i+j*4])
            with open(f'{device}_feature_exp.csv','w') as fpw:
                csv_writer = csv.writer(fpw)
                for l in n_all_shape_conf:
                    csv_writer.writerow(l)
    tar = tarfile.open("feature_exp.tar", "w")
    devices = ['Adreno640', "MaliG76"]
    for device in devices:
        out_name = f'{device}_feature_exp.csv'
        tar.add(out_name)
    tar.close()



if __name__ == '__main__':
    #show_every_step_perf()
    unpack_to_original_shape()
    #convert_tunelog_to_topi_log()
    restore_shape()
    select_best_for_multi_devices()
    for_multi_device()
    tar_all_files()