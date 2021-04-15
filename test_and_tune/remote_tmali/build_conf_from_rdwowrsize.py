import os
import json
import re
import os

template = r'{"input": ["opencl -keys=mali,opencl,gpu -device=mali -max_num_threads=256", "winograd_2", [1, 56, 56, 64, 64, 3, 3, [1, 1], [1, 1, 1, 1]], {}], "config": {"index": 34178780, "code_hash": null, "entity": [["inv_cp", "sp", [-1, 2]], ["inv_wp", "sp", [-1, 28, 2]], ["inv_hp", "sp", [-1, 7, 2]], ["kernel_cp", "sp", [-1, 2]], ["kernel_kp", "sp", [-1, 4]], ["data_cp", "sp", [-1, 32]], ["data_wp", "sp", [-1, 256, 1]], ["data_hp", "sp", [-1, 8, 1]], ["bgemm_kp", "sp", [-1, 16]], ["bgemm_wp", "sp", [-1, 28, 4]], ["bgemm_hp", "sp", [-1, 1]]]}, "result": [[0.0007045376], 0, 5.030421018600464, 1617974250.8814242], "version": 0.2, "tvm_version": "0.8.dev0"}'

devices = ['Adreno640', 'Adreno630', 'MaliG76']
log_format = 'log.{}.log'

for device in devices:
    work_group_conf = []
    logf = log_format.format(device)
    with open(logf) as fp:
        for i in range(16):
            shape = fp.readline().strip()
            fp.readline()
            trans = fp.readline().strip()
            matmul = fp.readline().strip()
            inv = fp.readline().strip()
            filte = fp.readline().strip()
            perf = fp.readline()
            shape_n = re.findall(r'tuningK=(\d+),C=(\d+),H=(\d+),W=(\d+),wino=2',
                                shape)
            shape_n = list(shape_n[0])
            shape_n = [int(s) for s in shape_n]
            shape_n[0:4] = [shape_n[2], shape_n[3], shape_n[0], shape_n[1]]
            trans_n = re.findall(r'(\d{0,3}),(\d{0,3}),(\d{0,3})', trans)[0]
            trans_n = [int(s) for s in trans_n]
            matmul_n = re.findall(r'(\d{0,3}),(\d{0,3}),(\d{0,3})', matmul)[0]
            matmul_n = [int(s) for s in matmul_n]
            inv_n = re.findall(r'(\d{0,3}),(\d{0,2}),(\d{0,3})', inv)[0]
            inv_n = [int(s) for s in inv_n]
            filte_n = re.findall(r'(\d{0,3}),(\d{0,3}),(\d{0,3})', filte)[0]
            filte_n = [int(s) for s in filte_n]
            perf_n = re.findall(r'.*in(.*)us', perf)[0]
            perf_n=float(perf_n)
            js_conf = json.loads(template)
            if shape_n[1] in [73, 149]:
                js_conf["input"][2] = [1] + shape_n+ [3, 3, [1, 1], [0,0,0,0]]
            else:
                js_conf["input"][2] = [1] + shape_n+ [3, 3, [1, 1], [1, 1, 1, 1]]
            js_conf["result"][0][0] = perf_n/1000/1000
            for spconf in js_conf["config"]['entity']:
                if spconf[0] == 'inv_cp':
                    spconf[2][1] = inv_n[2]
                elif spconf[0] == 'inv_wp':
                    spconf[2][1] = inv_n[0]
                elif spconf[0] == 'inv_hp':
                    spconf[2][1] = inv_n[1]
                elif spconf[0] == 'kernel_cp':
                    spconf[2][1] = filte_n[1]
                elif spconf[0] == 'kernel_kp':
                    spconf[2][1] = filte_n[0]
                elif spconf[0] == 'data_cp':
                    spconf[2][1] = trans_n[2]
                elif spconf[0] == 'data_wp':
                    spconf[2][1] = trans_n[0]
                elif spconf[0] == 'data_hp':
                    spconf[2][1] = trans_n[1]
                elif spconf[0] == 'bgemm_kp':
                    spconf[2][1] = matmul_n[1]
                elif spconf[0] == 'bgemm_wp':
                    spconf[2][1] = matmul_n[0]
                elif spconf[0] == 'bgemm_hp':
                    spconf[2][1] = matmul_n[2]
            work_group_conf.append(json.dumps(js_conf))
    with open('_from_rd_' + device + ".log",'w') as fp:
        for w in work_group_conf:
            fp.write(w+'\n')
