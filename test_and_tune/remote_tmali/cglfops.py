import json

with open('/home/stcadmin/work/tvm/test_and_tune/remote_tmali/Adreno630.topi_nchw4c.log') as fp:
    for i,f in enumerate(fp):
        sp = json.loads(f)
        if 'input' not in sp:
            continue
        a=sp['input'][2] 
        res=sp["result"][0][0]
        gflops=a[1]*a[2]*a[3]*a[4]*2/res/1e9
        print(i,gflops)