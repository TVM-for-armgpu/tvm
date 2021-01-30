import numpy as np
import tvm
import tvm.topi
from tvm import topi
import os
from tvm import te
## data shape
data_shape = (1, 3, 224, 224)
w_shape = (64,3,3,3)

## Data
sample_data = np.random.uniform(-1,1, size=data_shape ).astype("float32")
sample_p1 = np.random.uniform(-1,1, size=w_shape ).astype("float32")

## placeholder
input_data = tvm.te.placeholder( shape = data_shape, dtype = "float32", name="Input" )
p1 = tvm.te.placeholder( shape = w_shape, dtype="float32", name="p1" )

## Winograd conv2d
target="cuda"
with tvm.target.Target(target):
    conv = topi.cuda.conv2d_nchw_winograd(input_data
                                          ,p1 
                                          ,(1,1)
                                          ,(0,0)
                                          ,(1,1)
                                          ,"float32"  )
    print(conv.shape)
    sch = topi.cuda.schedule_conv2d_nchw_winograd([conv])
    winoMod = tvm.build( sch, [ input_data,p1,conv] , name='wino')

## Direct conv2d
with tvm.target.Target('cuda'):
    conv = topi.cuda.conv2d_nchw( input_data
                                    ,p1 
                                    ,[1,1]
                                    ,[0,0]
                                    ,[1,1] )
    sch = topi.cuda.schedule_conv2d_nchw([conv])
    simpleMod = tvm.build(sch, [input_data,p1], target, name='direct' )


## Real data
#ctx = tvm.context(target, 0)
ctx = tvm.gpu(0)
tvm_input = tvm.nd.array( sample_data , ctx )
tvm_p1 = tvm.nd.array( sample_p1, ctx )
oshape = conv.shape
oshape = tuple(int(i) for i in oshape)
print('oshape',oshape)
outnp = tvm.nd.array( np.random.uniform(-1,1, size=oshape).astype("float32"), ctx )

## Performance Testing
ev_wino = winoMod.time_evaluator(winoMod.entry_name, ctx, number=1,repeat=10)
ev_conv = simpleMod.time_evaluator(simpleMod.entry_name, ctx, number=1,repeat=100 )

timer = ev_conv( tvm_input, tvm_p1).mean*1e3
print("Conv with Direct algo -> ",timer)
timer = ev_wino( tvm_input, tvm_p1,outnp).mean*1e3
print("\n\n\nConv with Winograd Strassen algo -> ",timer )
