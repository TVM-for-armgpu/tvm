nohup python topi_nhcw4c.py MaliG72 > g72.log 2>&1&
nohup python topi_nhcw4c.py Adreno630 > 630.log 2>&1&
nohup python topi_nhcw4c.py Adreno640 > 640.log 2>&1&
nohup python topi_nhcw4c.py MaliG76 > g76.log 2>&1&

nohup python topi_spatial.py MaliG72 > g72.log 2>&1&
nohup python topi_spatial.py Adreno630 > 630.log 2>&1&
nohup python topi_spatial.py Adreno640 > 640.log 2>&1&
nohup python topi_spatial.py MaliG76 > g76.log 2>&1&



nohup python topi_winograd_conv3x3.py Adreno640 > 640.log 2>&1&
nohup python topi_winograd_conv3x3.py Adreno630 > 630.log 2>&1&
nohup python topi_winograd_conv3x3.py MaliG76 > g76.log 2>&1&


nohup python tunerelaymobilegpu_nhcw4c_batch.py Adreno640 > 640.log 2>&1&
nohup python tunerelaymobilegpu_nhcw4c_batch.py Adreno630 > 630.log 2>&1&
nohup python tunerelaymobilegpu_nhcw4c_batch.py MaliG76 > g76.log 2>&1&
nohup python topi_nhcw4c_mace.py MaliG76 > g76.log 2>&1&


nohup python topi_depthwise_nchw4c.py Adreno640 > 640.log 2>&1&
nohup python topi_depthwise_nchw4c.py Adreno630 > 630.log 2>&1&
nohup python topi_depthwise_nchw4c.py MaliG76 > g76.log 2>&1&

