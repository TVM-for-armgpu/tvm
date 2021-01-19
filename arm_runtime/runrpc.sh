export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/arm_runtime_jicheng/lib
cd /data/local/tmp/arm_runtime_jicheng
chmod +x ./tvm_rpc
./tvm_rpc server --host=0.0.0.0 --port=9000 --port-end=9090  --tracker=192.168.42.2:9090 --key=android
