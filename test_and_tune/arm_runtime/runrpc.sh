pkill -9 rvm_rpc
/data/local/tmp/nc 192.168.42.2 9090 1
if [ "$?" != "0" ];then
    exit 1
fi
/data/local/tmp/nc 9000 
if [ "$?" != "0" ];then
    exit 1
fi
echo "check server sucess"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/arm_runtime_jicheng/lib
cd /data/local/tmp/arm_runtime_jicheng
./tvm_rpc server --host=0.0.0.0 --port=9000 --port-end=9090  --tracker=192.168.42.2:9090 --key=android
