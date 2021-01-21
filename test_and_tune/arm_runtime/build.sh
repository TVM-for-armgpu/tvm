if [ $# -eq 2 ] ; then
    cmake .. -DCMAKE_CXX_COMPILER=/mnt/andriod/android-toolchain-arm64/bin/aarch64-linux-android-g++  -DCMAKE_LIBRARY_PATH=/mnt/andriod//opencl-sdk-1.2.2/ -DOpenCL_LIBRARIES=/mnt/andriod/opencl-sdk-1.2.2/libOpenCL.so
fi
/mnt/andriod/android-toolchain-arm64/bin/aarch64-linux-android-gcc  tcp_test.c -o nc
make runtime tvm_rpc -j6
devicesn=$(adb devices|grep -v attached |wc -l)
if [ "$devicesn" == "1" ];then
   adb connect 127.0.0.1:2222
fi

adb reconnect offline
adb  push libtvm_runtime.so /data/local/tmp/arm_runtime_jicheng/lib/
adb  push tvm_rpc /data/local/tmp/arm_runtime_jicheng/
adb  push runrpc.sh  /data/local/tmp/run.sh
adb  push nc /data/local/tmp/nc
adb  push libOpenCL.so  /data/local/tmp/arm_runtime_jicheng/lib/
