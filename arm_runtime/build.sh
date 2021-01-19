if [ $# -eq 2 ] ; then
    cmake .. -DCMAKE_CXX_COMPILER=/home/jicwen/andriod/android-toolchain-arm64/bin/aarch64-linux-android-g++  -DCMAKE_LIBRARY_PATH=/home/jicwen/andriod//opencl-sdk-1.2.2/ -DOpenCL_LIBRARIES=/home/jicwen/andriod/opencl-sdk-1.2.2/libOpenCL.so
fi
/home/jicwen/andriod/android-toolchain-arm64/bin/aarch64-linux-android-gcc  tcp_test.c -o nc
make runtime tvm_rpc -j6

/mnt/d/Programs/platform-tools/adb.exe  push libtvm_runtime.so /data/local/tmp/arm_runtime_jicheng/lib/
/mnt/d/Programs/platform-tools/adb.exe  push tvm_rpc /data/local/tmp/arm_runtime_jicheng/
/mnt/d/Programs/platform-tools/adb.exe  push runrpc.sh  /data/local/tmp/run.sh
/mnt/d/Programs/platform-tools/adb.exe  push nc /data/local/tmp/nc
/mnt/d/Programs/platform-tools/adb.exe  push libOpenCL.so  /data/local/tmp/arm_runtime_jicheng/lib/
