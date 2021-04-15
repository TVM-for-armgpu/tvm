#0000028e2c780e4e g76
#AQH7N17B14007975 g72
#8A9Y0G80H pixel 3xl 630
#9A221FFBA005D8 pixel 4xl 640
device_id="0000028e2c780e4e"
device_port=9002
if [ x"$1" == x"G72" ];then
    device_id="AQH7N17B14007975"
    device_port=9001
elif [ x"$1" == x"G76" ]; then
    device_id="0000028e2c780e4e"
    device_port=9002
elif [ x"$1" == x"630" ]; then
    device_id="8A9Y0G80H"
    device_port=9003
elif [ x"$1" == x"640" ]; then
    device_id="9A221FFBA005D8"
    device_port=9000
else
    echo "please give a device keyi[ G72 G76 630 640]"
    exit 0
fi
echo "====================using $1 $device_id $device_port============="

function check_device() {
    adb devices|grep ${device_id}
    ret=$?
    return $ret
}
function wait_device() {
    #/platform-tools/adb tcpip 5555
    check_device
    ret=$?
    while [[ $ret -ne 0 ]]
    do
        echo -e " \rwait device online ..\c"
        sleep 1
        echo -e " \rwait device online ....\c"
        check_device
        ret=$?
    done
}
function wait_network() {
    ifconfig |grep enp0s26u1u2
    ret=$?
    while [[ $ret -ne 0 ]]
    do
        ret=$(check_device)
        if [[ $ret != 0 ]];then
            break
        fi
        echo -e "\rwait netswork ready..\c"
        sleep 1
        echo -e "\rwait netswork ready....\c"
        ifconfig |grep enp0s26u1u2
        ret=$?
    done
    return $ret
}
function wait_port_released() {
    adb -s ${device_id}  shell  "pkill -9 tvm_rpc"
    adb -s ${device_id}  shell  "netstat -an|grep 9000" |grep tcp
    ret=$?
    while [[ $ret == 0 ]];
    do
        echo -e "\rwait port 9000 available.."
        sleep 5
        echo -e "\rwait port 9000 available...."
        adb -s ${device_id}  shell  "netstat -an|grep 9000" |grep tcp
        ret=$?
    done
    echo "port 9000 is available now"
}

while true; do
    echo "start to connect to device ${device_id}......................"
    wait_device

    #adb -s ${device_id}  shell  "svc usb setFunctions rndis"

    #wait_network
                                            
    #if [ x"${device_id}" != x"9A221FFBA005D8" ];then
        adb -s ${device_id} forward tcp:${device_port} tcp:9000
    #fi
    adb -s ${device_id} reverse tcp:9090  tcp:9090
    #sudo ifconfig enp0s29u1u6 192.168.42.2
    wait_port_released
    echo "start rpc_server"
    if [ x"${device_id}" != x"9A221FFBA005D8" ];then
        adb -s ${device_id}  shell  "chmod +x /data/local/tmp/arm_runtime_jicheng/run.sh"
        adb -s ${device_id}  shell  "/data/local/tmp/arm_runtime_jicheng/run.sh"
    else
        adb -s ${device_id}  shell su -c  "chmod +x /data/local/tmp/arm_runtime_jicheng/run.sh"
        adb -s ${device_id}  shell su -c  "/data/local/tmp/arm_runtime_jicheng/run.sh"
    fi
        
    sleep 2
done
