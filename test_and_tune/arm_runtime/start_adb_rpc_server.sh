function check_device() {
    /mnt/d/Programs/platform-tools/adb.exe devices|grep 9A221FFBA005D8
    ret=$?
    return $ret
}
function wait_device() {
	#/mnt/d/Programs/platform-tools/adb.exe tcpip 5555
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
	/mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "pkill -9 tvm_rpc"
	/mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "netstat -an|grep 9000" |grep tcp
	ret=$?
	while [[ $ret == 0 ]];
	do
		echo -e "\rwait port 9000 available.."
		sleep 5
		echo -e "\rwait port 9000 available...."
		/mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "netstat -an|grep 9000" |grep tcp
		ret=$?
	done
	echo "port 9000 is available now"
}

while true; do
    echo "start to connect to device 9A221FFBA005D8......................"
    wait_device

    /mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "svc usb setFunctions rndis"

	#wait_network
	#ret=$?
    #if [[ $ret != 0 ]];then
    #    continue
    #fi
    #echo "set device ip as 192.168.42.2"
    #sudo ifconfig enp0s26u1u2 192.168.42.2
	wait_port_released
    echo "start rpc_server"
    /mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "chmod +x /data/local/tmp/run.sh"
    /mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "/data/local/tmp/run.sh"
    sleep 2
done
