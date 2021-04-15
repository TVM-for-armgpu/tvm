function log() {
    time_now=`date +%F-%T`
    echo "$time_now  $1" 
}

function check_device() {
    adb devices|grep 9A221FFBA005D8
    ret=$?
    return $ret
}
function wait_device() {
	#adb tcpip 5555
    check_device
    ret=$?
    while [[ $ret -ne 0 ]]
    do
        log "\rwait device online ..\c"
        sleep 1
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
        log "\rwait netswork ready..\c"
        sleep 1
        ifconfig |grep enp0s26u1u2
        ret=$?
    done
	return $ret
}
function wait_port_released() {
	adb -s 9A221FFBA005D8  shell su -c "pkill -9 tvm_rpc"
	adb -s 9A221FFBA005D8  shell su -c "netstat -an|grep 9000" |grep tcp
	ret=$?
	while [[ $ret == 0 ]];
	do
		log "\rwait port 9000 available.."
		sleep 5
		adb -s 9A221FFBA005D8  shell su -c "netstat -an|grep 9000" |grep tcp
		ret=$?
	done
	log "port 9000 is available now"
}
device_id=9A221FFBA005D8
device_port=9000
while true; do
    log "start to connect to device 9A221FFBA005D8......................"
    wait_device

    #adb -s 9A221FFBA005D8  shell su -c "svc usb setFunctions rndis"
    adb -s ${device_id} forward tcp:${device_port} tcp:9000
    adb -s ${device_id} reverse tcp:9090  tcp:9090

	#wait_network
	#ret=$?
    #if [[ $ret != 0 ]];then
    #    continue
    #fi
    #echo "set device ip as 192.168.42.2"
    #sudo ifconfig enp0s26u1u2 192.168.42.2
	wait_port_released
    log "start rpc_server"
    adb -s 9A221FFBA005D8  shell su -c "chmod +x /data/local/tmp/arm_runtime_jicheng/run.sh"
    adb -s 9A221FFBA005D8  shell su -c "/data/local/tmp/arm_runtime_jicheng/run.sh"
    sleep 2
done
