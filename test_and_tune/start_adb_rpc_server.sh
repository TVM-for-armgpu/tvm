function log() {
    time_now=`date +%F-%T`
    echo "$time_now  $1" 
}

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
	/mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "pkill -9 tvm_rpc"
	/mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "netstat -an|grep 9000" |grep tcp
	ret=$?
	while [[ $ret == 0 ]];
	do
		log "\rwait port 9000 available.."
		sleep 5
		/mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "netstat -an|grep 9000" |grep tcp
		ret=$?
	done
	log "port 9000 is available now"
}

while true; do
    log "start to connect to device 9A221FFBA005D8......................"
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
    log "start rpc_server"
    /mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "chmod +x /data/local/tmp/run.sh"
    /mnt/d/Programs/platform-tools/adb.exe -s 9A221FFBA005D8  shell su -c "/data/local/tmp/run.sh" >> log.txt
    sleep 2
done
