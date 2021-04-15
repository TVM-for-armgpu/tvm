sleep 6h
python tune_relay_mobile_gpu_by_x86schedule.py
sed -i "s/network_name = 'mobilenet'/#network_name = 'mobilenet'/" tune_relay_mobile_gpu_by_x86schedule.py
python tune_relay_mobile_gpu_by_x86schedule.py
