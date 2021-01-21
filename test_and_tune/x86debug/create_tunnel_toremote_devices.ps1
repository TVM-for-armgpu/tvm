while ($true) {
    Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList  "ssh -Nf -R 20.78.32.118:9090:127.0.0.1:9090 azureuser@20.78.32.118"
    #Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList  "ssh -Nf -R 40.124.152.22:9090:127.0.0.1:9090 jicwen@40.124.152.22 -p 50000"
    Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList  "ssh -Nf -R 20.78.32.118:9000:192.168.42.129:9000 azureuser@20.78.32.118"
    #Start-Process -NoNewWindow -FilePath "powershell" -ArgumentList  "ssh -Nf -R 40.124.152.22:9000:192.168.42.129:9000 jicwen@40.124.152.22 -p 50000"
    ssh -Nf -R 20.78.32.118:2222:192.168.42.129:5555 azureuser@20.78.32.118
    #ssh -Nf -R 40.124.152.22:2222:192.168.42.129:5555 jicwen@40.124.152.22 -p 50000
    write-host "something wrong with this tunnel, sleep 3s and try again"
    Start-Sleep -Seconds 30
}
