# rsync
rsync  -av --exclude ./test/logs --exclude logs/ qc/ paperwork/
sudo fuser -v /dev/nvidia* |awk '{for(i=1;i<=NF;i++)print "kill -9 " $i;}' | sudo sh
