#!/bin/bash

#username
USRNAME=hong

# log tag
TAG=ssd-w

#directories
CEPH_DIR=/home/${USRNAME}/ceph
JOBDIR=${CEPH_DIR}/fio_test
WORKDIR=${CEPH_DIR}/build/
LOGDIR=${WORKDIR}/fio_log
# MNTDIR=${WORKDIR}/mnt_user # hdd
MNTDIR=/mnt/ceph_ssd # ssd

#files
VSTARTFILE=${CEPH_DIR}/src/vstart.sh
OPTIONFILE=${CEPH_DIR}/src/common/options.cc
JOBFILE=${JOBDIR}/second.job
#JOBFILE=${JOBDIR}/ceph-fio-seq-read-4k-1-30s.job

###############################################################
# running fio with job files or fio files
LAST_SHD=1

for SHD_NUM in 1 #2 3 4 5 8
do
LAST_BLKSIZE=4

# # hdd, ssd
# sed -i "2828,2835s/.set_default(${LAST_SHD})/.set_default(${SHD_NUM})/g" ${OPTIONFILE}

# # show line contents of option.cc
# echo ">> sed -n 2835p ${OPTIONFILE}"
# sed -n 2835p ${OPTIONFILE}
# sleep 4

for BLK_SIZE in 4 8 16 32 64 128 256 512 #32  # 16 
do

sed -i "5,6s/bs=${LAST_BLKSIZE}K/bs=${BLK_SIZE}K/g" ${JOBFILE}
# show line contents of vstart.sh
echo ">> sed -n 5p ${JOBFILE}"
sed -n 5p ${JOBFILE}
sleep 2

# 5 iteration
for ITER in 1 2 3 4 5
do

OUT_LOG=${TAG}_bs${BLK_SIZE}_r${ITER}

# # run vstart.sh
# cd ${WORKDIR}

# make -j30
# VSTART_DEST=~/ceph/build ../src/vstart.sh -l -n -d
# #echo "\n./bin/ceph fs subvolume create a subvolume"
# ./bin/ceph fs subvolume create a subvolume --size $((100*1024*1024*1024))
# SUBVOL=`./bin/ceph fs subvolume getpath a subvolume`
# echo "SUBVOL: $SUBVOL"

# # Kernel mount
# monip=$(grep 'mon host' ceph.conf | grep -o '[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}' | uniq | head -n 1)
# port=`grep 'mon host' ceph.conf | awk -F":" '{print $3}' | awk -F"," '{print $1}'`
# key=`grep -A 1 admin keyring | grep key | awk -F "=" '{print $2}' | sed -e 's/^[[:space:]]*//'`
# echo "sudo mount -t ceph ${monip}:${port},${monip}:$((port+1)):$SUBVOL ${MNTDIR} -o name=admin,secret=${key}=="
# sudo mount -t ceph ${monip}:${port},${monip}:$((port+1)):$SUBVOL ${MNTDIR} -o name=admin,secret=${key}==

# #run fio
cd ${MNTDIR}
echo ">> fio ${JOBFILE} --write_bw_log=${OUT_LOG}"
sudo fio ${JOBFILE} --write_bw_log=${OUT_LOG}

# fio_log -> ceph/build/fio_log
sudo mv *.log ${LOGDIR}

echo "rm -rf fio-seq-write"
sudo rm -rf fio-seq-write
sleep 3
# # run stop.sh
# cd ${WORKDIR}
# echo "umount kernel..."
# sudo umount ${MNTDIR}
# echo "Complete...!"
# sh ../src/stop.sh
done

LAST_BLKSIZE=`expr $BLK_SIZE`
done
#set the file contents as default(=1)
sed -i "5,6s/bs=${LAST_BLKSIZE}K/bs=4K/g" ${JOBFILE}
LAST_SHD=`expr $SHD_NUM`
done
#set the file contents as default(1)
sed -i "2828,2835s/.set_default(${LAST_SHD})/.set_default(1)/g" ${OPTIONFILE}
