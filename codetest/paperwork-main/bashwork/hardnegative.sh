cd /vc_data/users/v-mxiong/qc/QueryTitle/hardnegfold
cat hardnegative* > NegativeTrain
rm -rf hardnegative*
split -d -5636309 -a 2 NegativeTrain train
rm -rf NegativeTrain
i=0; for x in train0*; do mv $x train$i; let i=i+1; done
touch info
nodenum=$(ls -l train*|grep "^-" |wc -l)
echo -e "$nodenum\t1" > info

i=0; for x in train0*; do mv $x train$i; let i=i+1; done

for name in `ls -d */`; do cp ${name}NegativeVal ../JigsawMix/${name}NegativeVal2; done
for name in `ls -d */`; do cp ${name}NegativeTest ../JigsawMix/${name}NegativeTest2; done


for name in `ls -d */`; do mv ${name}NegativeVal ./${name}NegativeVal1; done
for name in `ls -d */`; do mv ${name}NegativeTest ./${name}NegativeTest1; done


for name in `ls -d */`; do cat ${name}NegativeVal1 ./${name}NegativeVal2 > ${name}NegativeVal; done
for name in `ls -d */`; do cat ${name}NegativeTest1 ./${name}NegativeTest2 > ${name}NegativeTest; done