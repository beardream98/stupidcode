cd ~/autodl-tmp/epsave/data/binary/Twetter

# cat ../QueryTitleAll/NegativeTrain ../QueryTitle0*/NegativeTrain > NegativeTrain
# cat ../QueryTitleAll/NegativeTest ../QueryTitle0*/NegativeTest > NegativeTest
# cat ../QueryTitleAll/PositiveTest ../QueryTitle0*/PositiveTest > PositiveTest
# cat ../QueryTitleAll/PositiveTrain ../QueryTitle0*/PositiveTrain > PositiveTrain
mkdir positivefold
mkdir negativefold
let PositiveTrainNum=$(wc -l PositiveTrain )
let PositiveTestNum=$(wc -l PositiveTest )
cd negativefold
split -d -$PositiveTrainNum -a 3 ../NegativeTrain train
split -d -$PositiveTestNum -a 3 ../NegativeTest test
i=0; for x in train0*; do mv $x train$i; let i=i+1; done
i=0; for x in test0*; do mv $x test$i; let i=i+1; done

trainfold=$(ls -l train*|grep "^-" |wc -l)
testfold=$(ls -l test*|grep "^-" |wc -l)
cd ../
touch ./negativefold/info
echo -e "$trainfold\t$testfold" > ./negativefold/info
cp -r PositiveTrain ./positivefold/train
cp -r PositiveTest ./positivefold/test




