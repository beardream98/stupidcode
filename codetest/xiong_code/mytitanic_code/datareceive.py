import csv
import numpy as np
filename='train.csv'
train_list=[]
with open(filename) as f:
    reader=csv.reader(f)
    header_row=next(reader)

    for reader_column in reader:
        del reader_column[0]
        del reader_column[2]
        del reader_column[6]
        del reader_column[7]
        if reader_column[2]=='male':
            reader_column[2]=1
        elif reader_column[2]=='female':
            reader_column[2]=0
        # 先暂时不处理票号 船舱号 等信息
        if reader_column[7]=='Q':
            reader_column[7]=1
        elif reader_column[7]=='C':
            reader_column[7]=2
        elif reader_column[7]=='S':
            reader_column[7]=3
        else :
            reader_column[7]=0
        if reader_column[3]=="":
            reader_column[3]=45
        train_list.append(reader_column)
train_data=np.asarray(train_list)
train_data=train_data.astype(float)
x_train=train_data[:,1:]

y_train=train_data[:,0]

filename='test.csv'
test_list=[]
with open(filename) as f:
    reader=csv.reader(f)
    header_row=next(reader)
    
    for reader_column in reader:
        
        del reader_column[0]
        del reader_column[1]
        del reader_column[5]
        del reader_column[6]
        if reader_column[1]=='male':
            reader_column[1]=1
        elif reader_column[1]=='female':
            reader_column[1]=0
        # 先暂时不处理票号 船舱号 等信息
        if reader_column[6]=='Q':
            reader_column[6]=1
        elif reader_column[6]=='C':
            reader_column[6]=2
        elif reader_column[6]=='S':
            reader_column[6]=3
        else :
            reader_column[6]=0
        if reader_column[2]=="":
            reader_column[2]=45
        test_list.append(reader_column)

test_data=np.asarray(test_list)
test_data[152,5]=8
test_data=test_data.astype(float)
x_test=test_data

filename='gender_submission.csv'
y_test_list=[]
with open(filename) as f:
    reader=csv.reader(f)
    header_row=next(reader)
    
    for reader_column in reader:
        del reader_column[0]
        y_test_list.append(reader_column)
y_test=np.asarray(y_test_list)
y_test=y_test.flatten()
y_test=y_test.astype(float)

np.save('x_train.npy',x_train)
np.save('x_test.npy',x_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)

