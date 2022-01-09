words_list=[['We', 'have', 'previously', 'reported', 'cross', 'sectional', 'age', 'differences', 'and', '4', 'year', 
'longitudinal', 'age', 'changes', 'in', 'mean', 'cortical', 'thickness', 'within', 'eight', 
'sulcal', 'regions', 'in', 'a', 'subset', 'of', '35', 'older', 'adults', 'from', 'the', 'Baltimore',
 'Longitudinal', 'Study', 'of', 'Aging', 'BLSA']]
upp_cnt_label=0
stop_list=["of","and","for","in","s"]
for words in words_list:

    for i in range(len(words)):
        
        if i>0 and words[i].isupper() and len(words[i])>3 and words[i][-1]==words[i-1][0]:
            temp=i-2
            if temp>0 and words[temp] not in stop_list and words[i][-2]==words[temp][0]:
                    
                upp_cnt_label+=1
                break
            if temp-1>0 and words[temp] in stop_list and words[i][-2]==words[temp-1][0]:
                upp_cnt_label+=1
                break
print(upp_cnt_label)