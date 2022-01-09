from typing import Counter


def trap1(height):
    left,right=0,len(height)-1
    if left==right:
        return 0
    max_index,water_num,count=left,0,0
    while left<=right:
        if height[left]>height[max_index]:
            water_num+=(left-max_index)*height[max_index]-count
            count=height[left]
            max_index=left
        else:
            count+=height[left]
        left+=1
    start,count=right,0
    while right>=max_index:
        if height[right]>height[start]:
            water_num+=(start-right)*height[start]-count
            count=height[right]
            start=right
        else:
            count+=height[right]
        right-=1
    return water_num

def trap2(height):
    #单调栈解法
    stack=[]
    if len(height)==0:
        return 0
    i,water_num=0,0
    while i<len(height):
        if len(stack)==0:
            stack.append(i)
            i+=1
        elif height[i]>height[stack[-1]] and len(stack)==1:
            stack.pop()
            stack.append(i)
            i+=1
        elif height[i]>height[stack[-1]] and len(stack)>1:
            temp_index=stack.pop()
            water_num+=(min(height[i],height[stack[-1]])-height[temp_index])*(i-stack[-1]-1)
        else:
            stack.append(i)
            i+=1
    return water_num

height=eval(input("enter height list split by ,:"))
print(trap2(height))
    

    
