def largestRectangleArea(heights):
    stack=[]
    if len(heights)==0:
        return 0
    heights.append(-1)
    i,max_cnt=0,0
    while i<len(heights):
        if len(stack)==0 or heights[i]>=heights[stack[-1]]:
            stack.append(i)
            i+=1
        elif heights[i]<heights[stack[-1]]:
            mid_index=stack.pop()
            if len(stack)==0:
                left_index=-1
            else:
                left_index=stack[-1]
            cnt=(i-left_index-1)*heights[mid_index]
            if cnt>max_cnt:
                max_cnt=cnt
    return max_cnt
heights=eval(input("enter list of heights,split by ,"))
print(largestRectangleArea(heights))
