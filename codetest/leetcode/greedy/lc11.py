def maxArea( height):
    n=len(height)
    i,j,max_area=0,n-1,0
    while(i<j):
        if height[i]>=height[j]:
            area=height[j]*(j-i)
            j-=1
        else:
            area=height[i]*(j-i)
            i+=1
        if area>max_area:
            max_area=area
    return max_area

height=eval(input("please enter a list :"))
print(maxArea(height))