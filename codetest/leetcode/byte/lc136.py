def singleNumber(self, nums: List[int]) -> int:
#异或运算符能保证 a+a=0
    ans = nums[0]
    for i in range(1, len(nums)):
        ans = ans ^ nums[i]
    return ans