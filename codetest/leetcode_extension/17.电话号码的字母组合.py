#
# @lc app=leetcode.cn id=17 lang=python3
#
# [17] 电话号码的字母组合
#
from typing import List
# @lc code=start
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        num2char={"2":["a","b","c"],"3":["d","e","f"],"4":["g","h","i"],"5":["j","k","l"],
                    "6":["m","n","o"],"7":["p","q","r","s"],"8":["t","u","v"],"9":["w","x","y","z"]}
        old_save=[""]
        new_save=[]
        for digit in digits:
            chars=num2char[digit]
            for old_s in old_save:
                for char in chars:
                    new_save.append(old_s+char)
            old_save=new_save[:]
            new_save=[]
        if old_save==[""]:
            return []
        return old_save
# @lc code=end



