#
# @lc app=leetcode.cn id=143 lang=python3
#
# [143] 重排链表
#
from functions import ListNode

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        
        slow,fast=head,head
        while(1):
            if fast.next and fast.next.next:
                fast=fast.next.next
                slow=slow.next
            else:
                break
        head2=slow.next
        if not head2:
            # 只有单个元素
            return head
        slow.next=None
        p,q=head2,head2.next
        while(q):
            r=q.next
            q.next=p
            p,q=q,r
        head2.next=None
        q=head
        while(p):
            pn=p.next
            qn=q.next

            q.next=p
            p.next=qn
            p,q=pn,qn
        return head 
# @lc code=end
from functions import show_linklist,create_linklist
# nums=[1,2,3,4,5]
# head=create_linklist(nums)


    



