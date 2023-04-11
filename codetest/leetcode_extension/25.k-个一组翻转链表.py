#
# @lc app=leetcode.cn id=25 lang=python3
#
# [25] K 个一组翻转链表
#

# @lc code=start
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
from typing import Optional


class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverseList(prehead,rear):
            #进入循环所有从prehead到rear都不为None
            if not rear:
                return 
            nexthead=rear.next
            newrear=prehead.next

            p,q,r=prehead,prehead.next,prehead.next.next
            while(q!=nexthead):
                q.next=p
                # r为None情况 即当nexthead为None时
                if r:
                    p,q,r=q,r,r.next
                else:
                    break

            newrear.next=nexthead
            prehead.next=rear
            return newrear
        fake_head=ListNode(-1)
        fake_head.next=head
        prehead,rear=fake_head,fake_head
        if k==1:
            return head
        while(rear!=None):
            for i in range(k):
                if rear.next:
                    rear=rear.next
                else:
                    rear=None
                    break
            rear=reverseList(prehead,rear)
            prehead=rear
        return fake_head.next        
# @lc code=end

from functions import create_linklist,show_linklist
from classes import ListNode

with open("testwrite","r") as file_object:
    nums=eval(file_object.readline().strip())
    k=eval(file_object.readline().strip())
So=Solution()
head=create_linklist(nums)
head=So.reverseKGroup(head,k)
print(show_linklist(head))