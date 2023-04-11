#
# @lc app=leetcode.cn id=160 lang=python3
#
# [160] 相交链表
#
from asyncio import ReadTransport
from functions import ListNode
# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        
# @lc code=end
rearB=headB
while(rearB): rearB=rearB.next
rearB.next=headA 
#假设有个fake节点在headB前面
slow_node,fast_node=rearB,rearB.next
while(slow_node!=fast_node):
    if fast_node and fast_node.next:
        fast_node=fast_node.next.next
    else:
        print(None)
    slow_node=slow_node.next
extrance_node=rearB
slow_node=slow_node.next
while(extrance_node!=slow_node):
    extrance_node=extrance_node.next
    slow_node=slow_node.next
