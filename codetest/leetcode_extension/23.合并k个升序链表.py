#
# @lc app=leetcode.cn id=23 lang=python3
#
# [23] 合并K个升序链表
#
from typing import List,Optional

# @lc code=start
# Definition for singly-linked list.
from collections import deque
import heapq
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        minHeap=[]
        for listi in lists:
            while listi:
                heapq.heappush(minHeap,listi.val)
                listi=listi.next
        fake_head=ListNode(-1)
        p=fake_head
        while minHeap:
            Node=ListNode(heapq.heappop(minHeap)) 
            p.next=Node
            p=p.next
            
        
        return fake_head.next

# @lc code=end
from classes import ListNode
from functions import create_linklist,show_linklist
import heapq
#解法1
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if len(lists)<1:
            return None
        def merge2List(head1,head2):
            fake_head=ListNode(-1)
            pre_node=fake_head
            while (head1 and head2):
                if  head1.val<head2.val:
                    pre_node.next=head1
                    pre_node=head1
                    head1=head1.next
                elif  head1.val>=head2.val:
                    pre_node.next=head2
                    pre_node=head2
                    head2=head2.next
            if head1:
                pre_node.next=head1
            elif head2:
                pre_node.next=head2

            return fake_head.next
        nodequeue=deque( lists)
        while(len(nodequeue)>1):
            head1=nodequeue.popleft()
            head2=nodequeue.popleft()
            head3=merge2List(head1,head2)
            nodequeue.append(head3)
        return nodequeue[0]

