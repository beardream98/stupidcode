#
# @lc app=leetcode.cn id=2 lang=python3
#
# [2] 两数相加
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        fake_head=ListNode(-1)
        head1,head2=l1,l2
        pre_node=fake_head
        extra_add=0
        while(head1 and head2):
            sum_two=head1.val+head2.val+extra_add
            extra_add=sum_two//10
            sum_two=sum_two%10
            Node=ListNode(sum_two)
            pre_node.next=Node
            pre_node=Node
            head1,head2=head1.next,head2.next
        while(head1):
            sum_two=head1.val+extra_add
            extra_add=sum_two//10
            sum_two=sum_two%10
            Node=ListNode(sum_two)
            pre_node.next=Node
            pre_node=Node
            head1=head1.next
        while(head2):
            sum_two=head2.val+extra_add
            extra_add=sum_two//10
            sum_two=sum_two%10
            Node=ListNode(sum_two)
            pre_node.next=Node
            pre_node=Node
            head2=head2.next
        if extra_add!=0:
            Node=ListNode(extra_add)
            pre_node.next=Node
        return fake_head.next
# @lc code=end
from functions import create_linklist,show_linklist
from classes import ListNode
from typer import open_file
with open_file("testwrite","r") as file_object:
    nums1=eval(file_object.readline().strip())
    nums2=eval(file_object.readline().strip())
head1=create_linklist(nums1)
head2=create_linklist(nums2)
fake_head=ListNode(-1)
pre_node=fake_head
extra_add=0
while(head1 and head2):
    sum_two=head1.val+head2.val+extra_add
    extra_add=sum_two//10
    sum_two=sum_two%10
    Node=ListNode(sum_two)
    pre_node.next=Node
    pre_node=Node
    head1,head2=head1.next,head2.next
while(head1):
    sum_two=head1.val+extra_add
    extra_add=sum_two//10
    sum_two=sum_two%10
    Node=ListNode(sum_two)
    pre_node.next=Node
    pre_node=Node
    head1=head1.next
while(head2):
    sum_two=head2.val+extra_add
    extra_add=sum_two//10
    sum_two=sum_two%10
    Node=ListNode(sum_two)
    pre_node.next=Node
    pre_node=Node
    head2=head2.next
if extra_add!=0:
    Node=ListNode(extra_add)
    pre_node.next=Node
print(show_linklist(fake_head.next))
# So=Solution()
