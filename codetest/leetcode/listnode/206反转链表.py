# Definition for singly-linked list.
from typing import List

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        
        def reversednode(prenode,node):
            if node==None:
                return prenode
            else:
                nextnode=node.next
                node.next=prenode
                return reversednode(node,nextnode)
        if not head or not head.next:
            return head
        secondnode=head.next
        head.next=None
        return reversednode(head,secondnode)

head=ListNode(0)
prenode=head
for i in range(1,5):
    node=ListNode(i)
    prenode.next=node
    prenode=node
So=Solution()
So.reverseList(head=head)


