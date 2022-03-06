# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head==None or head.next==None:
            return None
        #假设有个头结点，从头结点开始走了一步后
        slow_Node,fast_Node=head,head.next

        while(slow_Node!=fast_Node):
            if fast_Node.next and fast_Node.next.next:
                fast_Node=fast_Node.next.next
            else:
                return None
            slow_Node=slow_Node.next
        #pre_Node 从虚拟头结点走一步到head
        pre_Node,slow_Node=head,slow_Node.next
        while (pre_Node!=slow_Node):
            pre_Node=pre_Node.next
            slow_Node=slow_Node.next
        return pre_Node