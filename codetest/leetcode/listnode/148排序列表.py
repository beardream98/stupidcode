# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
from typing import List,Optional
class Solution:
    def merge2list(self,head1,head2):
        #head1 和head2 均不为空
        #合理初始化 伪头结点
        head3=ListNode(-1)
        rear3=head3

        while(head1 or head2):
            if head2==None or (head1!=None and head1.val<head2.val):
                temp_node=head1.next
                head1.next=None
                rear3.next=head1
                rear3=head1
                head1=temp_node
            elif head1==None or (head2!=None and head1.val>=head2.val):
                temp_node=head2.next
                head2.next=None
                rear3.next=head2
                rear3=head2
                head2=temp_node

        return head3.next,rear3
    def readListNode(self,head):
        nums=[]
        while(head):
            nums.append(head.val)
            head=head.next
        return nums
    def merge_listnode(self,head,m_num):
            f_head=ListNode(-1)
            f_head.next=head
            
            rear0=f_head
            head1=rear0.next
            while(head1):
                head2=head1
                for i in range(m_num):
                    if i==m_num-1 and head2:
                        temp_node=head2.next
                        head2.next=None
                        head2=temp_node
                    elif head2 and head2.next!=None:head2=head2.next
                if head2==None:
                    #第二条链为空不用排序了
                    return f_head.next

                head3=head2 
                for i in range(m_num):
                    if i==m_num-1 and head3:
                        temp_node=head3.next
                        head3.next=None
                        head3=temp_node
                    elif head3 and head3.next!=None:head3=head3.next

                #合并两个有序链表
                head1,rear2=self.merge2list(head1,head2)
                rear0.next=head1
                rear2.next=head3
                rear0=rear2
                head1=head3
            return f_head.next
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        #归并方法自底向上排序
        n=len(self.readListNode(head))
        cnt=1
        while(n!=1):
            head=self.merge_listnode(head,cnt)
            print(self.readListNode(head))
            cnt*=2
            if n%2==0:
                n=n//2
            else:
                n=n//2+1
        return head
        
def createlistNode(nums):
    f_head=ListNode(-1)
    pre_node=f_head
    for num in nums:
        Node=ListNode(num)
        pre_node.next=Node
        pre_node=Node
    return f_head.next


nums=[-1,5,3,4,0]
head=createlistNode(nums)

So=Solution()
So.sortList(head)




