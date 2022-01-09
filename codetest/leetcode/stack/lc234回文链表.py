# Definition for singly-linked list.


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def makeNode(Nodelist):
    head=ListNode(0)
    p_Node=head
    for i in range(len(Nodelist)):
        Node=ListNode(Nodelist[i])
        p_Node.next=Node
        p_Node=Node
    return head.next

def isPalindrome( head):
    length=0
    Node=head
    while(Node!=None):
        length+=1
        Node=Node.next
    if length==1:
        return True
    if length%2==0:
        left=length//2
        right=length//2+1
    else:
        left,right=length//2+1,length//2+1
    cnt,Node,p_Node,r_Node=1,head,None,None
    while(cnt<=left):
        cnt+=1
        r_Node=Node.next
        Node.next=p_Node
        p_Node=Node
        Node=r_Node
    if left==right:
        left,right=p_Node.next,Node
    else:
        left,right=p_Node,Node
    while left!=None and right!=None:
        if left.val!=right.val:
            return False
        else:
            left=left.next
            right=right.next
    return True

NodeList=eval(input("enter NodeList:"))
head=makeNode(NodeList)
print(isPalindrome(head))
