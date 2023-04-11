from classes import ListNode,TreeNode
def create_linklist(nums):
    fake_head=ListNode(0)
    pre_head=fake_head
    for num in nums:
        Node=ListNode(num)
        pre_head.next=Node
        pre_head=Node
    return fake_head.next
def show_linklist(head):
    return_list=[]
    while(head):
        return_list.append(head.val)
        head=head.next
    return return_list

def makeTree(tree_list):
    if len(tree_list)==0:
        return None
    Node_list=[]
    for i,tree_val in enumerate(tree_list):
        if tree_val=="null":
            Node_list.append(None)
            continue
        val=int(tree_val)
        Node=TreeNode(val)
        Node_list.append(Node)
        if i%2==1 and (i-1)/2>=0:
            Node_list[(i-1)//2].left=Node
        elif i%2==0 and (i-2)/2>=0:
            Node_list[(i-2)//2].right=Node
    return Node_list[0]
def inorderTraversal(root):
    mid_visit=[]
    stack_tree=[]
    top=root
    while(top or len(stack_tree)!=0):
        while(top):
            stack_tree.append(top)
            top=top.left
        #栈中元素左left都已经访问，出栈防止死循环
        #仍在栈中代表未访问右节点
        top=stack_tree.pop()
        mid_visit.append(top.val)
        top=top.right
    return mid_visit




