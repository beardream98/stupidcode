
from inspect import stack
import queue


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def inorderTraversal(root):
    mid_visit=[]
    stack_tree=[]
    top=root
    while(top or len(stack_tree)!=0):
        while(top):
            stack_tree.append(top)
            top=top.left
        top=stack_tree.pop()
        mid_visit.append(top.val)
        top=top.right
    return mid_visit
def buildTree1(preorder, inorder):
    n=len(preorder)
    def bulidNode(preindex,inorder_s):
        val,index=0,0
        if len(inorder_s)==0:
            return None
        for i in range(preindex,n):
            if preorder[i] in inorder_s:
                index=inorder_s.index(preorder[i])
                val=preorder[i]
                break
        Node=TreeNode(val=val)
        left=bulidNode(preindex+1,inorder_s[0:index])
        right=bulidNode(preindex+1,inorder_s[index+1:len(inorder_s)])
        Node.left,Node.right=left,right
        return Node
    root=bulidNode(0,inorder)
    return root
def buildTree2(preorder, inorder):
    #迭代方法 利用preorder 性质，右边的第一个要么是左儿子 要么在右边
    index_pre={ element:i for i,element in enumerate(inorder)}
    stack_Node=[]
    i=0
    root,index=TreeNode(val=preorder[i]),index_pre[preorder[i]]
    i+=1
    stack_Node.append({"Node":root,"l":0,"r":len(preorder)-1,"index":index})
    while(stack_Node):
        top=stack_Node[-1]
        # 当左子树不存在且有左子树位置时
        if top["Node"].left==None and top["index"]>top["l"]:
            top["Node"].left=TreeNode(val=preorder[i])
            l,r,index=top["l"],top["index"]-1,index_pre[preorder[i]]
            i+=1
            stack_Node.append({"Node":top["Node"].left,"l":l,"r":r,"index":index})
            continue
        #当右子树有位置时退栈并进栈否则退栈
        if top["Node"].right==None and top["index"]<top["r"]:
            top["Node"].right=TreeNode(val=preorder[i])
            l,r,index=top["index"]+1,top["r"],index_pre[preorder[i]]
            i+=1
            stack_Node.pop()
            stack_Node.append({"Node":top["Node"].right,"l":l,"r":r,"index":index})
        else:
            stack_Node.pop()
    return root

        
        



preorder=eval(input("please enter preorder of a tree:"))
inorder=eval(input("please enter inorder of a tree:"))
root=buildTree2(preorder,inorder)
print(inorderTraversal(root))

