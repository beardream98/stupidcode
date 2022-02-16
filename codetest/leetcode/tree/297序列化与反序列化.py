# Definition for a binary tree node.
from pydoc import describe


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if root==None:
            return []
        layer_stack=[root]
        data=[]
        flag=True
        while(flag):
            n=len(layer_stack)
            for i in range(n):
                node=layer_stack.pop(0)
                if node==None:
                    data.append("null")
                    layer_stack.append(None)
                    layer_stack.append(None)

                else:
                    data.append(node.val)
                    layer_stack.append(node.left)
                    layer_stack.append(node.right)
                    
            if all(ele is None for ele in layer_stack):
                flag=False
        return data
                

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        if len(data)==0:
            return None
        Node_list=[]
        for i,tree_val in enumerate(data):
            if tree_val=="null":
                Node_list.append(None)
                continue
            val=int(tree_val)
            Node=TreeNode(x=val)
            Node_list.append(Node)
            if i%2==1 and (i-1)/2>=0:
                Node_list[(i-1)//2].left=Node
            elif i%2==0 and (i-2)/2>=0:
                Node_list[(i-2)//2].right=Node
        return Node_list[0]
root=eval(input("please enter a list:"))
ser = Codec()
deser = Codec()
root=deser.deserialize(root)
ans = deser.deserialize(ser.serialize(root))
print(ans)
# [1,2,3,"null","null",4,5]