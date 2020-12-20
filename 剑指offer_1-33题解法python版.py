剑指offer刷题:

1.二维数组的查找:在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        for i in range(len(array)):
            for j in  range(len(array[0])):
                if array[i][j] == target:
                    return True
        return False



2.替换空格:请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
# -*- coding:utf-8 -*-
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        # write code here
        return str(s).replace(" ","%20")



3.从头到尾打印链表:输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
方法一：利用字符串，列表，元组的切片原理
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        res = []
        while listNode:
            res.append(listNode.val)
            listNode = listNode.next
        return res[::-1]

方法二：利用列表的内置函数append，insert
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        array = []
        if listNode:
            array.append(listNode.val)
            p = listNode.next
            while p:
                array.insert(0,p.val)
                p =p.next
        return array

方法三：利用列表中的reverse函数
暂时有问题，代码没有跑通



4.重建二叉树:输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
解题思路：
首先找到中序遍历中左子树的所有结点llen，再找到中序遍历中右子树的所有结点rlen，
然后用递归方法，
root.left =（先序遍历左子树的结点，中序遍历左子树的结点）
root.right = (先序遍历右子树的结点，中序遍历右子树的结点)

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        if len(pre) == 1:
            return TreeNode(pre[0])
        root = TreeNode(pre[0])
        llen = tin[ :tin.index(pre[0])]
        rlen = tin[tin.index(pre[0])+1: ]
        root.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1],llen)
        root.right = self.reConstructBinaryTree(pre[tin.index(pre[0])+1:],rlen)
        return root



5.用两个栈实现队列:用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
解题思路：栈是先进后出，队列是先进先出，利用Python列表的append和pop属性可以模拟队列的运行
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
    def push(self, node):
        # write code here
        self.stack1.append(node)
    def pop(self):
        # return xx
        if self.stack2 == []:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

注意：此题面字节跳动后端开发面试过


6.旋转数组的最小数字:把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
解法一：找到比列表第一个元素还小的数即为最小值
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        #此题有两种做法
        #第一种
        if len(rotateArray) == 0:
            return None
        j = rotateArray[0]
        for i in range(1,len(rotateArray)):
            if j > rotateArray[i]:
                return rotateArray[i]
解法二：遍历列表，如果发现前边的数大于后面的数，则后面的数即为要查找的最小数。
# -*- coding:utf-8 -*-
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        # write code here
        if len(rotateArray) == 0 :
            return 0
        for i in range(len(rotateArray)):
            j =i +1
            if j < len(rotateArray) and rotateArray[i] > rotateArray[j]:
                return rotateArray[j]


7.斐波那契数列:大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）,n<=39
解题思路：找规律题，先找出规律，然后归纳总结
# -*- coding:utf-8 -*-
class Solution:
    def Fibonacci(self, n):
        # write code here
        #if n == 0:
         #   return 0
        #res = [0,1]
        #for i in range(2,n+1):
         #   res.append(res[i-2]+res[i-1])
        #return res[-1]
        res = [0,1]
        while len(res) < n+1:
            res.append(res[-1] + res[-2])
        return res[n]



8.跳台阶:一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
解题思路：找规律题，先找出规律，然后归纳总结
解法一:
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        res = [0,1,2]
        if number <= 2:
            return res[number]
        while len(res) <= number:
            res.apppend(res[-1]+res[-2])
        return res[-1]

解法二:使用for循环
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        # write code here
        res = [0,1,2]
        if number <= 2:
            return res[number]
        for i in range(3,number+1):
            res.append(res[-1] + res[-2])
        return res[number]



9.变态跳台阶:一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。
解题思路：利用归纳总结法，找规律，第一个规律是第n个数是前边所有数的和加1，第二个规律是第n个数是2的n-1次方
解法一:
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number == 1:
            return 1
        count =2
        i = 2
        while i < number:
            count = count * 2
            i +=1
        return count

解法二:
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        if number == 1 :
            return 1
        count = 1
        for i in range(1,number):
            count = count * 2
        return count

解法三：
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloorII(self, number):
        # write code here
        res = [0,1]
        while len(res) < number+1:
            res.append(sum(res)+1)
        return res[number]


10.矩形覆盖:我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？
解题思路：找规律题，先找出规律，然后归纳总结。此题规律是第n个数等于前两个数之和
解法一:
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        # write code here
        if number <= 1:
            return number
        res = [1,2]
        for i in range(2,number+1):
            res.append(res[i-1]+res[i-2])
        return res[number-1]

解法二:
# -*- coding:utf-8 -*-
class Solution:
    def rectCover(self, number):
        res= [0,1,2]
        if number <= 2:
            return res[number]
        while len(res) <= number:
            res.append(res[-1] + res[-2])
        return res[number]



11.二进制中1的个数:输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。
class Solution:
    def NumberOf1(self, n):
        # write code here
        count=0
        if n < 0:
            n = n & 0xffffffff
        while n!=0:
            n=n&(n-1)
            count+=1
        return count



12.数值的整数次方:给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方,保证base和exponent不同时为0
解题思路：指数exponent分为大于0和小于0，如果大于0，则求得的结果即为最终的结果，如果小于0，则最后还要加上相反数
解法一:
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base ==0:
            return 0
        if exponent == 0:
            return 1
        if exponent > 0:
            a = 1
            for i in range(1,exponent+1):
                a =a * base
            return a
        elif exponent < 0:
            a = 1
            for i in range(1,-exponent+1):
                a =a * base
            return 1/a

解法二:
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0:
            return 0
        if exponent >0:
            a = 1
            for i in range(1,exponent+1):
                a = a * base
            return a
        elif exponent < 0:
            a = 1 
            for i in range(-1,exponent-1,-1):
                a = a * base
            return 1/a
        else:
            return 1

解法三：
# -*- coding:utf-8 -*-
class Solution:
    def Power(self, base, exponent):
        # write code here
        if base == 0:
            return 0
        elif exponent == 0:
            return 1
        elif exponent > 0:
            count = 1
            value = base
            while count < exponent:
                value = value * base
                count += 1
            return value
        else:
            count = -1
            value = base
            while count> exponent:
                value = value * base
                count += -1
            return 1/value


13.调整数组顺序使奇数位于偶数前边:输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，
偶数和偶数之间的相对位置不变。输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
解法一:
# -*- coding:utf-8 -*-
class Solution:
    def reOrderArray(self, array):
        # write code here
        n = len(array)
        i =0
        while i < n:
            if array[i] % 2 ==1:
                i+=1
            else:
                array.append(array[i])
                del(array[i])
                n -=1
        return array
注:此题不能用for循环,关键点在于每次找到偶数后进行删除,然后指针i不变



14.链表中倒数第K个结点:输入一个链表，输出该链表中倒数第k个结点。
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindKthToTail(self, head, k):
        # write code here
        p = head
        q = head
        if not head:
            return None
        count = 0 
        while p:
            if count == k:
                p = p.next
                q = q.next
            else:
                p = p.next
                count += 1
        if count < k:
            return None
        return q



15.反转链表:输入一个链表，反转链表后，输出新链表的表头。

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if not pHead or not pHead.next:
            return pHead
        p = pHead.next
        pHead.next = None
        while p:
            r = p.next
            p.next = pHead
            pHead = p
            p = r
        return pHead



16.合并两个排序的链表:输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

解法一:不知道为什么这种方法不行
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        head = pHead1
        if not pHead1 or not pHead2:
            return pHead1 or pHead2
        while pHead1 and pHead2:
            if pHead2.val < pHead1.val:
                r = pHead2.next
                pHead2.next = pHead1
                pHead1 = pHead2
            if pHead2.val > pHead1.val and pHead2.val < pHead1.next.val:
                r = pHead2.next
                pHead2.next = pHead1.next
                pHead1.next = pHead2.next
                pHead1 = pHead1.next
                pHead2 = r
            else:
                pHead1 = pHead1.next
        if pHead2:
            pHead1.next = pHead2
        return head

解法二:
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        dummy = ListNode(0)
        phead = dummy
        while pHead1 and pHead2:
            if pHead1.val > pHead2.val:
                r = pHead2.next
                dummy.next = pHead2
                pHead2 = r
            else:
                r = pHead1.next
                dummy.next = pHead1
                pHead1 = r
            dummy = dummy.next
        if pHead1:
            dummy.next = pHead1
        if pHead2:
            dummy.next = pHead2
        return phead.next

解法三：
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        if not pHead1 or not pHead2:
            return pHead1 or pHead2
        if pHead1.val > pHead2.val:
            pHead = pHead2
            p = pHead1
            q = pHead2.next
            a =  pHead
        else:
            pHead = pHead1
            p = pHead1.next
            q = pHead2
            a = pHead
        while p and q:
            if p.val > q.val:
                r = q.next
                a.next = q
                a =q
                q =r
            else:
                r = p.next
                a.next = p
                a = p
                p = r
        if p:
            a.next = p 
        else:
            a.next = q
        return pHead

17.输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）
解题思路：分为三种情况，
如果根节点相等，则比较左右子树是否相等，
B是否在A的左子树上，
B是否在A的右子树上，
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        result = False
        if pRoot1 and pRoot2:
            if pRoot1.val == pRoot2.val:
                result = self.same(pRoot1,pRoot2)
            if not result:          #如果result是错误的,那么not result就是正确的
                result = self.HasSubtree(pRoot1.left,pRoot2)
            if not result:          #如果result是错误的,那么not result就是正确的
                result = self.HasSubtree(pRoot1.right,pRoot2)
            return result
    def same(self,p1,p2):
        if not p2:
            return True
        if not p1:
            return False
        return p1.val == p2.val and self.same(p1.left,p2.left) and self.same(p1.right,p2.right)



18.二叉树的镜像:操作给定的二叉树，将其变换为源二叉树的镜像。

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回镜像树的根节点
    def Mirror(self, root):
        # write code here
        if not root:
            return 
        if not root.left and not root.right:
            return
        temp = root.left
        root.left = root.right
        root.right = temp
        self.Mirror(root.left)
        self.Mirror(root.right)
        

19.顺时针打印矩阵::输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
# -*- coding:utf-8 -*-
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        # write code here
        res = []
        while matrix:
            #打印最上边的
            res += matrix.pop(0)
            #打印最右边的
            if matrix and matrix[0]:
                for row in matrix:
                    res.append(row.pop())
            #打印最下边的
            if matrix :
                for i in matrix.pop()[::-1]:
                    res.append(i)
            #打印最左边的
            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    res.append(row.pop(0))
        return res

20.包含main函数的栈:定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
注释:这题有点不理解出题人的思路

解题思路:
设置一个栈和一个辅助栈,辅助栈用来记录最小值,
当进栈时,将辅助栈中的最后一个元素和要进栈的元素进行对比,将较小值压入辅助栈中,这样辅助栈最后一个元素即为最小值
当出栈时,将原始栈直接出栈即可
当取栈顶元素时,直接取即可
当求最小元素时,直接取辅助栈中最后一个元素即可

# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.stack = []
        self.minlist = []
    def push(self, node):
        #进栈操作
        # write code here
        if not self.minlist:
            self.minlist.append(node)
        else:
            self.minlist.append(min(self.minlist[-1],node))
        self.stack.append(node)
    def pop(self):
        # 出栈操作
        # write code here
        if not self.stack:
            return None
        self.minlist.pop()
        return self.stack.pop()
    def top(self):
        # 取栈顶元素操作
        # write code here
        if not self.stack:
            return None
        return self.stack[-1]
    def min(self):
        # 取最小值操作
        # write code here
        if not self.minlist:
            return None
        return self.minlist[-1]


21.栈的压入弹出:输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如
序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。
（注意：这两个序列的长度是相等的）
解题思路：
借用一个辅助的栈，遍历压栈顺序，先讲第一个放入栈中，这里是1，然后判断栈顶元素是不是出栈顺序的第一个元素，这里是4，很显然1≠4，所以我们继续压栈，直到相等以后开始出栈，出栈一个元素，则将出栈顺序向后移动一位，直到不相等，这样循环等压栈顺序遍历完成，如果辅助栈还不为空，说明弹出序列不是该栈的弹出顺序。
举例：
入栈1,2,3,4,5
出栈4,5,3,2,1
首先1入辅助栈，此时栈顶1≠4，继续入栈2
此时栈顶2≠4，继续入栈3
此时栈顶3≠4，继续入栈4
此时栈顶4＝4，出栈4，弹出序列向后一位，此时为5，,辅助栈里面是1,2,3
此时栈顶3≠5，继续入栈5
此时栈顶5=5，出栈5,弹出序列向后一位，此时为3，,辅助栈里面是1,2,3

简单算法：
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        # 设置一个辅助栈
        stack = []
        while popV:
            # 如果辅助栈的最后一个元素和popV中的第一个元素相同，则都出栈
            if stack and stack[-1] == popV[0]:
                stack.pop()
                popV.pop(0)
            # 如果pushV中有数据，则压入stack中
            elif pushV:
                stack.append(pushV.pop(0))
            # 如果辅助栈为空，或者辅助栈的最后一个元素和popV中的第一个元素不相同，如果pushV中没有数据，则证明不相等
            else:
                return False
        return True

复杂算法：
# -*- coding:utf-8 -*-
class Solution:
    def IsPopOrder(self, pushV, popV):
        # write code here
        stack = []
        if len(pushV) != len(popV):
            return False
        while pushV:
            a = pushV.pop(0)
            if  a != popV[0]:
                stack.append(a)
            else :
                popV.pop(0)
                if not popV:
                    break
                while stack[len(stack) -1] == popV[0]:
                    stack.pop()
                    popV.pop(0)
                    if not stack:
                        break
                        
        if stack:
            return False
        else:
            return True





22.从上往下打印二叉树:从上往下打印出二叉树的每个节点，同层节点从左至右打印。
解法一：
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        #定义两个列表res[] 和 queue[]
        if not root:
            return []
        res = []
        queue = []
        queue.append(root)
        while len(queue) >0:
            p = queue.pop(0)        #必须是出第一个,可以动手模拟一下
            res.append(p.val)
            if p.left:
                queue.append(p.left)
            if p.right:
                queue.append(p.right)
        return res

解法二：
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        stack1 = []
        stack2 = []
        if not root:
            return stack2
        stack1.append(root)
        stack2.append(root.val)
        while stack1:
            #如果队列中有结点，则出队
            p = stack1.pop(0)
            if p.left:
                stack1.append(p.left)
                stack2.append(p.left.val)
            if p.right:
                stack1.append(p.right)
                stack2.append(p.right.val)
        return stack2


变形:从上到下打印二叉树,按层打印


class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        queue = []
        if not root:
            return None
        queue.append(root)
        last = root
        nlast = root
        while len(queue)>0:
            p = queue.pop(0)
            if p.left:
                queue.append(p.left)
                nlast = p.left
            if p.right:
                queue.append(p.right)
                nlast = p.right
            if last == p:
                print('\n')
                last = nlast
        return True

加入列表中,按层打印
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        queue = []
        res = []
        result = []
        if not root:
            return None
        queue.append(root)
        last = root
        nlast = root
        while len(queue)>0:
            p = queue.pop(0)
            res.append(p.val)
            if p.left:
                queue.append(p.left)
                nlast = p.left
            if p.right:
                queue.append(p.right)
                nlast = p.right
            if last == p:
                result.append(res)
                res.clear()
                last = nlast
        return result        


23.二叉搜索树的后序遍历序列:输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。
假设输入的数组的任意两个数字都互不相同。
解题思路：根据最后一个数将序列分为两个部分，如果左部分全部小于该数，右部分全部大于该数，则证明是后序遍历，根据这个思想递归下去
# -*- coding:utf-8 -*-
class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence:
            return False
        root=sequence[-1]
        #左边的结点序列全部小于根节点
        for i in range(len(sequence)):
            if sequence[i]>root:
                break
        #右边的结点序列全部大于根节点
        for j in range(i,len(sequence)):
            if sequence[j]<root:
                return False
        left=right=True
        # 递归执行左子树
        if i>1:
            left=self.VerifySquenceOfBST(sequence[0:i])
        # 递归执行右子树
        if i<len(sequence)-1:
            right=self.VerifySquenceOfBST(sequence[i:-1])
        return left and right



24.二叉树的和为某一路径:输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def FindPath(self, root, expectNumber):
        # write code here
        if root == None:
            return []
        result = []
        stack = []
        stack.append((root,[root.val]))    #是一个元组同时进如列表中
        while stack:
            node ,path =stack.pop()
            if node.left == None and node.right == None and sum(path)== expectNumber:
                result.append(path)
            if node.right != None:
                stack.append((node.right,path+[node.right.val]))
            if node.left != None:
                stack.append((node.left,path+[node.left.val]))
        return result

25.复杂链表的复制:输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），
返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

程序思想:此题最好画图解决，画图更加清晰，可以分为两边遍历,第一遍遍历,存放next指针,第二遍遍历存放random指针,每找到一个random指针就从头开始遍历,找到random指针指向的
下一个结点,然后在新的链表中同时生成这个指向关系。第二遍中，tmp和tmp1指向原来的链表，ntmp和ntmp1指向新的链表，其中tmp每遍历一个结点，
tmp1就从头开始遍历，找到使tmp.random == tmp1的结点，然后将新链表中的对应关系连接起来
解法一:
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if not pHead:
            return None
        tmp = pHead
        newpHead = RandomListNode(tmp.label)    
        ntmp1 = newpHead
        #先连接next结点，每次构造一个新结点
        while tmp.next :
            ntmp2 = RandomListNode(tmp.next.label)
            ntmp1.next = ntmp2
            ntmp1 = ntmp1.next
            tmp =tmp.next
       #连接random结点，tmp和tmp1指向原来的结点，ntmp和ntmp1指向新链表的结点，tmp每遍历一个结点，tmp1就从头开始遍历
        #直到tmp.random = tmp1
        tmp = pHead
        ntmp = newpHead
        while tmp:
            if not tmp.random:
                tmp = tmp.next
                ntmp = ntmp.next
                continue
            tmp1 = pHead
            ntmp1 = newpHead
            while tmp1:
                if tmp.random == tmp1:
                    break
                else:
                    tmp1 = tmp1.next
                    ntmp1 = ntmp1.next
            ntmp.random = ntmp1
            tmp = tmp.next
            ntmp = ntmp.next
        return newpHead
   

解法二:
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if not pHead:
            return None
        #在原链上复制链，在每个节点后插入一个复制节点
        tmp = pHead
        while tmp:
            ntmp = RandomListNode(tmp.label)
            ntmp.next = tmp.next
            tmp.next = ntmp
            tmp = ntmp.next
        #遍历合成链，复制random关系
        tmp = pHead
        while tmp:
            tmp.next.random = tmp.random.next if tmp.random else None
            tmp = tmp.next.next
        #拆链
        tmp,ntmp = pHead,pHead.next
        newHead = pHead.next
        while tmp:
            tmp.next = ntmp.next
            tmp = tmp.next
            if not tmp:
                break
            ntmp.next = tmp.next
            ntmp = ntmp.next
        return newHead
            


26.二叉搜索树与双向链表:输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

解题思路：中序搜索二叉树转化成排序的双向链表，其实就是对二叉搜索树进行中序遍历，将遍历的结点存在列表中，然后对列表中的结点进行链接

解法一:非递归做法
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, pRootOfTree):
        # write code here
        if not pRootOfTree:
            return None
        
        p = pRootOfTree
        stack = []
        resStack = []
        #中序遍历
        while p or stack:
            if p :
                stack.append(p)
                p = p.left
            else:
                node = stack.pop()
                resStack.append(node)
                p = node.right
        resP =resStack[0]
        #遍历resStack栈,,增加指针
        while resStack:
            top = resStack.pop(0)
            if resStack:
                top.right = resStack[0]
                resStack[0].left = top
        return resP
        
解法二:递归做法

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Convert(self, root):
        if not root:
            return None
        if not root.left and not root.right:
            return root
         
        # 将左子树构建成双链表，返回链表头
        left = self.Convert(root.left)
        p = left
         
        # 定位至左子树的最右的一个结点
        while left and p.right:
            p = p.right
         
        # 如果左子树不为空，将当前root加到左子树链表
        if left:
            p.right = root
            root.left = p
         
        # 将右子树构造成双链表，返回链表头
        right = self.Convert(root.right)
        # 如果右子树不为空，将该链表追加到root结点之后
        if right:
            right.left = root
            root.right = right
             
        return left if left else root




27.字符串的排列:输入一个字符串,按字典序打印出该字符串中字符的所有排列。
例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

此题不是太理解，递归学的不好
# -*- coding:utf-8 -*-
class Solution:
    def Permutation(self, ss):
        # write code here
        l = []
        if len(ss) <= 1:
            return ss
        n = len(ss)
        i = 0
        while i < n:
            #temp = ss[i] + self.Permutation(ss[:i]+ss[i+1:])
            for j in self.Permutation(ss[:i]+ss[i+1:]):
                temp = ss[i] + str(j)
                if temp not in l:
                    l.append(temp)
            i += 1
        return l



28.数组中出现次数超过一半的数字:数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

解法一:对数组排序，找数组中间的数。如果中间数出现次数大于数组长度的一半，就返回

# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        numbers.sort()
        midnum = numbers[len(numbers)/2]
        if numbers.count(midnum) > len(numbers)/2:
            return midnum
        return 0

解法二:将列表中的值加入一个字典中,键记录列表中元素,值用来记录出现次数,最后遍历字典即可


# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        
        #定义一个字典,键存数组中的值,值存出现次数,最后遍历字典,
        #查找字典中出现次数是否有大于一半的
        dic = {}
        for item in numbers:
            if item not in dic:
                dic[item] = 1
            else:
                dic[item] += 1
        for item in dic.keys():
            if dic[item] > len(numbers)/2:
                return item
        return 0
解法三：时间复杂度有点大，不推荐使用。遍历列表，设置一个计数器，用来记录每个数字的出现次数

# -*- coding:utf-8 -*-
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        # write code here
        for i in numbers:
            count =0
            for j in numbers:
                if j ==i:
                    count += 1
            if count >len(numbers)/2:
                return i
        return 0


29.最小可k个数:输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

解法一:直接调用python的内函数sort排序,输出前k个数即可
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        if tinput == [] or k > len(tinput):
            return []
        tinput.sort()
        return tinput[:k]

解法二:利用快速排序的算法思想
# -*- coding:utf-8 -*-
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        # write code here
        def list_sort(lst):
            if not lst:
                return []
            tmp = lst[0]
            left = list_sort([x for x in lst[1:] if x < tmp])
            right = list_sort([x for x in lst[1:] if x > tmp])
            return left + [tmp] + right
        if tinput == [] or k > len(tinput):
            return []
        tinput = list_sort(tinput)
        return tinput[:k]

解法三:也可以用冒泡排序,直接插入排序,二分法排序,简单选择排序



30.连续子数组的最大和:HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。
今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。
但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8
(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

解法一:利用动态规划的思想（动态规划学的不好）
# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        n =len(array)
        dp = [i for i in array]
        for i in range(1,n):
            dp[i] = max(dp[i-1] + array[i],array[i])
        return max(dp)


解法二：遍历列表，max_sum记录最大值，用max记录每一次遍历的临时最大值，用sum记录和，遍历完列表即可得到最大值

# -*- coding:utf-8 -*-
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        max_sum= array[0]
        for i in range(len(array)):
            max = array[i]
            sum = array[i]
            for j in array[i+1:len(array)]:
                sum += j
                if sum > max:
                    max = sum
            if max > max_sum:
                max_sum =max
        return max_sum



31.整数中1出现的次数,(从1到n整数中1出现的次数): 
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,
但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

解法一:
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        return ''.join([str(i) for i in range(1, n+1)]).count('1')

解法二:
# -*- coding:utf-8 -*-
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        # write code here
        count =0 
        for i in range(1,n+1):
            for j in str(i):	#遍历字符串
                if "1" == j:
                    count += 1
        return count

解法三：分离出来每一位数字，然后进行统计



32.把数组排成最小的数:
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。
例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

解题思路：使用冒泡排序的思想，或者其他排序思想均可以，每次将最小值放在最前边

解法一:先将数组中每个元素转换成String类型，然后进行排序，如果str(a) + str(b) > str(b) + str(a),说明ab > ba，
应该把b排在a前面。使用冒泡排序编写程序如下：
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        n = len(numbers)
        for i in range(n):
            for j in range(i+1,n):
                if int(str(numbers[i]) + str(numbers[j])) > int(str(numbers[j]) + str(numbers[i])):
                    numbers[i],numbers[j] = numbers[j] ,numbers[i]
        return "".join([str(i) for i in numbers])

解法二:
# -*- coding:utf-8 -*-
class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        lmb = lambda n1, n2: int(str(n1) + str(n2)) - int(str(n2) + str(n1))
        a = sorted(numbers, cmp=lmb)
        return ''.join([str(i) for i in a])

说明:这种方式不知道为什么在pycharm中运行不通,sorted函数里面可以跟cmp或者key关键字


33.丑数:把只包含质因子2、3和5的数称作丑数（Ugly Number）。
例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

解析:通俗易懂的解释：
首先从丑数的定义我们知道，一个丑数的因子只有2,3,5，那么丑数p = 2 ^ x * 3 ^ y * 5 ^ z，换句话说一个丑数一定由另一个丑数乘以2或者乘以3
或者乘以5得到，那么我们从1开始乘以2,3,5，就得到2,3,5三个丑数，在从这三个丑数出发乘以2,3,5就得到4，6,10,6，9,15,10,15,25九个丑数，
我们发现这种方法会得到重复的丑数，而且我们题目要求第N个丑数，这样的方法得到的丑数也是无序的。

思路：我们可以维护三个list，分别是乘2得到的丑数，乘3得到的丑数，乘5得到的丑数，但这样复杂度较高，而且会得到重复的丑数。
实际上每次我们只用比较3个数：用于乘2的最小的数、用于乘3的最小的数，用于乘5的最小的数。这样只需要维护三个指针，而不用维护三个数组。

def GetUglyNumber_Solution(self, index):
    # write code here
    if index <= 0:
        return 0
    res = [1]
    p2 = 0 # p2指向小于newUgly且最大的乘以2后可能成为下一个丑数的丑数
    p3 = 0 # p3指向小于newUgly且最大的乘以3后可能成为下一个丑数的丑数
    p5 = 0 # p5指向小于newUgly且最大的乘以5后可能成为下一个丑数的丑数
    for i in range(index-1):
        n = min(res[p2]*2, res[p3]*3, res[p5]*5)
        res.append(n)
        if res[-1] == res[p2]*2:
            p2 += 1
        if res[-1] == res[p3]*3:
            p3 += 1
        if res[-1] == res[p5]*5:
            p5 += 1
    return res[-1]



34.第一个只出现一次的字符:在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 
如果没有则返回 -1（需要区分大小写）.

解法一:这种方式时间复杂度为n^2

# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code 
        n = len(s)
        for i in range(n):
            tmp = s[:i] + s[i+1:]
            if s[i] not in tmp:
                return i
        return -1


解法二:利用一个字典,键用来存储出现的字母,,值用来记录出现的次数,时间复杂度为n

# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code 
        dic = {}
        for i in range(len(s)):
            if s[i] not in dic:
                dic[s[i]] = 1
            else:
                dic[s[i]] += 1
        for i in range(len(s)):
            if dic[s[i]] == 1:
                return i
        return -1


解法三:利用一个有序进入字典,即OrderedDict,把所有的值按顺序加入字典中的键,值为一个列表[0,1],0表示出现的索引位置,1表示出现的次数

# -*- coding:utf-8 -*-
from collections import OrderedDict
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code 
        dic = OrderedDict()
        for i in range(len(s)):
            if s[i] not in dic:
                dic[s[i]] = [i,1]
            else:
                dic[s[i]][1] += 1
        for item in dic.values():
            if item[1]  == 1:
                return item[0]
        return -1

       
解法四：
# -*- coding:utf-8 -*-
class Solution:
    def FirstNotRepeatingChar(self, s):
        # write code here
        for i in s:
            if s.count(i) == 1:
                return s.index(i)
        return -1
