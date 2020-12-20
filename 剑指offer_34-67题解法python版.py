剑指offer_34-67题解法


34.数组中的逆序对:在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。
并将P对1000000007取模的结果输出。 即输出P%1000000007
输入描述:
题目保证输入的数组中没有的相同的数字

数据范围：

	对于%50的数据,size<=10^4

	对于%75的数据,size<=10^5

	对于%100的数据,size<=2*10^5


解法:此题有难度


35.两个链表的公共结点:输入两个链表，找出它们的第一个公共结点。

解法一:先找出两个链表的差,对于长的链表先遍历差个单位,然后同时遍历,这种不知道为什么不行
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if not pHead1 or not pHead2:
            return None
        p1,p2 = pHead1,pHead2
        len1 = len2 = 0
        while p1 :
            len1 += 1
            p1 = p1.next
        while p2 :
            len2 += 1
            p2 = p2.next
        if p1 > p2:
            count =0
            while count < len1 - len2:
                p1  = p1.next
                count += 1
        else :
            count = 0
            while count < len2 - len1:
                p2 = p2.next
                count += 1
        while p1 and p2 :
            if p1 == p2:
                return p1
            else:
                p1 = p1.next
                p2 = p2.next
        return None


解法二:借助一个列表,遍历两个链表,将所有的结点都加入列表中,然后从最后一个结点倒着,将相同的结点加入另一个列表中,如果遇到不相同的结点,就返回
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if not pHead1 or not pHead2:
            return None
        res1 = []
        res2 = []
        result = []
        p1,p2 = pHead1,pHead2
        while p1:
            res1.append(p1)
            p1 = p1.next
        while p2:
            res2.append(p2)
            p2 = p2.next
        while res1 and res2:
            node1 = res1.pop()
            node2 = res2.pop()
            if node1 == node2:
                result.append(node1)
        if result:
            return result.pop()

解法三:设置一个集合,将链表一中的每一个结点都加入集合中,遍历链表2,如果有结点在这个集合中则返回
# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        # write code here
        if not pHead1 or not pHead2:
            return None
        res_set = set()
        while pHead1:
            res_set.add(pHead1)
            pHead1 = pHead1.next
        while pHead2:
            if pHead2 in res_set:
                return pHead2
            else:
                pHead2 = pHead2.next
        return None

36.数字在排序数组中出现的次数:统计一个数字在排序数组中出现的次数。


解法一:直接调用python函数的count功能
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        return data.count(k)


解法二:暴力解法,直接遍历数组,一个一个对比,时间复杂度为n
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        count = 0
        for i in data:
            if i == k :
                count += 1
        return count


解法三:利用二分法查找,先找到等于k的最左边的索引值,,再找到最右边的索引值,然后索引之差就是k的重复次数

# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        #找出来最左边索引
        def getfirst(data,k):
            left = 0
            right = len(data) -1 
            while right >= left:
                if data[left] == k:
                    return left
                mid = (left + right)//2
                if data[mid] > k:
                    right = mid -1
                elif data[mid] < k:
                    left = mid +1
                else:
                    if data[mid] == k and data[mid-1]!=k:
                        return mid
                    right = mid -1
            return -1
        # 找出来最右边索引
        def getlast(data,k):
            left = 0
            right = len(data)-1
            while left <= right:
                if data[right] == k:
                    return right
                mid = (right + left)//2
                if data[mid] > k:
                    right = mid-1
                elif data[mid] < k:
                    left = mid + 1
                else:
                    if data[mid] ==k and data[mid+1]!=k:
                        return mid
                    left = mid+1
            return -1
        #左右索引之差即为k的个数
        num = 0
        if data:
            first = getfirst(data,k)
            last = getlast(data,k)
            if first > -1 and last > -1:
                num = last - first +1
                return num
        return num




37.二叉树的深度:输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。


解法一:递归法

class Solution:
    def TreeDepth(self, pRoot):
 
        if pRoot==None:return 0
        return max(self.TreeDepth(pRoot.left),self.TreeDepth(pRoot.right))+1

解法二:利用树的层次遍历算法,按照换行输出,每一行加入一个列表中,然后再把该行的所有数加入一个大列表中
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        res = []
        queue = []
        result = []
        root = pRoot
        queue.append(root)
        last = root
        nlast = root
        while queue:
            root = queue.pop(0)
            res.append(root)
            if root.left:
                queue.append(root.left)
                nlast = root.left
            if root.right:
                queue.append(root.right)
                nlast = root.right
            if last == root:
                result.append(res)
                res = []
                last = nlast
        return len(result)

解法三:利用层次遍历的算法,每遍历一个结点就把该结点和结点所在的层数以列表的形式加入列表中,最后求最后一个结点所在的层数,即为树的最大深度

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def TreeDepth(self, pRoot):
        # write code here
        if not pRoot:
            return 0
        root = pRoot
        res = []
        queue = []
        queue.append([1,root])		#1代表树的层数,root代表树的结点
        while queue:
            node = queue.pop(0)
            lno = node[0]			#获取当前树所在的层数
            root = node[1]			#获取当前树的结点
            res.append(node)
            if root.left:
                queue.append([lno+1,root.left])			#把层数和结点同时以数组的形式入列表
            if root.right:
                queue.append([lno+1,root.right])
        return res[-1][0]					#返回最大层数


38.平衡二叉树:输入一棵二叉树，判断该二叉树是否是平衡二叉树。
解题思路：
如果二叉树的每个节点的左子树和右子树的深度不大于1，它就是平衡二叉树。
先写一个求深度的函数，再对每一个节点判断，看该节点的左子树的深度和右子树的深度的差是否大于1

# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def IsBalanced_Solution(self, root):
        if not root:
            return True
        if abs(self.maxDepth(root.left) - self.maxDepth(root.right)) > 1:
            return False
        return self.IsBalanced_Solution(root.left) and self.IsBalanced_Solution(root.right)
    
    def maxDepth(self, root):
        if not root: 
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

最后一行深度加1是因为加上根节点，所以深度加1



39.数组中只出现一次的数字:一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。


解法一:利用count函数,遍历列表,如果出现次数等于1次,就加入列表中
# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        res = []
        for i in range(len(array)):
            if array.count(array[i]) == 1:
                res.append(array[i])
        return res

解法二:利用字典,遍历列表,每遍历一个就加入字典中,值用来记录出现次数,最后遍历字典即可

# -*- coding:utf-8 -*-
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        # write code here
        dic = {}
        res = []
        for i in array:
            if i not in dic :
                dic[i] = 1
            else:
                dic[i] += 1
        for i in dic.items():
            if i[1] == 1:
                res.append(i[0])
        return res

40.和为S的连续正数序列:小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。
但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。
现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

输入描述:输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序

解法二更简单
解法一:不是太懂
# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        l,r,sum,res =1,2,3,[]
        while l < r:
            if sum > tsum:
                sum -=l
                l +=1
            else:
                if sum == tsum:
                    res.append([i for i  in range(l,r+1)])
                r += 1
                sum +=r
        return res

解法二:设置两个for循环,依次遍历,找到相邻的几个数和等于指定的数字,添加到列表中

# -*- coding:utf-8 -*-
class Solution:
    def FindContinuousSequence(self, tsum):
        # write code here
        res =[]
        for i in range(1,tsum):
            sum = i
            for j in randge(i+1,tsum//2+2):
                sum +=j
                if sum ==tsum:
                    res.append([x for x in range(i,j+1)])
                    break
                elif sum > tsum:
                    break
        return res


41.和为s的两个数字:
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。

解法一:遍历列表,找到两个值和为tsum并且乘积最小的即可
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        res =[]
        min_result = tsum * tsum
        for i in range(len(array)):
            for j in range(i+1,len(array)):
                if array[i] + array[j] == tsum and array[i]*array[j] < min_result:
                    min_result = array[i] * array[j]
                    res.append(array[i])
                    res.append(array[j])
        return res



解法二：
# -*- coding:utf-8 -*-
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        # write code here
        max = 0
        res =[]
        for i in range(len(array)):
            j = tsum - array[i]
            if j in array:
                num = array[i] * j
                if num > max:
                    res.append(array[i])
                    res.append(j)
        return res[-2:]



42.左旋转字符串:
汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，
请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

解法一:
利用字符串的特性将前n个数和后面的数交换位置即可
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        if n == 0 or  len(s) <=1:
            return s
        else:
            return s[n:] + s[:n]


解法二：
类似于数据结构中循环链表的特点，先初始化一个列表，然后计算该位置的元素向左移动n位后的索引
# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        if n ==0 or n > len(s):
            return s
        res = [0 for i in range(len(s))]
        for i in range(len(s)):
            index = (i-n+len(s))%len(s)
            res[index] = s[i]
        return ''.join(res)


解法三:

# -*- coding:utf-8 -*-
class Solution:
    def LeftRotateString(self, s, n):
        # write code here
        length = len(s)
        if n == 0 or length <=1:
            return s
        else:
            s = list(s)
            n = n%length
            self.string_reverse(s,0,n-1)
            self.string_reverse(s,n,length -1)
            self.string_reverse(s,0,length -1)
            return "".join(s)
    def string_reverse(self,string,start,end):
        while start < end:
            tmp = string[start]
            string[start] =string[end]
            string[end] = tmp
            start +=1
            end -=1


43.翻转单词顺序列:
牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。
如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

解法一:
# -*- coding:utf-8 -*-
class Solution:
    def ReverseSentence(self, s):
        # write code here
        return " ".join(s.split(" ")[::-1])



44.扑克牌顺子:
解题思路：
对列表进行排序，统计大王和小王的数量，即0的数量，如果除去0的数量以后，有两个数相等，则不满足。如果最后的数和第一个数的差大于0的数量+非0的数量，那么就不相等。
# -*- coding:utf-8 -*-
class Solution:
    def IsContinuous(self, numbers):
        # write code here
        if not numbers:
            return False
        tmp = sorted(numbers)
        count = 0
        for i in tmp:
            if i == 0:
                count += 1
            else:
                break
        new_num = tmp[count:]
        for i in new_num:
            if new_num.count(i) > 1:
                return False
        if new_num[-1] - new_num[0] > count + len(new_num) -1:
            return False
        return True


45.孩子们的游戏:


解法一:
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        #s使用list模拟循环链表,用cui作为指向list的下标位置
        #当cur移动到list末尾直接指向list头部,当删除一个数后list的长度和cur的值相等则cur指向0
        if n < 1 or m < 1:
            return -1
        childNum = [x for x in range(n)]
        print(childNum)
        cur = 0
        while len(childNum) > 1:
            for i in range(1,m):
                cur += 1
                if cur == len(childNum):
                    cur = 0
            childNum.remove(childNum[cur])
            if cur == len(childNum):
                cur = 0
        return childNum[-1]


解法二:利用约瑟夫环:
先初始化child当做孩子数，模拟循环链表
# -*- coding:utf-8 -*-
class Solution:
    def LastRemaining_Solution(self, n, m):
        # write code here
        if not n or not m:
            return -1
        child = [x for x in range(n)]
        start =0
        while len(child) >1:
            end = (start +m-1)%len(child)
            child.pop(end)
            start = end
        return child[0]


46.求1+2+3+4+......+n:
# 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

解法一:
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        return sum(list(range(1,n+1)))



解法二:
# -*- coding:utf-8 -*-
class Solution:
    def Sum_Solution(self, n):
        # write code here
        #如果a和b都不为0,则返回b,如果a为0,b不为0,则返回a,如果a不为0,b为0,则返回b
        return n and n + self.Sum_Solution(n-1)

搞清楚 n and n+1 在Python中的关系


47.不用加减乘除做加法:写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

链接：https://www.nowcoder.com/questionTerminal/59ac416b4b944300b617d4f7f111b215?f=discussion
来源：牛客网
# -*- coding:utf-8 -*-
class Solution: 
    def Add(self, a, b):           
        while(b): 
           a,b = (a^b) & 0xFFFFFFFF,((a&b)<<1) & 0xFFFFFFFF
        return a if a<=0x7FFFFFFF else ~(a^0xFFFFFFFF)



48.把字符串转化为整数:将一个字符串转换成一个整数，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0

此题没有任何意义

思路:将字符串转化为数字,需要考虑很多情况,有无正负号,有无非法字符,是否是空字符串,此题找不到正确答案,下面的解法会导致栈溢出
# -*- coding:utf-8 -*-
class Solution:
    def StrToInt(self, s):
        # write code here
        if not s:
            return 0
        flag_dic = {'+':1,'-':-1}
        num={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':0}
        first = s[0]
        if first in ['+','-']:
            flag = flag_dic[first]
            if len(s) == 1:
                return 0
            x = 0
            for i in s[1:]:
                if i not in num:
                    return 0
                x = x*10 + num[i]
            return flag * x
        else:
            x = 0
            for i in s:
                if i not in num:
                    return 0
                x = x*10 + num[i]
            return x




49.数组中重复的数字:在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。
请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

解法一:设置一个集合,如果不在集合中,就加入集合,如果发现在集合中,则存入列表中
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        num_set = set()
        for i in numbers:
            if i not in num_set:
                num_set.add(i)
            else:
                duplication[0]=i
                return True
        return False

说明:用列表也可以求解

解法二:两重循环
# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        for i in range(len(numbers)):
            for j in range(i+1,len(numbers)):
                if numbers[i] == numbers[j]:
                    duplication[0]= numbers[i]
                    return True
        return False


解法三：
利用python函数中的count函数进行统计

# -*- coding:utf-8 -*-
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        # write code here
        for i in range(len(numbers)):
            if numbers.count(numbers[i]) >1:
                duplication[0] = numbers[i]
                return True
        return False



50.构建乘积数组:

此题比较简单，思路略过

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

解法一：
# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        # write code here
        B =[0] * len(A)             #因为空数组不能通过索引赋值,所以初始化数组全部为0
        for i in range(len(A)):
            mul = 1
            for j in range(len(A)):
                if i != j:          #如果索引不相等,则计算乘积
                    mul = mul * A[j]
                else:               #如果索引相等,则直接跳过
                    continue
            B[i] = mul              #将乘积存入数组
        return B


解法二：

# -*- coding:utf-8 -*-
class Solution:
    def multiply(self, A):
        # write code here
        if not A:
            return None
        B = []
        for i in range(len(A)):
            n = 1
            for j in range(len(A)):
                if i == j :
                    continue
                n = n*A[j]
            B.append(n)
        return B



51.正则表达式匹配:
请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 
在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

思路:
链接：https://www.nowcoder.com/questionTerminal/45327ae22b7b413ea21df13ee7d6429c?answerType=1&f=discussion
来源：牛客网

思路：当模式中的第二个字符是“*”时：
如果字符串第一个字符跟模式第一个字符不匹配，则模式后移2个字符，继续匹配。如果字符串第一个字符跟模式第一个字符匹配，可以有3种匹配方式：
1.模式后移2字符，相当于x*被忽略；
2.字符串后移1字符，模式后移2字符，相当于x*匹配一位；
3.字符串后移1字符，模式不变，即继续匹配字符下一位，相当于x*匹配多位；
当模式中的第二个字符不是“*”时：
如果字符串第一个字符和模式中的第一个字符相匹配，那么字符串和模式都后移一个字符，然后匹配剩余的部分。
如果字符串第一个字符和模式中的第一个字符相不匹配，直接返回False。
# -*- coding:utf-8 -*-
class Solution:
    # s, pattern都是字符串
    def match(self, s, pattern):
        # write code here
        if (len(s) == 0 and len(pattern) ==0):  #如果子串和模式串都为0,则返回正确
            return True
        if (len(s)>0 and len(pattern) == 0):        #如果子串不为0,并且模式串为0,则返回错误
            return False
        if len(pattern) > 1 and pattern[1] == "*":          #如果模式串中第二个字符为*号时,
            if s and(pattern[0] == "." or s[0] ==pattern[0]):   #如果模式串中第二个字符为*号时,并且子串和模式串第一个匹配上
                f1 = self.match(s[1:],pattern)    #多个
                f2 = self.match(s[1:],pattern[2:])    #匹配一个
                f3 = self.match(s,pattern[2:])    #匹配0个
                if f1 or f2 or f3:                  #如果f1,f2,f3出现一个为True,则返回结果为True
                    return True
                else:
                    return False
            else:
                return self.match(s,pattern[2:])        #如果子串和模式串第一个字符没有匹配上,则模式串后移两位
        elif s and (pattern[0] == '.' or s[0]==pattern[0]):     #如果第二个字符不为*时,且第一个字符能匹配上
            return self.match(s[1:],pattern[1:])            #子串和模式串分别后移一位
        else:
            return False

注释：此题没有任何意义，就算会做这题，下次再碰到改编的题，就又不会做了


52.表示数值的字符串:
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 
但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

# -*- coding:utf-8 -*-
class Solution:
    # s字符串
    def isNumeric(self, s):
        # write code here
        try :
            ss = float(s)
            return True
        except:
            return False



53.字符流中第一个不重复的字符:
请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
当从该字符流中读出前六个字符"google"时，第一个只出现一次的字符是"l"。

此题没啥意义

# -*- coding:utf-8 -*-
class Solution:
    # 返回对应char
    def __init__(self):
        self.dic = {}
        self.s = ''

    def FirstAppearingOnce(self):
        # write code here
        for i in self.s:
            if self.dic[i] ==1:
                return i
        return "#"

    def Insert(self, char):     #每次只输入一个字符
        # write code here
        self.s = self.s+char
        if char in self.dic:
            self.dic[char]+=1
        else:
            self.dic[char] =1

54.链表中环的入口结点:
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def EntryNodeOfLoop(self, pHead):
        # write code here
        node_set = set()
        while pHead:
            if pHead in node_set:
                return pHead
            else:
                node_set.add(pHead)
                pHead= pHead.next

说明:用列表也可以



55.在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if not pHead or not pHead.next :
            return pHead
        tmp = ListNode(0)
        tmp.next = pHead
        pre = tmp
        p = pHead
        while p:
            if p.next and p.val == p.next.val:
                while p.next  and p.next.val == p.val:
                    p = p.next
                pre.next = p.next
                p = p.next
            else:
                pre = p
                p = p.next
        return tmp.next



56.二叉树的下一个结点:
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

解题思路：
如果它是一个父节点，输出右孩子的最左边结点，
如果它是父节点的左孩子结点，则返回它的父节点
如果它是父节点的右孩子结点，则不断向上搜索父节点，然后返回
# -*- coding:utf-8 -*-
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        # write code here
        if pNode.right :        #如果是父节点,且父节点有有孩子
            pNode = pNode.right     #找到右孩子的最左边节点
            while pNode.left:
                pNode = pNode.left  
            return pNode
        while pNode.next and pNode.next.right == pNode:     #如果不是父节点,并且是父节点的右孩子,那么找父结点
            pNode = pNode.next
        if pNode.next:
            return pNode.next
        return None


说明:画图模拟一遍,这个题有点难度,顺序不能打乱,一定得先判断是否有右孩子,然后判断是否是父结点的右孩子,然后判断是否是父结点的左孩子



57.对称二叉树:
请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self,pRoot):
        # write code here
        if not pRoot:
            return True
        return self.compare(pRoot.left,pRoot.right)
    def compare(self,pRoot1,pRoot2):
        if not pRoot1 and not pRoot2:
            return True
        if not pRoot1 or not pRoot2:
            return False
        if pRoot1.val == pRoot2.val:
            if self.compare(pRoot1.left,pRoot2.right) and self.compare(pRoot1.right,pRoot2.left):
                 return True
        return False



58.按之字型打印二叉树:
请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

解法一:自己写的

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        queue =[]
        res = []
        result = []
        queue.append(pRoot)
        root = pRoot
        last = root
        nlast = root
        while len(queue)>0:
            root = queue.pop(0)
            res.append(root.val)
            if root.left:
                queue.append(root.left)
                nlast = root.left
            if root.right:
                queue.append(root.right)
                nlast = root.right
            if last == root:
                result.append(res)
                last  = nlast
                res = []
        return_list = []
        for i,j in enumerate(result):
            if i %2 != 0:
                return_list.append(j[::-1])
            else:
                return_list.append(j[::])
        return return_list


解法二:参考华科平凡

链接：https://www.nowcoder.com/questionTerminal/91b69814117f4e8097390d107d2efbe0?f=discussion&toCommentId=5076059
来源：牛客网

class Solution:
    def Print(self, pRoot):
        if not pRoot:
            return []
        nodeStack=[pRoot]
        result=[]
        while nodeStack:
            res = []
            nextStack=[]
            for i in nodeStack:
                res.append(i.val)
                if i.left:
                    nextStack.append(i.left)
                if i.right:
                    nextStack.append(i.right)
            nodeStack=nextStack
            result.append(res)
        returnResult=[]
        for i,v in enumerate(result):
            if i%2==0:
                returnResult.append(v)
            else:
                returnResult.append(v[::-1])
        return returnResult


59.把二叉树打印为多行:
从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

解法一:参考华科平凡
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        # write code here
        if not pRoot:
            return []
        root = pRoot
        res =[]
        result =[]
        queue =[]
        queue.append(pRoot)
        while len(queue):
            res = []
            node_level = []             #用来装每一层的结点,
            for i in queue:
                res.append(i.val)       #将每一层的结点都加入列表中
                if i.left:
                    node_level.append(i.left)
                if i.right:
                    node_level.append(i.right)
            queue = node_level
            result.append(res)          #将每一层的结点都加入一个大列表中    
        return result


解法二:自己想的:
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


60.序列化二叉树:

解法一:摘自牛客网>

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.flag = -1
    def Serialize(self, root):      #序列化二叉树
        # write code here
        if not root:
            return '#,'
        return str(root.val)+','+self.Serialize(root.left)+self.Serialize(root.right)       #连接成字符串
    def Deserialize(self, s):       #反序列化二叉树
        # write code here
        self.flag += 1
        l = s.split(',')
         
        if self.flag >= len(s):
            return None
        root = None
         
        if l[self.flag] != '#':
            root = TreeNode(int(l[self.flag]))
            root.left = self.Deserialize(s)
            root.right = self.Deserialize(s)
        return root


61.二叉搜索树的第k个结点:
给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。

解法一:按照中序遍历二叉树就可以,注意返回的是结点,而不是结点的值

# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        # write code here
        if not pRoot:
            return pRoot
        if k <= 0 :
            return None
        root = pRoot 
        res = []
        result = []
        while res or root:
            if root:
                res.append(root)
                root = root.left
            else:
                root = res.pop()
                result.append(root)
                root = root.right
        if k >len(result):
            return None
        return result[k-1]

62.数据流中的中位数:
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，
那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.data=[]
    def Insert(self, num):
        # write code here
        self.data.append(num)
        self.data.sort()
    def GetMedian(self, data):
        # write code here
        length=len(self.data)
        if length%2==0:
            #下面这行代码有问题，为什么除以2就会报错，除以2.0就不会报错
            return(self.data[length//2]+self.data[length//2-1])/2.0
        else:
            return self.data[int(length//2)]



63.滑动窗口最大值:
给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，
他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}，
 {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

解法:利用列表的形式,遍历列表,每次找出连续3个数中的最大值
# -*- coding:utf-8 -*-
class Solution:
    def maxInWindows(self, num, size):
        # write code here
        max_windows = []
        if size <= 0:
            return max_windows
        for i in range(len(num)):
            if i+size <= len(num):
                temp = max(num[i:i+size])
                max_windows.append(temp)
        return max_windows



64.矩阵中的路径:
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，
向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路径，
但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

说明:参考自郭家兴
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self._dict = {}
    def hasPath(self, matrix, rows, cols, path):
        # write code here
        # 将矩阵转成二维数组表示
        x = [list(matrix[i*cols:(i+1)*cols]) for i in range(rows)]
        for i in range(rows):
            for j in range(cols):
                self._dict = {}
                if self.dfs(x, i, j, path):
                    return True
        return False

    def dfs(self, matrix, i, j, p):
        if not (0 <= i< len(matrix) and 0 <= j < len(matrix[0])):  # 越界
            return False
        if matrix[i][j] != p[0]:  # 不匹配
            return False
        if self._dict.get((i,j)) is not None:  # 重复路径
            return False
        self._dict[(i,j)] = 1
        if not p[1:]:
            return True
        # 向上下左右寻找
        return self.dfs(matrix,i+1,j,p[1:]) or self.dfs(matrix,i-1,j,p[1:]) or self.dfs(matrix,i,j+1,p[1:]) or self.dfs(matrix,i,j-1,p[1:])
65.机器人的运动范围:
地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 
例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

解题思路：
这道题跟前一道题一样，也是回溯法，分析题目，我们需要两个全局变量：标志数组和计数变量；需要一个函数来计算行坐标和列坐标的数位之和；
终止条件包括三种情况：越界、重复、行坐标和列坐标的数位之和超过k，然后流程和上一道题相同。
设置一个字典，用来记录坐标是否被访问过，设置一个计数器，用来记录符合要求的坐标个数，自定义一个函数，用来计算行坐标和列坐标的数位之和，
当输入的坐标越界时，或者当坐标已经在字典中，表示被访问过，或者当行坐标和列坐标之和超过k时，就返回

# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self._dict ={}        #设置字典用来记录能够到达的格子是否被访问过,键为坐标点
        self.count = 0
    def get_sum(self,i,j):        #用来得到坐标的各位数字之和
        num = 0
        while i :
            temp = i%10
            i = i/10
            num += temp
        while j:
            temp = j%10
            j = j/10
            num+= temp
        return num
    def dfs(self,matrix,k,i,j):
        if not(0<=i<len(matrix) and 0 <= j<len(matrix[0])):  #数组越界
            return 
        if self.get_sum(i,j)>k:        #不满足要求
            return 
        if self._dict.get((i,j)) is not None:        #用来判断是够加入字典中
            return
        self._dict[(i,j)] =1        #如果数组没有越界,小于k,并且没有在字典中,就加入字典中
        self.count +=1              #count用来记录个数
        #向上向下左右寻找
        self.dfs(matrix,k,i+1,j)        #向下寻找
        self.dfs(matrix,k,i-1,j)        #向上寻找
        self.dfs(matrix,k,i,j+1)        #向右寻找
        self.dfs(matrix,k,i,j-1)        #向左寻找
    def movingCount(self, threshold, rows, cols):
        # write code here
        x = [[1 for i in range(cols)] for j in range(rows)]
        self.dfs(x,threshold,0,0)
        return self.count


66.剪绳子:
给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积
是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。

解法一:动态规划
# -*- coding:utf-8 -*-
class Solution:
    def cutRope(self, number):
        # write code here
        res = 1
        if number <=1:
            return 0
        elif number <=2:
            return 1
        elif number <=3:
            return 2
        prod = [0,1,2,3]
        for i in range(4,number +1):
            max =0
            for j in range(1,i//2+1):
                pro = prod[j]*prod[i-j]
                if pro >max:
                    max = pro
            prod.append(max)
        return prod[number]

解法二:贪心算法

# -*- coding:utf-8 -*-
class Solution:
    def cutRope(self, number):
        # write code here
        res =1
        if number <=1:
            return 0
        elif number <=2:
            return 1
        elif number <=3:
            return 2
        elif number >3:
            if number%3 ==0:
                res = 3**(number//3)
            elif number%3 ==1:
                res = 3**(number//3-1)*4
            else:
                res = 3**(number//3)*(number%3)
        return res


附录:

动态规划经典例题——最长公共子序列和最长公共子串

#求两个字符串的最大公共子序列(不连续)
def lcs(string1,string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1] +1
            else:
                res[i][j] = max(res[i-1][j] ,res[i][j-1])
    return res[-1][-1]


求两个字符串的最大公共子序列（连续）
def LCstring(string1,string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1+1)] for j in range(len2+1)]
    result = 0
    for i in range(1,len2+1):
        for j in range(1,len1+1):
            if string2[i-1] == string1[j-1]:
                res[i][j] = res[i-1][j-1]+1
                result = max(result,res[i][j])
    return result
print(LCstring("helloworld","loop"))





将两个有序链表合并为一个有序链表
list1 =[1,3,5,7,9]
list2 =[2,4,6,8,10]
i =j =0
res =[]
while i <len(list1) and j <len(list2):
    if list1[i] > list2[j]:
        res.append(list2[j])
        j +=1
    else:
        res.append(list1[i])
        i +=1
if i >=len(list1):
    res.extend(list2[j:len(list2)])
if j >=len(list2):
    res.extend(list1[i:len(list1)])
print(res)

快速排序:
def quick_sort(list,start,end):
    if start < end:

        provit = paixu(list,start,end)
        quick_sort(list,start,provit-1)
        quick_sort(list,provit+1,end)

def paixu(list,start,end):
    i = start
    j =  end
    tmp  = list[i]
    while i<j:
        while i <j and list[j] > tmp:
            j -=1
        if i <j:
            list[i] = list[j]
            i +=1
        while i <j and list[i] <tmp:
            i +=1
        if i<j:
            list[j] = list[i]
            j -=1
        res[i] = tmp
        return i

if __name__ == "__main__":
    res = [9,8,7,6,5,4,3,2,1]
    quick_sort(res,0,len(res)-1)
    print(res)
















