1.二维数组的查找
解法一：暴力解法
解题思路：挨个遍历，直到找到符合条件的值
public class Solution {
    public boolean Find(int target, int [][] array) {
        int row = array.length;
        int col = array[0].length;
        for (int i =0;i<col;i++){
            for(int j =0;j<row;j++){
                if(array[i][j]==target){
                    return true;
                }
            }
        }
        return false;
    }
}

解法二：
public class Solution {
    public boolean Find(int target, int [][] array) {
        int row = array.length;
        int col = array[0].length;
        int i =row -1;
        int j = 0;
        while(i>=0 &&j<col){
            if(array[i][j]>target){    //如果当前值比目标值大，说明目标值在上边的行中，
                i--;
            }else if(array[i][j]<target){//如果当前值比目标值小，说明目标值要么在本行中
                j++;
            }else{
                return true; 
            }
        }
        return false;
    }
}

2.替换空格
解法一：利用字符串函数的replaceAll函数，再利用正则表达式进行匹配

public class Solution {
    public String replaceSpace(StringBuffer str) {
    	return str.toString().replaceAll("\\s","%20");
    }
}

解法二：利用字符串函数中的replace函数，将空格替换为相应的字符
public class Solution {
    public String replaceSpace(StringBuffer str) {
    	return str.toString().replace(" ","%20");
    }
}

3.从尾到头打印链表：输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
方法一:
/**
*    public class ListNode {
*        int val;
*        ListNode next = null;
*
*        ListNode(int val) {
*            this.val = val;
*        }
*    }
*
*/
import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        ArrayList<Integer> list= new ArrayList<Integer>();
        while(listNode!=null){
            list.add(0, .val);
            listNode = listNode.next;
        }
        return list;
    }
}

方法二:利用ArrayList中的addFirst函数或者addLast函数





合并两个排序的链表:
解法一:创建一个新的结点的方法
/*
public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}*/
public class Solution {
    public ListNode Merge(ListNode list1,ListNode list2) {
        if(list1 ==null){
            return list2;
        }
        if(list2 == null){
            return list1;
        }
        ListNode head = new ListNode(-1);
        ListNode p = head;
        while(list1 !=null && list2 !=null){
            if(list1.val > list2.val){
                p.next = list2;
                list2 = list2.next;
            }else{
                p.next = list1;
                list1 = list1.next;
            }
            p = p.next;
        }
        if(list1 !=null){
            p.next = list1;
        }
        if(list2 !=null){
            p.next =list2;
        }
        return head.next;
    }
}

解法二:不创建新的结点方法   略

4.重建二叉树

解题思路：关键在于Arrays.copyOfRange函数的运用

import java.util.Arrays;

/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        if(pre.length ==0 && in.length ==0){
            return null;
        }
        TreeNode root = new TreeNode(pre[0]);
        for(int i =0;i<in.length;i++){
            if(root.val == in[i]){
                root.left = reConstructBinaryTree(Arrays.copyOfRange(pre,1,i+1),Arrays.copyOfRange(in,0,i));
                root.right = reConstructBinaryTree(Arrays.copyOfRange(pre,i+1,pre.length),Arrays.copyOfRange(in,i+1,in.length));
                break;
            }
            
        }
        return root;
    }
}

5.用两个栈实现队列
解题思路：
import java.util.Stack;

public class Solution {
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    
    public void push(int node) {
        stack1.push(node);
    }
    
    public int pop() {
        if(stack2.size()<=0){
            while(stack1.size()!=0){
                stack2.push(stack1.pop());
            }
        }
        return stack2.pop(); 
    }
}

6.旋转数组的最小数字
解法一：暴力解法
直接遍历数组，找出来前一个比后一个大的即为要查找的数字
import java.util.ArrayList;
public class Solution {
    public int minNumberInRotateArray(int [] array) {
        for(int i =0;i<array.length;i++){
            for(int j= i+1;j<array.length;j++){
                if(array[i]>array[j]){
                    return array[j];
                }
            }        
        }
        return 0;
    }
}

解法二：二分查找法
import java.util.ArrayList;
public class Solution {
    public int minNumberInRotateArray(int [] array) {
        int i=0,j=array.length-1;
        while(i<j){
            if(array[i]<array[j]){
                return array[i];
            }
            int mid = (i+j)/2;
            if(array[mid]>array[i]){
                i =mid+1;
            }else if(array[mid]<array[j]){
                j = mid;
            }else{
                i++;
            }
        }
        return array[i];
        
    }
}

7.斐波那契数列：大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0，第1项是1）。n<=39
解题思路：
解法一：利用一个ArrayList数组，循环相加即可得到
import java.util.ArrayList;
public class Solution {
    public int Fibonacci(int n) {
        int i = 0,j=1,count=2;
        ArrayList<Integer> list = new ArrayList<Integer>();
        list.add(0);
        list.add(1);
        if(n<2){
            return n;
        }
        while(count <=n){
            list.add(list.get(count-1)+list.get(count-2));
            count ++;
        }
        return list.get(n);
    }
}

解法二：利用递归方法
public class Solution {
    public int Fibonacci(int n) {
        if(n<2){
            return n;
        }
        return Fibonacci(n-1)+Fibonacci(n-2);
    }
}

解法三：利用一个固定长度的数组

public class Solution {
    public int Fibonacci(int n) {
        if(n<2){
            return n;
        }
        int [] list = new int[40];
        list[0]=0;
        list[1]=1;
        for(int i=2;i<=n;i++){
            list[i] = list[i-1]+list[i-2];
        }
        return list[n];
    }
}


8.跳台阶：
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
解法一：利用递归方法
public class Solution {
    public int Fibonacci(int n) {
        if(n<2){
            return n;
        }
        return Fibonacci(n-2)+Fibonacci(n-1);
    }
}

解法二：利用ArrayList方法

import java.util.ArrayList;
public class Solution {
    public int JumpFloor(int target) {
        ArrayList<Integer> list = new ArrayList<Integer>();
       if(target <3){
           return target;
       }
        list.add(0);
        list.add(1);
        list.add(2);
        for(int count =3;count <=target;count++){
            list.add(list.get(count-1)+list.get(count-2));
        }
        return list.get(target);
    }
}

9.变态跳台阶:一只青蛙一次可以跳上1级台阶,也可以跳上2级,.....它也可以跳上n级,求该青蛙跳到N级台阶总共有多少中跳法
解题思路:青蛙跳到一阶台阶,共1种跳法;跳到两阶台阶,共2种跳法,跳到3阶台阶,共8种跳法

解法一:借用ArrayList方法
import java.util.ArrayList;
public class Solution {
    public int JumpFloorII(int target){
        ArrayList<Integer> list = new ArrayList<Integer>();
        if(target<3){
            return target;
        }
        list.add(0);
        list.add(1);
        list.add(2);
        for(int count =3;count<=target;count++){
            list.add(list.get(count-1) *2);
        }
        return list.get(target);
    }
}

10.矩形覆盖
解题思路:本质上还是斐波那契数列,先找规律,再写代码

解法一:递归调用法
public class Solution {
    public int RectCover(int target) {
        if(target<=2){
            return target;
        }
        return RectCover(target-2) + RectCover(target-1);
    }
}

解法二:利用ArrayList完成


import java.util.ArrayList;
public class Solution {
    public int RectCover(int target) {
        if(target < 3){
            return target;
        }
        ArrayList<Integer> list = new ArrayList<Integer>();
        list.add(0);
        list.add(1);
        list.add(2);
        for(int i = 3;i<=target;i++){
            list.add(list.get(i-2) + list.get(i-1));
        }
        return list.get(target);
    }
}