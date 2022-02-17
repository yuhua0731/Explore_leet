# Strategic Approach to DP

## Framework for DP Problems

------

Now that we understand the basics of DP and how to spot when DP is applicable to a problem, we've reached the most important part: actually solving the problem. In this section, we're going to talk about a framework for solving DP problems. This framework is applicable to nearly every DP problem and provides a clear step-by-step approach to developing DP algorithms.

> For this article's explanation, we're going to use the problem [Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) as an example, with a top-down (recursive) implementation. Take a moment to read the problem description and understand what the problem is asking.

Before we start, we need to first define a term: **==state==**. In a DP problem, a **state** is a set of variables that can sufficiently describe a scenario. These variables are called **==state variables==**, and we only care about relevant ones. For example, to describe every scenario in Climbing Stairs, there is only 1 relevant state variable, the current step we are on. We can denote this with an integer i. If i = 6, that means that we are describing the state of being on the 6th step. Every unique value of i represents a unique **state**.

> You might be wondering what "relevant" means here. Picture this problem in real life: you are on a set of stairs, and you want to know how many ways there are to climb to say, the 10th step. We're definitely interested in what step you're currently standing on. However, we aren't interested in what color your socks are. You could certainly include sock color as a state variable. Standing on the 8th step wearing green socks is a different state than standing on the 8th step wearing red socks. However, changing the color of your socks will not change the number of ways to reach the 10th step from your current position. Thus the color of your socks is an **irrelevant** variable. In terms of figuring out how many ways there are to climb the set of stairs, the only **relevant** variable is what stair you are currently on.



### The Framework

To solve a DP problem, we need to combine 3 things:

1. **A function or data structure that will compute/contain the answer to the problem for every given state**.

   For Climbing Stairs, let's say we have an function dp where dp(i) returns the number of ways to climb to the *i**t**h* step. Solving the original problem would be as easy as return dp(n).

   How did we decide on the design of the function? The problem is asking "How many distinct ways can you climb to the top?", so we decide that the function will represent how many distinct ways you can climb to a certain step - literally the original problem, but generalized for a given state.

   > Typically, top-down is implemented with a recursive function and hash map, whereas bottom-up is implemented with nested for loops and an array. When designing this function or array, we also need to decide on state variables to pass as arguments. This problem is very simple, so all we need to describe a state is to know what step we are currently on i. We'll see later that other problems have more complex states.

2. **A recurrence relation to transition between states.**

   A recurrence relation is an equation that relates different states with each other. Let's say that we needed to find how many ways we can climb to the 30th stair. Well, the problem states that we are allowed to take either 1 or 2 steps at a time. Logically, that means to climb to the 30th stair, we arrived from either the 28th or 29th stair. Therefore, the number of ways we can climb to the 30th stair is equal to the number of ways we can climb to the 28th stair plus the number of ways we can climb to the 29th stair.

   The problem is, we don't know how many ways there are to climb to the 28th or 29th stair. However, we can use the logic from above to define a recurrence relation. In this case, dp(i) = dp(i - 1) + dp(i - 2). As you can see, information about some states gives us information about other states.

   > Upon careful inspection, we can see that this problem is actually the Fibonacci sequence in disguise! This is a very simple recurrence relation - typically, finding the recurrence relation is the most difficult part of solving a DP problem. We'll see later how some recurrence relations are much more complicated, and talk through how to derive them.

3. **Base cases, so that our recurrence relation doesn't go on infinitely.**

   The equation dp(i) = dp(i - 1) + dp(i - 2) on its own will continue forever to negative infinity. We need base cases so that the function will eventually return an actual number.

   Finding the base cases is often the easiest part of solving a DP problem, and just involves a little bit of logical thinking. When coming up with the base case(s) ask yourself: What state(s) can I find the answer to without using dynamic programming? In this example, we can reason that there is only 1 way to climb to the first stair (1 step once), and there are 2 ways to climb to the second stair (1 step twice and 2 steps once). Therefore, our base cases are dp(1) = 1 and dp(2) = 2.

   > We said above that we don't know how many ways there are to climb to the 28th and 29th stairs. However, using these base cases and the recurrence relation from step 2, we can figure out how many ways there are to climb to the 3rd stair. With that information, we can find out how many ways there are to climb to the 4th stair, and so on. Eventually, we will know how many ways there are to climb to the 28th and 29th stairs.



### Example Implementations

Here is a basic top-down implementation using the 3 components from the framework:

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        def dp(i): 
            """A function that returns the answer to the problem for a given state."""
            # Base cases: when i is less than 3 there are i ways to reach the ith stair.
            if i <= 2: 
                return i
            
            # If i is not a base case, then use the recurrence relation.
            return dp(i - 1) + dp(i - 2)
        
        return dp(n)
```

Do you notice something missing from the code? We haven't memoized anything! The code above has a time complexity of *O*(2*n*) because every call to dp creates 2 more calls to dp. If we wanted to find how many ways there are to climb to the 250th step, the number of operations we would have to do is approximately equal to the number of atoms in the universe.

In fact, without the memoization, this isn't actually dynamic programming - it's just basic recursion. Only after we optimize our solution by adding memoization to avoid repeated computations can it be called DP. As explained in chapter 1, memoization means caching results from function calls and then referring to those results in the future instead of recalculating them. This is usually done with a hashmap or an array.

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        def dp(i):
            if i <= 2: 
                return i
            if i not in memo:
                # Instead of just returning dp(i - 1) + dp(i - 2), calculate it once and then
                # store the result inside a hashmap to refer to in the future.
                memo[i] = dp(i - 1) + dp(i - 2)
            
            return memo[i]
        
        memo = {}
        return dp(n)
```

With memoization, our time complexity drops to *O*(*n*) - astronomically better, literally.

> You may notice that a hashmap is overkill for caching here, and an array can be used instead. This is true, but using a hashmap isn't necessarily bad practice as some DP problems will require one, and they're hassle-free to use as you don't need to worry about sizing an array correctly. Furthermore, when using top-down DP, some problems do not require us to solve every single subproblem, in which case an array may use more memory than a hashmap.

We just talked a whole lot about top-down, but what about bottom-up? Everything is pretty much the same, except we will start from our base cases and iterate up to our final answer. As stated before, bottom-up implementations usually use an array, so we will use an array dp where dp[i] represents the number of ways to climb to the *ith* step.

> Notice that the implementation still follows the framework exactly - the framework holds for both top-down and bottom-up implementations.



### To Summarize

With DP problems, we can use logical thinking to find the answer to the original problem for certain inputs, in this case we reason that there is 1 way to climb to the first stair and 2 ways to climb to the second stair. We can then use a recurrence relation to find the answer to the original problem for any state, in this case for any stair number. Finding the recurrence relation involves thinking about how moving from one state to another changes the answer to the problem.

This is the essence of dynamic programming. Here's a quick animation for Climbing Stairs:



### Next Up

For the rest of the explore card, we're going to use the framework to solve multiple examples, while explaining the thought process behind how to apply the framework at each step. It may be useful to refer back to the section of this article titled "The framework" as you move along the card. For now, take a deep breath - this was a lot to take in, but soon you will be equipped to start solving DP problems on your own.



TODO:

## Example 198. House Robber

------

> This is the first of 6 articles where we will use a framework to work through example DP problems. The framework provides a blueprint to solve DP problems, but when you are just starting to learn DP, deriving some of the logic yourself may be difficult. The objective of these articles is to talk through how to use the framework to work through each problem, and our goal is that, by the end of this, you will be able to independently tackle most DP problems using this framework.

In this article, we will be looking at the [House Robber](https://leetcode.com/problems/house-robber/) problem. In an earlier section of this explore card, we talked about how House Robber fits the characteristics of a DP problem. It's asking for the maximum of something, and our current decisions will affect which options are available for our future decisions. Let's see how we can use the framework to develop an algorithm for this problem.

1. A **function or array** that answers the problem for a given state

First, we need to decide on state variables. As a reminder, state variables should be fully capable of describing a scenario. Imagine if you had this scenario in real life - you're a robber and you have a lineup of houses. If you are at one of the houses, the only variable you would need to describe your situation is an integer - the index of the house you are currently at. Therefore, the only state variable is an integer, say \text{i}i, that indicates the index of a house.

> If the problem had an added constraint such as "*you are only allowed to rob up to k houses*", then \text{k}k would be another necessary state variable. This is because being at, say house 4 with 3 robberies left is different than being at house 4 with 5 robberies left.
>
> You may be wondering - why don't we include a state variable that is a boolean indicating if we robbed the previous house or not? We certainly could include this state variable, but we can develop our recurrence relation in a way that makes it unnecessary. Building an intuition for this is difficult at first, but it becomes easier with practice.

The problem is asking for "the maximum amount of money you can rob". Therefore, we would use either a function \text{dp(i)}dp(i) that returns the maximum amount of money you can rob up to and including house \text{i}i, or an array \text{dp}dp where \text{dp[i]}dp[i] represents the maximum amount of money you can rob up to and including house \text{i}i.

This means that after all the subproblems have been solved, \text{dp[i]}dp[i] and \text{dp(i)}dp(i) both return the answer to the original problem for the subarray of \text{nums}nums that spans 00 to \text{i}i inclusive. To solve the original problem, we will just need to return \text{dp[nums.length - 1]}dp[nums.length - 1] or \text{dp(nums.length - 1)}dp(nums.length - 1), depending if we do bottom-up or top-down.

2. A **recurrence relation** to transition between states

> For this part, let's assume we are using a top-down (recursive function) approach. Note that the top-down approach is closer to our natural way of thinking and it is generally easier to think of the recurrence relation if we start with a top-down approach.

Next, we need to find a recurrence relation, which is typically the hardest part of the problem. For any recurrence relation, a good place to start is to think about a general state (in this case, let's say we're at the house at index \text{i}i), and use information from the problem description to think about how other states relate to the current one.

If we are at some house, logically, we have 2 options: we can choose to rob this house, or we can choose to not rob this house.

1. If we decide not to rob the house, then we don't gain any money. Whatever money we had from the previous house is how much money we will have at this house - which is \text{dp(i - 1)}dp(i - 1).
2. If we decide to rob the house, then we gain \text{nums[i]}nums[i] money. However, this is only possible if we did not rob the previous house. This means the money we had when arriving at this house is the money we had from the previous house without robbing it, which would be however much money we had 2 houses ago, \text{dp(i - 2)}dp(i - 2). After robbing the current house, we will have \text{dp(i - 2) + nums[i]}dp(i - 2) + nums[i] money.

From these two options, we always want to pick the one that gives us maximum profits. Putting it together, we have our recurrence relation: \text{dp(i)} = \max(\text{dp(i - 1), dp(i - 2) + nums[i]})dp(i)=max(dp(i - 1), dp(i - 2) + nums[i]) .

3. **Base cases**

The last thing we need is base cases so that our recurrence relation knows when to stop. The base cases are often found from clues in the problem description or found using logical thinking. In this problem, if there is only one house, then the most money we can make is by robbing the house (the alternative is to not rob the house). If there are only two houses, then the most money we can make is by robbing the house with more money (since we have to choose between them). Therefore, our base cases are:

1. \text{dp(0) = nums[0]}dp(0) = nums[0]
2. \text{dp(1)} = \max( \text{nums[0], nums[1]})dp(1)=max(nums[0], nums[1])



### Top-down Implementation

Now that we have established all 3 parts of the framework, let's put it together for the final result. Remember: we need to memoize the function!

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        def dp(i):
            # Base cases
            if i == 0: 
                return nums[0]            
            if i == 1: 
                return max(nums[0], nums[1])            
            if i not in memo:
                memo[i] = max(dp(i - 1), dp(i - 2) + nums[i]) # Recurrence relation
            return memo[i]
        
        memo = {}
        return dp(len(nums) - 1)
```



### Bottom-up Implementation

Here's the bottom-up approach: everything is the same, except that we use an array instead of a hash map and we iterate using a for-loop instead of using recursion.

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1: 
            return nums[0]
        
        dp = [0] * len(nums)
        
        # Base cases
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]) # Recurrence relation
        
        return dp[-1]
```

For both implementations, the time and space complexity is O(n)*O*(*n*). We'll talk about time and space complexity of DP algorithms in depth at the end of this chapter. Here's an animation that shows the algorithm in action:



### Up Next

Now that you've seen the framework in action, try solving these problems (located on the next 2 pages) on your own. If you get stuck, come back here for hints:

**746. Min Cost Climbing Stairs**



<details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding state variables and<span>&nbsp;</span><code style="box-sizing: border-box; font-family: SFMono-Regular, Consolas, &quot;Liberation Mono&quot;, Menlo, Courier, monospace; font-size: 1em; padding: 2px 4px; color: rgb(199, 37, 78); background-color: rgb(249, 242, 244); border-radius: 4px;">dp</code></summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Let<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{dp(i)}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">dp(i)</span></span></span></span></span></span><span>&nbsp;</span>be the minimum cost necessary to reach step<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.66786em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">i</span></span></span></span></span></span>.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding the recurrence relation</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">We can arrive at step<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.66786em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">i</span></span></span></span></span></span><span>&nbsp;</span>from either step<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{i - 1}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.66786em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">i&nbsp;-&nbsp;1</span></span></span></span></span></span><span>&nbsp;</span>or step<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{i - 2}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.66786em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">i&nbsp;-&nbsp;2</span></span></span></span></span></span>. Choose whichever one is cheaper.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding base cases</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Since we can start from either step 0 or step 1, the cost to reach these steps is 0.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"><strong style="box-sizing: border-box; font-weight: bolder;">1137. N-th Tribonacci Number</strong></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding state variables and<span>&nbsp;</span><code style="box-sizing: border-box; font-family: SFMono-Regular, Consolas, &quot;Liberation Mono&quot;, Menlo, Courier, monospace; font-size: 1em; padding: 2px 4px; color: rgb(199, 37, 78); background-color: rgb(249, 242, 244); border-radius: 4px;">dp</code></summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Let<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{dp(i)}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">dp(i)</span></span></span></span></span></span><span>&nbsp;</span>represent the<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">i^{th}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.849108em; vertical-align: 0em;"></span><span class="mord" style="box-sizing: border-box;"><span class="mord mathdefault" style="box-sizing: border-box; font-family: KaTeX_Math; font-style: italic;">i</span><span class="msupsub" style="box-sizing: border-box; text-align: left;"><span class="vlist-t" style="box-sizing: border-box; display: inline-table; table-layout: fixed;"><span class="vlist-r" style="box-sizing: border-box; display: table-row;"><span class="vlist" style="box-sizing: border-box; display: table-cell; vertical-align: bottom; position: relative; height: 0.849108em;"><span class="" style="box-sizing: border-box; display: block; height: 0px; position: relative; top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="box-sizing: border-box; display: inline-block; overflow: hidden; width: 0px; height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight" style="box-sizing: border-box; display: inline-block; font-size: 0.7em;"><span class="mord mtight" style="box-sizing: border-box;"><span class="mord mathdefault mtight" style="box-sizing: border-box; font-family: KaTeX_Math; font-style: italic;">t</span><span class="mord mathdefault mtight" style="box-sizing: border-box; font-family: KaTeX_Math; font-style: italic;">h</span></span></span></span></span></span></span></span></span></span></span></span></span><span>&nbsp;</span>tribonacci number.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding the recurrence relation</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Use the equation given in the problem description.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding base cases</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Use the base cases given in the problem description.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"><strong style="box-sizing: border-box; font-weight: bolder;">740. Delete and Earn</strong></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding preprocessing steps</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"><strong style="box-sizing: border-box; font-weight: bolder;">Sort</strong><span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{nums}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.43056em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">nums</span></span></span></span></span></span><span>&nbsp;</span>and<span>&nbsp;</span><strong style="box-sizing: border-box; font-weight: bolder;">count</strong><span>&nbsp;</span>how many times each number occurs in<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{nums}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.43056em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">nums</span></span></span></span></span></span>.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding state variables and<span>&nbsp;</span><code style="box-sizing: border-box; font-family: SFMono-Regular, Consolas, &quot;Liberation Mono&quot;, Menlo, Courier, monospace; font-size: 1em; padding: 2px 4px; color: rgb(199, 37, 78); background-color: rgb(249, 242, 244); border-radius: 4px;">dp</code></summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Let<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{dp(i)}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">dp(i)</span></span></span></span></span></span><span>&nbsp;</span>be the maximum number of points you can earn between<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.66786em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">i</span></span></span></span></span></span><span>&nbsp;</span>and the end of the sorted<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{nums}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.43056em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">nums</span></span></span></span></span></span><span>&nbsp;</span>array.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding the recurrence relation</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">When we are at index<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.66786em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">i</span></span></span></span></span></span><span>&nbsp;</span>we have 2 options:</p><ol style="box-sizing: border-box; margin-top: 0px; margin-bottom: 1em;"><li style="box-sizing: border-box; color: rgb(90, 90, 90);">Take all numbers that match<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{nums[i]}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">nums[i]</span></span></span></span></span></span><span>&nbsp;</span>and skip all<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{nums[i] + 1}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">nums[i]&nbsp;+&nbsp;1</span></span></span></span></span></span>.</li><li style="box-sizing: border-box; color: rgb(90, 90, 90);">Do not take<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{nums[i]}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">nums[i]</span></span></span></span></span></span><span>&nbsp;</span>and move to the first occurrence of<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{nums[i] + 1}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">nums[i]&nbsp;+&nbsp;1</span></span></span></span></span></span>.</li></ol><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Choose whichever option yields the most points.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"><strong style="box-sizing: border-box; font-weight: bolder;">Bonus Hint:</strong><span>&nbsp;</span>When is the first option guaranteed to be better than the second option?</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding base cases</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">If we have reached the end of the<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{nums}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.43056em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">nums</span></span></span></span></span></span><span>&nbsp;</span>array<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{(i = nums.length)}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">(i&nbsp;=&nbsp;nums.length)</span></span></span></span></span></span><span>&nbsp;</span>then return<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">0</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.64444em; vertical-align: 0em;"></span><span class="mord" style="box-sizing: border-box;">0</span></span></span></span></span><span>&nbsp;</span>because we cannot gain any more points.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"><br style="box-sizing: border-box;"></p><hr style="box-sizing: content-box; height: 1px; margin: 10px 0px; border: none; overflow: hidden; padding: 0px; background-color: rgb(221, 221, 221);"></details></details></details></details></details></details></details></details></details></details>



## House Robber

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and **it will automatically contact the police if two adjacent houses were broken into on the same night**.

Given an integer array `nums` representing the amount of money of each house, return *the maximum amount of money you can rob tonight **without alerting the police***.

 

**Example 1:**

```
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
```

**Example 2:**

```
Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
```

 

**Constraints:**

- `1 <= nums.length <= 100`
- `0 <= nums[i] <= 400`



## Min Cost Climbing Stairs

You are given an integer array `cost` where `cost[i]` is the cost of `ith` step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index `0`, or the step with index `1`.

Return *the minimum cost to reach the top of the floor*.

 

**Example 1:**

```
Input: cost = [10,15,20]
Output: 15
Explanation: You will start at index 1.
- Pay 15 and climb two steps to reach the top.
The total cost is 15.
```

**Example 2:**

```
Input: cost = [1,100,1,1,1,100,1,1,100,1]
Output: 6
Explanation: You will start at index 0.
- Pay 1 and climb two steps to reach index 2.
- Pay 1 and climb two steps to reach index 4.
- Pay 1 and climb two steps to reach index 6.
- Pay 1 and climb one step to reach index 7.
- Pay 1 and climb two steps to reach index 9.
- Pay 1 and climb one step to reach the top.
The total cost is 6.
```

 

**Constraints:**

- `2 <= cost.length <= 1000`
- `0 <= cost[i] <= 999`

> Hide Hint #1 

Say f[i] is the final cost to climb to the top from step i. Then f[i] = cost[i] + min(f[i+1], f[i+2]).

```python
def minCostClimbingStairs(self, cost: List[int]) -> int:
    # dp
    dp = [0, 0] # pre_2, pre_1
    co = cost[:2]
    for i in range(2, len(cost)):
        dp = [dp[1], min(pre + c for pre, c in zip(dp, co))]
        co = [co[1], cost[i]]
    return min(pre + c for pre, c in zip(dp, co))
```



## N-th Tribonacci Number

The Tribonacci sequence Tn is defined as follows: 

T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.

Given `n`, return the value of Tn.

 

**Example 1:**

```
Input: n = 4
Output: 4
Explanation:
T_3 = 0 + 1 + 1 = 2
T_4 = 1 + 1 + 2 = 4
```

**Example 2:**

```
Input: n = 25
Output: 1389537
```

 

**Constraints:**

- `0 <= n <= 37`
- The answer is guaranteed to fit within a 32-bit integer, ie. `answer <= 2^31 - 1`.

> Hide Hint #1 

Make an array F of length 38, and set F[0] = 0, F[1] = F[2] = 1.

> Hide Hint #2 

Now write a loop where you set F[n+3] = F[n] + F[n+1] + F[n+2], and return F[n].

```python
def tribonacci(self, n: int) -> int:
    ans = [0, 1, 1]
    if n < len(ans):
        return ans[n]
    for _ in range(2, n):
        ans = [ans[1], ans[2], sum(ans)]
    return ans[2]
```



## Delete and Earn

You are given an integer array `nums`. You want to maximize the number of points you get by performing the following operation any number of times:

- Pick any `nums[i]` and delete it to earn `nums[i]` points. Afterwards, you must delete **every** element equal to `nums[i] - 1` and **every** element equal to `nums[i] + 1`.

Return *the **maximum number of points** you can earn by applying the above operation some number of times*.

 

**Example 1:**

```
Input: nums = [3,4,2]
Output: 6
Explanation: You can perform the following operations:
- Delete 4 to earn 4 points. Consequently, 3 is also deleted. nums = [2].
- Delete 2 to earn 2 points. nums = [].
You earn a total of 6 points.
```

**Example 2:**

```
Input: nums = [2,2,3,3,3,4]
Output: 9
Explanation: You can perform the following operations:
- Delete a 3 to earn 3 points. All 2's and 4's are also deleted. nums = [3,3].
- Delete a 3 again to earn 3 points. nums = [3].
- Delete a 3 once more to earn 3 points. nums = [].
You earn a total of 9 points.
```

 

**Constraints:**

- `1 <= nums.length <= 2 * 104`
- `1 <= nums[i] <= 104`

> Hide Hint #1 

If you take a number, you might as well take them all. Keep track of what the value is of the subset of the input with maximum M when you either take or don't take M.

```python
def deleteAndEarn(self, nums: List[int]) -> int:
    dp = [0, 0]  # max if curr is picked, max if curr is not picked
    pre = -1
    count = collections.Counter(nums)
    for k in sorted(count.keys()):
        if k == pre + 1:  # current value is adjacent to the previous one
            dp = [dp[1] + k * count[k], max(dp)]
        else:
            # current value is not adjacent to the previous one
            dp = [max(dp) + k * count[k], max(dp)]
        pre = k
    return max(dp)
```



## Multidimensional DP

------

The dimensions of a DP algorithm refer to the number of state variables used to define each state. So far in this explore card, all the algorithms we have looked at required only one state variable - therefore they are **one-dimensional**. In this section, we're going to talk about problems that require multiple dimensions.

Typically, the more dimensions a DP problem has, the more difficult it is to solve. Two-dimensional problems are common, and sometimes a problem might even require [five dimensions](https://leetcode.com/problems/maximize-grid-happiness/). The good news is, the framework works regardless of the number of dimensions.

The following are common things to look out for in DP problems that require a state variable:

- An index along some input. This is usually used if an input is given as an array or string. This has been the sole state variable for all the problems that we've looked at so far, and it has represented the answer to the problem if the input was considered only up to that index - for example, if the input is \text{nums = [0, 1, 2, 3, 4, 5, 6]}nums = [0, 1, 2, 3, 4, 5, 6], then \text{dp(4)}dp(4) would represent the answer to the problem for the input \text{nums = [0, 1, 2, 3, 4]}nums = [0, 1, 2, 3, 4].
- A second index along some input. Sometimes, you need two index state variables, say \text{i}i and \text{j}j. In some questions, these variables represent the answer to the original problem if you considered the input starting at index \text{i}i and ending at index \text{j}j. Using the same example above, \text{dp(1, 3)}dp(1, 3) would solve the problem for the input \text{nums = [1, 2, 3]}nums = [1, 2, 3], if the original input was \text{[0, 1, 2, 3, 4, 5, 6]}[0, 1, 2, 3, 4, 5, 6].
- Explicit numerical constraints given in the problem. For example, "you are only allowed to complete \text{k}k transactions", or "you are allowed to break up to \text{k}k obstacles", etc.
- Variables that describe statuses in a given state. For example "true if currently holding a key, false if not", "currently holding \text{k}k packages" etc.
- Some sort of data like a tuple or bitmask used to indicate things being "visited" or "used". For example, "\text{bitmask}bitmask is a mask where the i^{th}*i**t**h* bit indicates if the i^{th}*i**t**h* city has been visited". Note that mutable data structures like arrays cannot be used - typically, only immutable data structures like numbers and strings can be hashed, and therefore memoized.

Multi-dimensional problems make us think harder about deciding what our function or array will represent, as well as what the recurrence relation should look like. In the next article, we'll walk through another example using the framework with a 2D DP problem.



## Top-down to Bottom-up

------

As we've said in the previous chapter, **usually** a top-down algorithm is easier to implement than the equivalent bottom-up algorithm. With that being said, it is useful to know how to take a completed top-down algorithm and convert it to bottom-up. There's a number of reasons for this: first, in an interview, if you solve a problem with top-down, you may be asked to rewrite your solution in an iterative manner (using bottom-up) instead. Second, as we mentioned before, bottom-up **usually** is more efficient than top-down in terms of runtime.

**Steps to convert top-down into bottom-up**

1. Start with a completed top-down implementation.
2. Initialize an array \text{dp}dp that is sized according to your state variables. For example, let's say the input to the problem was an array \text{nums}nums and an integer \text{k}k that represents the maximum number of actions allowed. Your array \text{dp}dp would be 2D with one dimension of length \text{nums.length}nums.length and the other of length \text{k}k. The values should be initialized as some default value opposite of what the problem is asking for. For example, if the problem is asking for the maximum of something, set the values to negative infinity. If it is asking for the minimum of something, set the values to infinity.
3. Set your base cases, same as the ones you are using in your top-down function. Recall in House Robber, \text{dp(0) = nums[0]}dp(0) = nums[0] and \text{dp(1) = max(nums[0], nums[1])}dp(1) = max(nums[0], nums[1]). In bottom-up, \text{dp[0] = nums[0]}dp[0] = nums[0] and \text{dp[1] = max(nums[0], nums[1])}dp[1] = max(nums[0], nums[1]).
4. Write a for-loop(s) that iterate over your state variables. If you have multiple state variables, you will need nested for-loops. These loops should **start iterating from the base cases**.
5. Now, each iteration of the inner-most loop represents a given state, and is equivalent to a function call to the same state in top-down. Copy the logic from your function into the for-loop and change the function calls to accessing your array. All \text{dp(...)}dp(...) changes into \text{dp[...]}dp[...].
6. We're done! \text{dp}dp is now an array populated with the answer to the original problem for all possible states. Return the answer to the original problem, by changing \text{return dp(...)}return dp(...) to \text{return dp[...]}return dp[...].

------

Let's try a quick example using the House Robber code from before. Here's a completed top-down solution:

```python
class Solution:
    def rob(self, nums: List[int]) -> int:

        def dp(i: int) -> int:
            # Base cases
            if i == 0:
                return nums[0]
            elif i == 1:
                return max(nums[0], nums[1])
            
            if i not in memo:
                # Use recurrence relation to calculate dp[i].
                memo[i] = max(dp(i - 1), dp(i - 2) + nums[i])
            
            return memo[i]
        
        memo = {}
        return dp(len(nums) - 1)
```

First, we initialize an array \text{dp}dp sized according to our state variables. Our only state variable is \text{i}i which can take \text{n}n values.

```python
class Solution:
    def rob(self, nums: List[int]) -> int:        
        n = len(nums)
        dp = [0] * n
        
        return dp[n - 1]
```

Second, we should set our base cases. \text{dp[0] = nums[0]}dp[0] = nums[0] and \text{dp[1] = max(nums[0], nums[1])}dp[1] = max(nums[0], nums[1]). To avoid index out of bounds, we should also just return \text{nums[0]}nums[0] if theres only one house.

```python
class Solution:
    def rob(self, nums: List[int]) -> int:   
        n = len(nums)
        if n == 1:
            return nums[0]
        dp = [0] * n
        
        #Base Cases
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        return dp[n - 1]
```

Next, write a for-loop to iterate over the state variables, starting from the base cases.

```python
class Solution:
    def rob(self, nums: List[int]) -> int:   
        n = len(nums)
        if n == 1:
            return nums[0]
        dp = [0] * n
        
        #Base Cases
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, n):
            pass
        
        return dp[n - 1]
```

Lastly, copy the recurrence relation over from the top-down solution and put it in the for-loop. Return \text{dp[n - 1]}dp[n - 1].

```python
class Solution:
    def rob(self, nums: List[int]) -> int:   
        n = len(nums)
        if n == 1:
            return nums[0]
        dp = [0] * n
        
        #Base Cases
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        
        for i in range(2, n):
            # Use recurrence relation to calculate dp[i].
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        
        return dp[n - 1]
```



## Example 1770. Maximum Score from Performing Multiplication Operations

------

> For this problem, we will again start by looking at a top-down approach.

In this article, we're going to be looking at the problem [Maximum Score from Performing Multiplication Operations](https://leetcode.com/problems/maximum-score-from-performing-multiplication-operations/). We can tell this is a DP problem because it is asking for a maximum score, and every time we choose to use a number from \text{nums}nums, it affects all future possibilities. Let's solve this problem with the framework:

1. A **function or array** that answers the problem for a given state

Since we're doing top-down, we need to decide on two things for our function \text{dp}dp. What state variables we need to pass to it, and what it will return. We are given two input arrays: \text{nums}nums and \text{multipliers}multipliers. The problem says we need to do \text{m}m operations, and on the i^{th}*i**t**h* operation, we gain score equal to \text{multipliers[i]}multipliers[i] times a number from either the left or right end of \text{nums}nums, which we remove after the operation. That means we need to know 3 things for each operation:

1. How many operations have we done so far; this tells us what number from \text{multipliers}multipliers we will be using?
2. The index of the leftmost number remaining in \text{nums}nums.
3. The index of the rightmost number remaining in \text{nums}nums.

We can use one state variable, \text{i}i, to indicate how many operations we have done so far, which means \text{multipliers[i]}multipliers[i] is the current multiplier to be used. For the leftmost number remaining in \text{nums}nums, we can use another state variable, \text{left}left, that indicates how many left operations we have done so far. If we have done, say 3 left operations, if we were to do another left operation we would use \text{nums[3]}nums[3] (because nums is 0-indexed). We can say the same thing for the rightmost remaining number - let's use a state variable \text{right}right that indicates how many right operations we have done so far.

It may seem like we need all 3 of these state variables, but we can formulate an equation for one of them using the other two. If we know how many elements we have picked from the leftside, \text{left}left, and we know how many elements we have picked in total, \text{i}i, then we know that we must have picked \text{i - left}i - left elements from the rightside. The original length of \text{nums}nums is \text{n}n, which means the index of the rightmost element is \text{right = n - 1 - (i - left)}right = n - 1 - (i - left). Therefore, we only need 2 state variables: \text{i}i and \text{left}left, and we can calculate \text{right}right inside the function.

Now that we have our state variables, what should our function return? The problem is asking for the maximum score from some number of operations, so let's have our function \text{dp(i, left)}dp(i, left) return the maximum possible score if we have already done \text{i}i total operations and used \text{left}left numbers from the left side. To answer the original problem, we should return \text{dp(0, 0)}dp(0, 0).



2. A **recurrence relation** to transition between states

At each state, we have to perform an operation. As stated in the problem description, we need to decide whether to take from the left end (\text{nums[left]}nums[left]) or the right end (\text{nums[right]}nums[right]) of the current \text{nums}nums. Then we need to multiply the number we choose by \text{multipliers[i]}multipliers[i], add this value to our score, and finally remove the number we chose from \text{nums}nums. For implementation purposes, "removing" a number from \text{nums}nums means incrementing our state variables \text{i}i and \text{left}left so that they point to the next two left and right numbers.

Let \text{mult} = \text{multipliers[i]}mult=multipliers[i] and \text{right = nums.length - 1 - (i - left)}right = nums.length - 1 - (i - left). The only decision we have to make is whether to take from the left or right of \text{nums}nums.

- If we choose left, we gain \text{mult} \cdot \text{nums[left]}mult⋅nums[left] points from this operation. Then, the next operation will occur at \text{(i + 1, left + 1)}(i + 1, left + 1). \text{i}i gets incremented at every operation because it represents how many operations we have done, and \text{left}left gets incremented because it represents how many left operations we have done. Therefore, our total score is \text{mult} \cdot \text{nums[left] + dp(i + 1, left + 1)}mult⋅nums[left] + dp(i + 1, left + 1).
- If we choose right, we gain \text{mult} \cdot \text{nums[right]}mult⋅nums[right] points from this operation. Then, the next operation will occur at \text{(i + 1, left)}(i + 1, left). Therefore, our total score is \text{mult} \cdot \text{nums[right] + dp(i + 1, left)}mult⋅nums[right] + dp(i + 1, left).

Since we want to maximize our score, we should choose the side that gives more points. This gives us our recurrence relation:

\text{dp(i, left)} = \max(\text{mult} \cdot \text{nums[left]} + \text{dp(i + 1, left + 1)}, \text{ mult} \cdot \text{nums[right]} + \text{dp(i + 1, left)})dp(i, left)=max(mult⋅nums[left]+dp(i + 1, left + 1), mult⋅nums[right]+dp(i + 1, left))

Where \text{mult} \cdot \text{nums[left]} + \text{dp(i + 1, left + 1)}mult⋅nums[left]+dp(i + 1, left + 1) represents the points we gain by taking from the left end of \text{nums}nums plus the maximum points we can get from the remaining \text{nums}nums array and \text{mult} \cdot \text{nums[right]} + \text{dp(i + 1, left)}mult⋅nums[right]+dp(i + 1, left) represents the points we gain by taking from the right end of \text{nums}nums plus the maximum points we can get from the remaining \text{nums}nums array.

3. **Base cases**

The problem statement says that we need to perform \text{m}m operations. When \text{i}i equals \text{m}m, that means we have no operations left. Therefore, we should return \text{0}0.



### Top-down Implementation

Let's put the 3 parts of the framework together for a solution to the problem.

Protip: for Python, the [functools](https://docs.python.org/3/library/functools.html) module provides super handy tools that automatically memoize a function for us. We're going to use the `@lru_cache` decorator in the Python implementation.

> If you find yourself needing to memoize a function in an interview and you're using Python, check with your interviewer if using modules like functools is OK.

This particular problem happens to have very tight time limits. For Java, instead of using a hashmap for the memoization, we will use a 2D array. For Python, we're going to limit our cache size to \text{2000}2000.

```python
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        # lru_cache from functools automatically memoizes the function
        @lru_cache(2000)
        def dp(i, left):
            # Base case
            if i == m:
                return 0

            mult = multipliers[i]
            right = n - 1 - (i - left)
            
            # Recurrence relation
            return max(mult * nums[left] + dp(i + 1, left + 1), 
                       mult * nums[right] + dp(i + 1, left))
                       
        n, m = len(nums), len(multipliers)
        return dp(0, 0)
```



### Bottom-up Implementation

In the bottom-up implementation, the array works the same way as the function from top-down. \text{dp[i][left]}dp[i][left] represents the max score possible if \text{i}i operations have been performed and \text{left}left left operations have been performed.

Earlier in the explore card, we learned that while bottom-up is typically faster than top-down, it is often harder to implement. This is because the order in which we iterate needs to be precise. You'll see in the implementations below that we use the same math to calculate \text{right}right, and the same recurrence relation but we need to iterate backwards starting from \text{m}m (because the base case happens when \text{i}i equals \text{m}m). We also need to initialize \text{dp}dp with one extra row so that we don't go out of bounds in the first iteration of the outer loop.

```python
class Solution:
    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        n, m = len(nums), len(multipliers)
        dp = [[0] * (m + 1) for _ in range(m + 1)]
        
        for i in range(m - 1, -1, -1):
            for left in range(i, -1, -1):
                mult = multipliers[i]
                right = n - 1 - (i - left)
                dp[i][left] = max(mult * nums[left] + dp[i + 1][left + 1], 
                                  mult * nums[right] + dp[i + 1][left])        
        return dp[0][0]
```

The time and space complexity of both implementations is O(m^2)*O*(*m*2) where \text{m}m is the length of \text{multipliers}multipliers. We will talk about more in depth about time and space complexity at the end of this chapter.



### Up Next

Try the next two problems on your own. The first one is a very classical computer science problem and popular in interviews. If you get stuck, come back here for hints:

**1143. Longest Common Subsequence**



<details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding state variables and dp</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Let<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{dp(i, j)}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">dp(i,&nbsp;j)</span></span></span></span></span></span><span>&nbsp;</span>represent the longest common subsequence between the string<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{text1}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.64444em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">text1</span></span></span></span></span></span><span>&nbsp;</span>up to index<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.66786em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">i</span></span></span></span></span></span><span>&nbsp;</span>and<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{text2}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.64444em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">text2</span></span></span></span></span></span><span>&nbsp;</span>up to index<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{j}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.8623em; vertical-align: -0.19444em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">j</span></span></span></span></span></span>.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding the recurrence relation</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">If<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{text1[i] == text2[j]}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">text1[i]&nbsp;==&nbsp;text2[j]</span></span></span></span></span></span>, then we should use this character, giving us<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{1 + dp(i - 1, j - 1)}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">1&nbsp;+&nbsp;dp(i&nbsp;-&nbsp;1,&nbsp;j&nbsp;-&nbsp;1)</span></span></span></span></span></span>. Otherwise, we can either move one character back from<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{text1}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.64444em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">text1</span></span></span></span></span></span>, or 1 character back from<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{text2}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.64444em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">text2</span></span></span></span></span></span>. Try both.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding base cases</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">If<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.66786em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">i</span></span></span></span></span></span><span>&nbsp;</span>or<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{j}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.8623em; vertical-align: -0.19444em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">j</span></span></span></span></span></span><span>&nbsp;</span>becomes less than<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{0}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.64444em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">0</span></span></span></span></span></span>, then we're out of bounds and should<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{return 0}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.64444em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">return&nbsp;0</span></span></span></span></span></span>.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"><strong style="box-sizing: border-box; font-weight: bolder;">221. Maximal Square</strong></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding state variables and dp</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Let<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{dp[row][col]}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">dp[row][col]</span></span></span></span></span></span><span>&nbsp;</span>represent the largest possible square whose bottom right corner is on<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{matrix[row][col]}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 1em; vertical-align: -0.25em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">matrix[row][col]</span></span></span></span></span></span>.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding the recurrence relation</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Any square with a<span>&nbsp;</span><span class="maths katex-rendered" style="box-sizing: border-box;"><span class="katex" style="box-sizing: border-box; font: 1.21em / 1.2 KaTeX_Main, &quot;Times New Roman&quot;, serif; text-indent: 0px; text-rendering: auto;"><span class="katex-mathml" style="box-sizing: border-box; position: absolute; clip: rect(1px, 1px, 1px, 1px); padding: 0px; border: 0px; height: 1px; width: 1px; overflow: hidden;"><math><semantics><annotation encoding="application/x-tex">\text{0}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true" style="box-sizing: border-box;"><span class="base" style="box-sizing: border-box; position: relative; white-space: nowrap; width: min-content; display: inline-block;"><span class="strut" style="box-sizing: border-box; display: inline-block; height: 0.64444em; vertical-align: 0em;"></span><span class="mord text" style="box-sizing: border-box;"><span class="mord" style="box-sizing: border-box;">0</span></span></span></span></span></span><span>&nbsp;</span>cannot have a square on it, and should be ignored. Otherwise, let's say you had a 3x3 square. Look at the bottom right corner of this 3x3 square. What do the squares above, to the left, and to the up-left of this square have in common? All 3 of those squares are the bottom-right square of a square that is (at least) 2x2.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><details open="" style="box-sizing: border-box; display: block;"><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><summary style="box-sizing: border-box; display: list-item; touch-action: manipulation;">Click here to show hint regarding base cases</summary><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);">Just make sure to stay in bounds.</p><p style="box-sizing: border-box; margin: 0px 0px 1em; color: rgb(90, 90, 90);"></p><hr style="box-sizing: content-box; height: 1px; margin: 10px 0px; border: none; overflow: hidden; padding: 0px; background-color: rgb(221, 221, 221);"></details></details></details></details></details></details>



## Maximum Score from Performing Multiplication Operations

You are given two integer arrays `nums` and `multipliers` of size `n` and `m` respectively, where `n >= m`. The arrays are **1-indexed**.

You begin with a score of `0`. You want to perform **exactly** `m` operations. On the `ith` operation **(1-indexed)**, you will:

- Choose one integer `x` from **either the start or the end** of the array `nums`.
- Add `multipliers[i] * x` to your score.
- Remove `x` from the array `nums`.

Return *the **maximum** score after performing* `m` *operations.*

 

**Example 1:**

```
Input: nums = [1,2,3], multipliers = [3,2,1]
Output: 14
Explanation: An optimal solution is as follows:
- Choose from the end, [1,2,3], adding 3 * 3 = 9 to the score.
- Choose from the end, [1,2], adding 2 * 2 = 4 to the score.
- Choose from the end, [1], adding 1 * 1 = 1 to the score.
The total score is 9 + 4 + 1 = 14.
```

**Example 2:**

```
Input: nums = [-5,-3,-3,-2,7,1], multipliers = [-10,-5,3,4,6]
Output: 102
Explanation: An optimal solution is as follows:
- Choose from the start, [-5,-3,-3,-2,7,1], adding -5 * -10 = 50 to the score.
- Choose from the start, [-3,-3,-2,7,1], adding -3 * -5 = 15 to the score.
- Choose from the start, [-3,-2,7,1], adding -3 * 3 = -9 to the score.
- Choose from the end, [-2,7,1], adding 1 * 4 = 4 to the score.
- Choose from the end, [-2,7], adding 7 * 6 = 42 to the score. 
The total score is 50 + 15 - 9 + 4 + 42 = 102.
```

 

**Constraints:**

- `n == nums.length`
- `m == multipliers.length`
- `1 <= m <= 103`
- `m <= n <= 105`
- `-1000 <= nums[i], multipliers[i] <= 1000`

> Hide Hint #1 

At first glance, the solution seems to be greedy, but if you try to greedily take the largest value from the beginning or the end, this will not be optimal.

> Hide Hint #2 

You should try all scenarios but this will be costy.

> Hide Hint #3 

Memoizing the pre-visited states while trying all the possible scenarios will reduce the complexity, and hence dp is a perfect choice here.

```python
# TODO
def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
```



## Longest Common Subsequence

Given two strings `text1` and `text2`, return *the length of their longest **common subsequence**.* If there is no **common subsequence**, return `0`.

A **subsequence** of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

- For example, `"ace"` is a subsequence of `"abcde"`.

A **common subsequence** of two strings is a subsequence that is common to both strings.

 

**Example 1:**

```
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
```

**Example 2:**

```
Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
```

**Example 3:**

```
Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
```

 

**Constraints:**

- `1 <= text1.length, text2.length <= 1000`
- `text1` and `text2` consist of only lowercase English characters.

> Hide Hint #1 

Try dynamic programming. `DP[i][j]` represents the longest common subsequence of text1[0 ... i] & text2[0 ... j].

> Hide Hint #2 

`DP[i][j] = DP[i - 1][j - 1] + 1` , if `text1[i] == text2[j]` `DP[i][j] = max(DP[i - 1][j], DP[i][j - 1])` , otherwise

```python
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = 0
    for i in range(n + 1):
        dp[0][i] = 0

    for i, j in product(range(m), range(n)):
        if text1[i] == text2[j]:
            dp[i + 1][j + 1] = 1 + dp[i][j]
        else:
            dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[-1][-1]
```



## Maximal Square

Given an `m x n` binary `matrix` filled with `0`'s and `1`'s, *find the largest square containing only* `1`'s *and return its area*.

 

**Example 1:**

![img](image_backup/3-Strategic Approach to DP/max1grid.jpg)

```
Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 4
```

**Example 2:**

![img](image_backup/3-Strategic Approach to DP/max2grid.jpg)

```
Input: matrix = [["0","1"],["1","0"]]
Output: 1
```

**Example 3:**

```
Input: matrix = [["0"]]
Output: 0
```

 

**Constraints:**

- `m == matrix.length`
- `n == matrix[i].length`
- `1 <= m, n <= 300`
- `matrix[i][j]` is `'0'` or `'1'`.

```python
def maximalSquare(self, matrix: List[List[str]]) -> int:
    # square, much easier
    # dp
    dp = [int(i) for i in matrix[0]]
    ans = max(dp)
    for row in matrix[1:]:
        temp = [int(row[0])]
        for idx in range(1, len(row)):
            temp.append(0 if row[idx] == '0' else 1 +
                        min([temp[-1], dp[idx - 1], dp[idx]]))
        ans = max(ans, max(temp))
        dp = temp
    return ans ** 2
```



## Time and Space Complexity

------

Finding the time and space complexity of a dynamic programming algorithm may sound like a daunting task. However, this task is usually not as difficult as it sounds. Furthermore, justifying the time and space complexity in an explanation is relatively simple as well. One of the main points with DP is that we never repeat calculations, whether by tabulation or memoization, we only compute a state once. Because of this, the time complexity of a DP algorithm is directly tied to the number of possible states.

If computing each state requires F*F* time, and there are n*n* possible states, then the time complexity of a DP algorithm is O(n \cdot F)*O*(*n*⋅*F*). With all the problems we have looked at so far, computing a state has just been using a recurrence relation equation, which is O(1)*O*(1). Therefore, the time complexity has just been equal to the number of states. To find the number of states, look at each of your state variables, compute the number of values each one can represent, and then multiply all these numbers together.

Let's say we had 3 state variables: \text{i}i, \text{k}k, and \text{holding}holding for some made up problem. \text{i}i is an integer used to keep track of an index for an input array \text{nums}nums, \text{k}k is an integer given in the input which represents the maximum actions we can do, and \text{holding}holding is a boolean variable. What will the time complexity be for a DP algorithm that solves this problem? Let \text{n = nums.length}n = nums.length and \text{K}K be the maximum actions possible given in the input. \text{i}i can be from \text{0}0 to \text{nums.length}nums.length, \text{k}k can be from \text{0}0 to \text{K}K, and \text{holding}holding }can be true or false. Therefore, there are \text{n} \cdot \text{K} \cdot \text{2}n⋅K⋅2 states. If computing each state is O(1)*O*(1), then the time complexity will be O(n \cdot K \cdot 2) = O(n \cdot K)*O*(*n*⋅*K*⋅2)=*O*(*n*⋅*K*).

Whenever we compute a state, we also store it so that we can refer to it in the future. In bottom-up, we tabulate the results, and in top-down, states are memoized. Since we store states, the space complexity is equal to the number of states. That means that in problems where calculating a state is O(1)*O*(1), the time and space complexity are the same. In many DP problems, there are optimizations that can improve both complexities - we'll talk about this later.



> Which of the following require state variables?

- [x] Remaining number of moves allowed
- [ ] Original length of the input
- [x] Current index along the input
- [x] Number of keys currently being held
- [ ] The original number of moves allowed

