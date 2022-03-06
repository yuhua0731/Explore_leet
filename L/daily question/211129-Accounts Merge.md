Given a list of `accounts` where each element `accounts[i]` is a list of strings, where the first element `accounts[i][0]` is a name, and the rest of the elements are **emails** representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails **in sorted order**. The accounts themselves can be returned in **any order**.

 

**Example 1:**

```
Input: accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Output: [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
Explanation:
The first and second John's are the same person as they have the common email "johnsmith@mail.com".
The third John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
```

**Example 2:**

```
Input: accounts = [["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"],["Ethan","Ethan5@m.co","Ethan4@m.co","Ethan0@m.co"],["Hanzo","Hanzo3@m.co","Hanzo1@m.co","Hanzo0@m.co"],["Fern","Fern5@m.co","Fern1@m.co","Fern0@m.co"]]
Output: [["Ethan","Ethan0@m.co","Ethan4@m.co","Ethan5@m.co"],["Gabe","Gabe0@m.co","Gabe1@m.co","Gabe3@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co","Hanzo3@m.co"],["Kevin","Kevin0@m.co","Kevin3@m.co","Kevin5@m.co"],["Fern","Fern0@m.co","Fern1@m.co","Fern5@m.co"]]
```

 

**Constraints:**

- `1 <= accounts.length <= 1000`
- `2 <= accounts[i].length <= 10`
- `1 <= accounts[i][j] <= 30`
- `accounts[i][0]` consists of English letters.
- `accounts[i][j] (for j > 0)` is a valid email.

#### First approach:

==Union-find==

1. Define edge and node: node i = accounts[i], while an edge [x, y] indicates that node x and node y share a same email address.
2. traverse on accounts, generating a dict that use email address as key and account list as value.
3. if an email address maps to more than one account, then we should add edges to connect those accounts together.
4. with union find, we now have a list representing root account idx for each account.
5. Finally, we merge all emails into a list for each root account. And return this answer.

```python
def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
    # union find
    # first: we need to find hiding edges from input
    # if an email exists in both account, then there should be an edge between these two accounts
    n = len(accounts)
    graph = [i for i in range(n)] # union find root idx list

    def find(x):
        if graph[x] != x: graph[x] = find(graph[x])
        return graph[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x != root_y: graph[root_x] = root_y

    email_account = dict() # key = email | value = account list
    for idx, email in enumerate(accounts):
        for e in email[1:]:
            if e in email_account: email_account[e].append(idx)
            else: email_account[e] = [idx]
    for value in email_account.values():
        # if there are more than 1 account share the same email, connect these accounts
        # value = [0, 1, 2], zip(value, value[1:]) = [[0, 1], [1, 2]]
        if len(value) > 1: [union(i, j) for i, j in zip(value, value[1:])]

    root_email = collections.defaultdict(set) # key = root account index | value: a set of all emails
    for account_idx, root in enumerate(graph):
        root_email[find(root)].update(accounts[account_idx][1:])

    return [accounts[root_idx][0:1] + list(sorted(email_list)) for root_idx, email_list in root_email.items()]
	# accounts[root_idx][0:1] return a list, while accounts[root_idx][0] return an element, which is an int.
```

