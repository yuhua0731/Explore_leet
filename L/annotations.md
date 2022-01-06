# annotations

Annotations work like this:

```python
@annotations_name
def func_1():
	pass
==>> annotations_name(func_1)
```

## @property in class

```python
class Warrior:
    def __init__(self, attack = 5):
        self.health = 50
        self.attack = attack
    
    @property
    def is_alive_prop(self):
        return self.health > 0
    
    def is_alive_func(self):
      	return self.health > 0

if __name__ == '__main__':
    chuck = Warrior()
    chuck.is_alive_prop # correct, <class 'bool'>
    chuck.is_alive_prop() # TypeError: 'bool' object is not callable
    check.is_alive_func # correct, <class 'method'>
    chuck.is_alive_func() # correct, <class 'bool'>
    
```

`@property` annotation: when you add this annotation ahead of a function, you can call it without append brackets to it. For instance, `Alan.is_alive` is acceptable. It is equivalent to `foo = property(foo)`. It return a property object instead of function object, which requires arguments.

## @enum.unique

- `@enum.unique` Enum members must have unique names while their values can be duplicate, unless you add an annotation ahead of Enum class like the following:

```python
from enum import Enum, unique
@unique
class Mistake(Enum):
    ONE = 1
    TWO = 2
    THREE = 2
```

## @functools.cache

creating a thin wrapper around a ==dictionary== lookup for the function ==arguments==.

```python
@cache
def factorial(n):
    return factorial(n - 1) if n else 1
# no previously cached result, makes 11 recursive calls
factorial(10)
# just looks up cached value result
factorial(5)
# makes two new recursive calls, the other 10 are cached
factorial(12)
```