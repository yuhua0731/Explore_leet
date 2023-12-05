## Master Theorem

If `a ≥ 1` and `b > 1` are constants and `f(n)` is an asymptotically positive function, then the time complexity of a recursive relation is given by

```
T(n) = aT(n/b) + f(n)

where, T(n) has the following asymptotic bounds:
    1. If f(n) = O(nlogb a-ϵ), then T(n) = Θ(nlogb a).
    2. If f(n) = Θ(nlogb a), then T(n) = Θ(nlogb a * log n).
    3. If f(n) = Ω(nlogb a+ϵ), then T(n) = Θ(f(n)).

ϵ > 0 is a constant.
```



```
T(n) = 3T(n/2) + n^2
Here, a = 3, n/b = n/2, f(n) = n^2

logb a = log2 3 ≈ 1.58 < 2
ie. f(n) < nlogb a+ϵ , where, ϵ is a constant.

Case 3 implies here. Thus, T(n) = f(n) = Θ(n2) 
```

> ###### Master Theorem Limitations
>
> The master theorem cannot be used if:
>
> - T(n) is not monotone. eg. `T(n) = sin n`
> - `f(n)` is not a polynomial. eg. `f(n) = 2n`
> - a is not a constant. eg. `a = 2^n`
> - `a < 1`