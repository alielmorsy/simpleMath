# How the Exponential Function Works in SIMD

This document explains how my implementation of the exp function works. First, I would like to thank the maintainers of
SLEEF. I could not have managed this without their implementation. I attempted to make my version somewhat different
from
theirs, but I am not sure if I succeeded.


## Step 1

We want to compute $e^x$ efficiently. However, polynomial approximations are accurate only for small intervals (
especially low-degree polynomials).  
To handle any $x$, we need to **split it into a small remainder $r$ and an integer multiple of $\ln 2$**, so we can
apply a polynomial approximation to $r$.

We use the identity:  
$$
e^{\ln 2} = 2
$$

We can split $x$ into:
$$
x = q \cdot \ln 2 + r
$$

where $q$ is an integer chosen to minimize $|r|$.  
We can compute $q$ as:

$$
q = \text{round}\left(\frac{x}{\ln 2}\right)
$$

Then the remainder is:

$$
r = x - q \cdot \ln 2
$$

Because we chose $q$ as the nearest integer to $x / \ln 2$, the remainder is bounded by:

$$
|r| \le \frac{\ln 2}{2}
$$

This ensures $r$ is small enough for a **polynomial approximation** of $e^r$ to be accurate.

Finally, we can rewrite the exponential as:

$$
e^x = e^{q \cdot \ln 2 + r} = e^{q \cdot \ln 2} \cdot e^r = 2^q \cdot e^r
$$

Here, $2^q$ can be computed efficiently, and $e^r$ is approximated using a polynomial.



> While implementing, I tried truncating and rounding to the nearest and somehow rounding gave me better results, but
> that was at early testing while writing maybe there was a bug back then. But, it worth testing for sure

## Step 2

> Note: This step will assume we work with doubles as it will be easier to explain and everyone knows how double stored
> in memory.

From the last step we got this:
$$
e^x = 2^q \cdot e^r
$$

In this step, we are required to calculate the $2^q$ part. The first option that came to my mind was to use my
implementation of `sm_powi_ps` (a SIMD-powered integer exponentiation function). However, after thinking more deeply, I
realized this would be a terrible idea — especially when $q$ can be as large as 32 or even higher. A loop-based
exponentiation (like repeated squaring or multiplication) would introduce unnecessary branches, latency, and worst of
all, **poor performance in vectorized code**.

Then I remembered a clever bit-level trick that works **only for powers of 2 with integer exponents** — and it's blazing
fast. It relies on the **IEEE 754 double-precision floating-point format**.

### Why This Trick Works

We know that $2^q$ is a power of two, and in IEEE 754, such numbers have a very special bit pattern:

- **Sign bit**: 0 (positive)
- **Exponent field**: $1023 + q$ (because exponent is biased by 1023)
- **Mantissa (fraction)**: all zeros (since $2^q$ is exactly representable as $1.0 \times 2^q$)

So the entire double-precision value of $2^q$ can be constructed by setting the exponent to $1023 + q$, and leaving the
mantissa zero.

Now, the mantissa is 52 bits long. That means **shifting a number by 52 bits moves it into the exponent field** if we
treat the value as a 64-bit integer.

## Step 3

Now the first part $2^q$ of our equation $e^r = 2^q \cdot e^r$ is done. It's time to move to the other part $e^r$.

We made sure from Step 1 that $r$ is very small and it's bounded to  $ln2/2$. That means our range was reduced to be
$$
r \in \left[-\frac{\ln 2}{2}, \frac{\ln 2}{2}\right]
$$

We will use **Horner's method** as it efficient and approximates fast. This method is an optimized version of Taylor's
series. That's how taylor series would look like
$$
e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \dots
$$

We can use Horner's method to optimize it by doing:

$$
e^x \approx 1 + x \Big( 1 + x \Big( \frac{1}{2} + x \Big( \frac{1}{6} + x \Big( \frac{1}{24} + x \frac{1}{120} \Big) \Big) \Big) \Big)
$$

That's a good starting point, but here we have an issue taylor series coefficients in general do not behave well in
floating points. That's why our second optimization comes in **minimax polynomial**.

### Minimax Polynomial

A minimax polynomial is a method to find the best possible polynomial of a given degree that approximates a function
over a specific interval. “Best” here means it minimizes the largest error anywhere in that interval, so it’s as
accurate as possible while staying efficient.

> I feel like that's a bad explanation to it but I did't go deep isnide that algorith.

--- 
I used sollya to generate coefficients using this expression

```
fpminimax((exp(x)-1-x)/x^2, degree, [|double,double,double,double,double,double,double,double,double,double|], [-log(2)/2, log(2)/2]);
```

I used `degree = 5` for floats and `degree = 9` for doubles.

## Step 4

Now we have both parts calculated in our expression
$$
e^x = 2^q \cdot e^r
$$

We can do a simple float multiply and return the result and voilà! The exponent is calculated



> Something you may not care about, but I have to write it somewhere. That took me more than 12hour to understand and
> implement. A friend even told me to get a life

