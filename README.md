# Haskell

### Type

​	Haskell 定义变量的方式为

```haskell
<var> :: <type> -- variable declaration scheme
a :: Int -- integer
b :: String -- string
c :: Int -> Int -- function
d :: (Int, Int) -- pair
f :: [Int] -- list of integers
```

##### immutability

​	haskell 的变量具有不变性, 赋值后无法进行修改, 但是可以进行二次赋值. 也就是说 `i++` 的操作不被允许, 但是 `i=i+1` 可以.

##### function

​	声明函数的方式

```haskell
foo :: Arg1 -> Arg2 -> ... -> Result
```

即所有的参数类型和返回值类型通过 `->` 符号连接, 这样的函数形式天然对「柯里化」友好.

​	上述只是函数的声明, 如果要使用函数还需要进行定义

```haskell
add :: Int -> Int -> Int
add x y = x + y
```

此外, 函数还可以基于模式匹配的方式进行多次定义, 先定义的模式具有优先匹配权, 例如, 定义斐波那契数列的计算方式

```haskell
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2)
```

上面的写法等价于

```haskell
fib n 
  | n == 0 = 0
  | n == 1 = 1
  | otherwise = fib (n-1) + fib (n-2)
```

haskell 函数可以接受任意 n 个参数, 这里 n 小于函数总参数个数. 当输入的参数少于总参数个数时, 自动进行柯里化, 例如, 

```haskell
foo :: Int -> Int -> Int
foo x y = x + y
bar = foo 1
```

这里 `bar` 是函数签名为 `Int -> Int` 的函数.

​	函数同样可以接受函数作为参数, 只是需要将对应的函数签名用 `()` 括起来.

```haskell
wrapper :: (Int -> Int) -> Int -> Int
```

##### lambda function

​	Haskell 可以通过 `\` 开始定义一个 lambda 函数, 语法为

```haskell
\x y ... -> <body>
```

##### list

​	Haskell 直接用 `[]` 表示 List 类型, List 可以存放任意相同类型的变量.

```haskell
empty_lst = []
a = 1 : [] -- a = [1]
b = 2 : a -- b = [2:1]
c = [1..100] -- a list of integers from 1 to 100.
```

由于 Haskell 变量的不变性, 若需要取出一个元素, 则需要

```haskell
a = [1,2,3]
x:xs = a -- x = 1, xs = [2,3]
```

可以看出, Haskell 的 List 实际上是一个栈.

###### 	获取列表长度

```haskell
len :: [a] -> Int -- a is a generic type
len [] = 0
len (_:tail) = 1 + len tail
```

###### 	获取列表最后一个元素

```haskell
lastElem :: [a] -> a
lastElem [x] = x
lastElem (_:tail) = lastElem tail
```

##### operators as function names

​	Haskell 允许重载运算符, 并且给予了非常大的自由度, 更准确地说, 使用 `()` 包裹任意运算符可充当一个函数名, 例如, 

```haskell
(+-) :: (Int, Int) -> Int -> (Int, Int)
(+-) (x, y) v = (x+v, y-v)
(x, y) = (0,0) +- 1
```

为一个整数tuple, 创建了一个 `+-` 运算符, 将第一个元素加上一个值, 第二个元素减去该值.

​	由于运算符本质只是一个函数名, 这一点与C++的运算符重载并不相同, 所以运算符极易产生命名冲突. 很多运算符已经在标准库 Prelude 中被定义, 所以如果要定义自己的运算符, 应该避免导入 Prelude 中的运算符函数.

#### Data

##### Algebric Data Type (ADT)

- sum type: unions
- product type: tuples
- algebric data type: combination of sum types and product types.

haskell 可以通过 `data` 关键字定义代数类型, 例如 

```haskell
data Maybe a = Nothing | Just a
```

是 haskell 版本 的 `Option` 定义.

```haskell
data List a = Empty | Cons a (List a)

data Tree a = Leaf a | Node (Tree a) a (Tree a)
```

#### Pattern Matching

​	模式匹配在 Haskell 中随处可见, 例如在函数定义中, 参数既可以是字面量, 又可以是参数名, 只是字面量只能匹配变量相符的参数, 而变量名可以匹配任何值的变量.

​	模式匹配还适用于将 ADT 类型进行展开, 并赋予对应的变量名, 以便进行进一步的处理. 例如, 对 `Maybe` 中的值进行一次 map

```haskell
fmap :: (a -> b) -> Maybe a -> Maybe b
fmap _ Nothing = Nothing
-- pattern matching here.
fmap f (Just x) = Just (f x)
```

对于不需要用到的变量可以用 ` _` 忽略, 

​	基于 ADT 的思想, Haskell 很多的复杂类型实际上是简单类型+符号的组合, 都可以通过模式匹配的方式提取出来. 例如, 列表可以通过 `:` 符号进行提取, 规则是“从左至右, 最右为列表剩余部分”, 若提取元素数量超出列表元素数量, 则匹配失败.

##### As-pattern

​	使用模式匹配将变量展开之后, 则原本的变量名将丢失, as-pattern 可以同时保留变量名和进行模式匹配. 例如, 将列表头部元素复制一份,

```haskell
dupHead a :: [a] -> [a]
dupHead [] = []
dupHead l@(x:_) = x:l -- as-pattern
```

这里 `l` 为原始列表变量名, 括号内的为匹配展开的元素, `x` 表示头部元素.

##### case expression

```haskell
f x1 ... xn = case (x1, ..., xn) of 
(p11, ..., p1n) -> e1
...
(pk1, ..., pkn) -> ek
```

`->` 右侧的表达式必须都具有相同的类型.

#### Polymorphism

##### Generics

​	haskell 支持泛型的语法非常简单, 类似于为函数定义一个参数

```haskell
data List a = Empty | Cons a (List a)
lst1 :: List Int -- specialization
lst2 :: List Char
lst3 :: List String
```

​	携带泛型参数的函数

```haskell
filterList :: (t -> Bool) -> List t -> List t
filterList _ Empty = Empty
filterList p (Cons x xs)
  | p x = Cons x (filterList p xs)
  | otherwise = filterList p xs
```

#### Type Class

​	Haskell 的 type class 类似于其它语言中的 interface, 在 type class 的定义中有一组函数, 例如

```haskell
-- Eq 为 type class 名
-- a 为类型参数
class Eq a where 
  -- (==) 重载运算符
  (==) :: a -> a -> Bool
  (/=) :: a -> a -> Bool
```

可以通过 `instance` 关键字为某个类型实现接口, 例如, 

```haskell
data Direction = Left | Right
instance Eq Direction where 
  (==) Left Left = True
  (==) Left Right = False
  (==) Right Left = False
  (==) Right Right = True
```

​	利用 type class 可以为泛型参数提供接口限制, 例如

```haskell
foo :: (Eq a) => a -> a -> String
foo x y = case x == y of 
  True -> "Same Direction"
  False -> "Different Direction"
```

​	有时, 当某个类型实现了某个接口时, 则其代数类型也可以自动实现某个接口. 例如, 当 `a` 实现了 `Eq` 时, 则 `Maybe a` 应该也可以实现 `Eq`,

```haskell
instance Eq a => Eq (Maybe a) where 
  Nothing == Nothing = True
  Just _ == Nothing = False
  Nothing == Just _ = False
  Just x == Just y = x == y
```

#### Lazy Evaluation

​	惰性求值: 任何表达式的结果在「必须使用」之前不会被执行. 所谓必须使用, 即程序必须知道表达式对应的结果才可以继续执行. 例如, 模式匹配中必须知道变量的值才能知道该匹配哪个模式. 

​	虽然模式匹配会推动表达式的求值, 但是只会推动到刚好可以继续进行的程度, 例如, 

```haskell
foo :: a -> Maybe a
foo a = Just a

bar :: Maybe a -> [a]
bar Nothing = []
bar Just x = [x]
```

当调用 `bar (foo 3^500)` 时, 需要进行模式匹配, 因此需要知道对 `foo 3^500` 进行求值, 但是 `x` 并没有被使用, 所以 `3^500` 可以不被执行.

​	正是惰性求值机制的存在, Haskell 可以实现

```haskell
repeat :: a -> [a]
repeat x = x : repeat x

take :: Int -> [a] -> [a]
take n _      | n <= 0 =  []
take _ []              =  []
take n (x:xs)          =  x : take (n-1) xs
```

​	显然, 惰性求值最大的缺点在于程序的执行顺序会被打乱, 这样会带来很多意料之外的结果.

##### pure function

​	纯函数: 只要函数的输入相同, 就会产生相同的输出, 并且不会产生副作用(side effects). 所谓副作用, 是指不会对函数之外的环境产生影响, 例如, 修改全局变量、写入文件等.

​	如果一个表达式是由一组纯函数组成的, 那么其执行结果就不会受到执行顺序的影响.

##### adding strictness

​	lazy evaluation 并总是好的. 将执行延迟到需要用时才开始, 就意味着需要存储所有用到的上下文, 例如, 对一个列表求和, 就需要记住所有的列表元素. 

​	Haskell 允许通过 `seq` 函数对表达式立即求值, 

```haskell
seq :: a -> b -> b
```

其接受两个任意类型的参数, 并总是返回第二个, 但是会通过对第一个参数添加依赖的方式迫使编译器对第一个参数进行立即求值.

##### short-circuiting operators

​	在布尔运算符中 (`&&`, `||`), 若左侧表达式的结果已经满足条件, 则右侧表达式可以不被执行. 这在很多其它语言(C++, Java)中同样如此, 例如

```haskell
(&&) :: Bool -> Bool -> Bool
True  && x = x -- x is evaluated
False && _ = False -- x is ignored
```

#### Monoid

​	Monoid 在数学上指「幺半群」.

​	「半群 (semigroup)」是一种代数结构, 对于集合 $A$, 若存在一种运算 $<>: (A,A) \rightarrow A$, 且满足结合律 $(a <> b) <> c = a <> (b <> c)$. 则称代数结构 $\{<>, A\}$ 是一个半群.

​	幺半群即单位半群, 是一种带单位元的半群. 所谓单位元, 即 $A$ 中存在一个元素 $u$, 满足 $\forall x \in A, u <> x = x, x <> u = x$.

​	幺半群在 Haskell 中的定义为

```haskell
class Monoid m where
  unit :: m -- the unit
  (<>) :: m -> m -> m -- the <> operation
```

幺半群的思想适用于非常多的常见运算, 例如列表操作

```haskell
instance Monoid [a] where
  munit = []
  (<>) x [] = x
  (<>) [] y = y
  (<>) (x:xs) y = x : (xs <> y)
```

#### IO

​	纯函数的最大缺点在于不能修改任何外部环境, 一个完全由纯函数组成的程序可以说没有任何用处, 连打印到终端观察程序运行的状态都无法做到.

​	在 Haskell 中, 将非纯函数统统看作 `IO` 类型. 例如, Haskell 的打印函数为

```haskell
putStrLn :: String -> IO ()
```

传入一个字符串, 返回一个 `IO ()` 类型. 

​	IO 本质是一个可以接受一个泛型参数的 Monad, Monad 的概念在之后才会介绍, 所以先不介绍 IO 的实现.

##### Combining IO

​	IO 操作在 Haskell 是可以叠加的,

```haskell
(>>) :: IO a -> IO b -> IO b
```

从而将两个 IO 操作串联

```haskell
main = putStrLn "Hello" >> putStrLn "world!"
```

这样的两个操作虽有先后关系, 但是没有依赖关系. 若是后一个操作依赖于前一个操作的结果, 则需要定义新的函数

```haskell
(>>=) :: IO a -> (a -> IO b) -> IO b
```

其将第二个参数由直接接受 `IO b` 改为了接受一个可以由结果 `a` 产生 `IO b` 的函数, 这样用户就可以决定如何由结果 `a` 产生结果 `b`. 

#### Record Syntax

​	在创建自定义类型时, 传统的语法为

```haskell
data D = C T1 T2 T3
```

其中 `T1 T2 T3` 可以看作自定义类型中字段的类型名. Record syntax 可以提供一种更详细的定义

```haskell
data D = C { field1 :: T1, field2 :: T2, field3 :: T3 }
```

这里 `field1` 看似是字段名, 本质上是一个函数

```haskell
field1 :: D -> T1
```

可以通过该函数+实例获取对应的字段.

​	可以通过下面的方式创建一个实例

```haskell
C { field3 = ..., field1 = ..., field2 = ... }
```

顺序无关, 也不一定需要初始化所有的字段.

#### Functors

##### high kind types

​	在 Haskel 中, 类型本身也有类型, 称为 "kinds". 在 ghci 中可以通过 `:k` 命令查看类型的 kind. 例如

```haskell
Prelude> :k Int
Int :: *
Prelude> :k Maybe Int
Maybe Int :: *
```

这里的 `*` 表示一种类型, `Int :: *` 即表示 `Int` 是一种类型.

​	既然 `Maybe Int ` 是一种类型, 那么 `Maybe` 本身是什么类型呢? 

```haskell
Prelude> :k Maybe
Maybe :: * -> *
```

可以看到 `Maybe` 的类型与函数的类型十分类似, 只是其参数类型和结果类型均为 `*`, 即某种类型. 因此 `Maybe` 的类型就很清楚了, 这是一种将某种类型转换为另一种类型的类型. 例如 `Maybe`  可以将 `Int` 类型转换为 `Maybe Int` 类型. 这样的类型称为「高阶类型(High Kind Type, HKT)」.常见的 HKT 还有 `[]`, 例如其可以将 `Int` 转换为 `[Int]` 类型. 

​	HKT 不只有 `* -> *` 一种类型, 考虑自定义类型

```haskell
data JoinList m a = Empty
                  | Single m a
                  | Append m (JoinList m a) (JoinList m a)
```

不需要借助 ghci, 可以直接看出 `JoinList` 的类型为 `* -> * -> *`, 因为其接受两个类型参数 `m, a`, 输出一个类型 `JoinList m a`. 

​	甚至对于函数构建运算符 `->`

```haskell
Prelude> :k (->)
(->) :: * -> * -> *
```

`->` 接受两个参数, 第一个是函数第一个参数的类型, 第二个是返回值类型, 输出一个函数类型.

##### functor

​	functor 在范畴学上指「函子」. 设 $C, D$ 为两个范畴, 从 $C$ 到 $D$ 的函子为一个映射 $F$ 满足

- 将每个对象 $X\in C$ 映射到 $F(X)\in D$ 上,

- 将每个态射 $f: X\rightarrow Y\in C$ 映射至一态射 $F(f): F(X)\rightarrow F(Y)\in D$, 且满足

  - 对任何对象 $X\in C$, 恒有 $F(\bold{\mathrm{id}}_X) = \bold{\mathrm{id}}_{F(X)}$
  - 对任何态射 $f: X\rightarrow Y, g:Y\rightarrow Z$, 恒有 $F(f\circ g) = F(f) \circ F(g)$. 

  换言之, 函子会保持单位态射与态射的复合.

​	在 Haskell 中, functor 的定义为

```haskell
-- f 为函子
-- (a -> b) 为态射
class Functor f where
  fmap :: (a -> b) -> f a -> f b
```

`fmap` 接受一个 `a -> b` 的态射, 输出一个 `f a -> f b` 的态射, 此即为函子. 从 `fmap` 的定义可以看出, `Functor` 的任何实例必须是 `* -> *` 的 kind, 例如 `Maybe`

```haskell
instance Functor Maybe where
  fmap _ Nothing = Nothing
  fmap f (Just a) = Just (f a)
```

​	如果将 instance 改为 `(->) e`, 前面提到过 `->` 的 kind 为 `* -> * -> *`, 因此 `(->) e` 的 kind 为 `* -> *`, 符合 Functor 的要求. 

​	令 `f = (->) e`, 则 `f a = e -> a`, 则 fmap 的类型可以写为

```haskell
fmap :: (a -> b) -> (e -> a) -> (e -> b)
```

可以看到, 这刚好是 Haskell `.` 运算符的定义. 

```haskell
instance Functor ((->) e) where
  fmap = (.)
```