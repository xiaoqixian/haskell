-- recusive functions
func :: Integer -> Integer
func 0 = 0
func n = n + func(n-1)
-- calling a function is like using a command

-- function braching by conditions
hailstone :: Integer -> Integer
hailstone n
    | even n = n `div` 2
    | otherwise = 3 * n + 1

-- function with multiple arguments
multipleArg :: Int -> Int -> Int
multipleArg x y = x + y

-- List
lst :: [Integer]
lst = [12, 231, 32]

-- sequential list
seq :: [Integer]
seq = [1..100]

-- String is just a list of chars
str1 :: [Char]
str1 = ['h', 'e', 'l', 'l', 'o']

str2 :: String
str2 = "hello"
-- str1 == str2

-- cons operator `:` takes an element and a list
-- and produces a new list with the element propended to the front.
expr1 = [2, 3, 4] Prelude.== 2 : 3 : 4 : []

-- `(elem:[list])` can be considered as a list variable
expr2 = (1 : [2]) Prelude.== [1 , 2]

-- a list can be expressed as `firstElem : tail`
listLength :: [Integer] -> Integer
listLength [] = 0
listLength (_:tail) = 1 + listLength tail
-- such operations can be nested

-- enum
data Thing = Shoe
            | Data
            | Ship

enum :: Thing
enum = Shoe

-- enum can carry any number of variables
data FailureDouble = Err | Ok Double

-- function pattern matching
-- foo (Constr1 a b)   = ...
-- foo (Constr2 a)     = ...
-- foo (Constr3 a b c) = ...
-- foo Constr4         = ...
-- underscore `_` matches everything
-- x@pat match a value against pattern `pat`, 
-- and gives the entire value to `x`
-- pattern can be nested


-- case expressions
-- case exp of
  -- pat1 -> exp1
  -- pat2 -> exp2
  -- ...

-- recursive data types
data IntList = Empty | Cons Int IntList

-- doing map on a list
absAll :: IntList -> IntList
absAll Empty = Empty
absAll (Cons x xs) = Cons (abs x) (absAll xs)

-- define a general IntList map function
exampleList = Cons (-1) (Cons 2 (Cons (-6) Empty))
square x = x * x

mapIntList :: (Int -> Int) -> IntList -> IntList
mapIntList _ Empty = Empty
mapIntList f (Cons x xs) = Cons (f x) (mapIntList f xs)

outputList :: IntList
outputList = mapIntList square exampleList

-- Filter
filterIntList :: (Int -> Bool) -> IntList -> IntList
filterIntList _ Empty = Empty
filterIntList predicate (Cons x xs)
    | predicate x = Cons x (filterIntList predicate xs)
    | otherwise = filterIntList predicate xs

-- Polymorphism
data List t = E | C t (List t)

-- define a generalized map
mapList :: (t -> u) -> List t -> List u
mapList _ E = E
mapList mapper (C x xs) = C (mapper x) (mapList mapper xs)

-- define a generalized filter
filterList :: (t -> Bool) -> List t -> List t
filterList _ E = E
filterList predicate (C x xs) 
    | predicate x = C x (filterList predicate xs)
    | otherwise = filterList predicate xs

-- define a generalized fold
foldList :: u -> (t -> u -> u) -> List t -> u
foldList acc _ E = acc
foldList init folder (C x xs) = foldList (folder x init) folder xs

-- partial function
-- define a safe head
safeHead :: [a] -> Maybe a
safeHead [] = Nothing
safeHead (x:_) = Just x


-- lambda function
-- lambda functions start with a back slash
greaterThan100 :: [Integer] -> [Integer]
greaterThan100 = filter (\x -> x > 100)

-- operator section
-- if `?` is an operator, then (?y) is equivalent to the function
-- \x -> x ? y.
greaterThan100_2 :: [Integer] -> [Integer]
greaterThan100_2 = filter (> 100)

-- function composition
foo :: (b -> c) -> (a -> b) -> (a -> c)
foo f g x = f (g x)

-- with function composition
-- we can combine multiple functions into one function
myFunc :: [Integer] -> Bool
myFunc xs = even (length (greaterThan100 xs))

-- currying and partial application
-- Astonishing Facts: all functions in Haskell take 
-- only one argument.
-- For functions take more than one argument, they take 
-- one argument and generate a new function
-- so W -> X -> Y -> Z is equivalent to W -> (X -> (Y -> Z))
-- functions take more than one argument is just a shorthand
--
-- This idea of representing multi-argument functions as one-argument
-- functions returning functions is known as currying.

-- curry a function
curry :: ((a, b) -> c) -> a -> b -> c
curry f x y = f (x, y)

-- uncurry a function
uncurry :: (a -> b -> c) -> (a, b) -> c
uncurry f (x, y) = f x y

-- uncurry is particular useful when you want to apply a function
-- to a pair.


-- partial application
-- The idea of partial application is that we can take a function
-- of multiple arguments and apply it to just some of its arguments,
-- and get out a function of the remaining arguments.
--
-- the arguments should be ordered from "leat to greatest variation"
completeFunc :: Int -> Int -> Bool
completeFunc x y = x > y

partialFunc :: Int -> Bool
partialFunc = completeFunc 100

-- `.` operator is like a pipe
foobar :: [Integer] -> Integer
foobar = sum . map (\x -> 7*x + 2) . filter (>3)
-- foobar is equivalent to sum (map (\x -> 7*x + 2) (filter (>3)))


-- Type Classes
-- A type class is defined as `class name type_parameter where`
-- A type class polymorphic function is restricted promise that 
-- the function will work for any type the caller chooses, as
-- long as the chosen type is an instance of the required type class.
class MEq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool

data Foo = F Int | G Char

-- to implement type class Eq for type Foo
instance MEq Foo where 
    (F i1) == (F i2) = i1 Prelude.== i2
    (G c1) == (G c2) = c1 Prelude.== c2
    _ == _ = False

-- type class can also give default implementation
class NEq a where 
    eq, neq :: a -> a -> Bool
    eq x y = not (neq x y)
    neq x y = not (eq x y)

-- type classes like `Eq` can be automatically derived 

-- a type class example
class Listable a where
    toList :: a -> [Int]

instance Listable Int where
    toList x = [x]

instance Listable Bool where
    toList True = [1]
    toList False = [0]


-- Lazy evaluation
-- Under a lazy evaluation strategy, evaluation of function arguments
-- is delayed as long as possible: they are not evaluated until it 
-- actually becomes necessary to do so.
--
-- Pattern Matching Drives evaluation
--  1. Expressions are only evaluted when pattern-matched
--  2. Only as far as necessary for hte match to proceed, and no further.

-- classic examples: repeat and take
repeat :: Int -> [Int]
repeat x = x : Main.repeat x

take :: Int -> [Int] -> [Int]
take n _ | n <= 0 = []
take _ [] = []
take n (x:xs) = x : Main.take (n-1) xs

res1 :: [Int]
-- accually it only repeat 3 times here
res1 = Main.take 3 (Main.repeat 7)

-- lazy evaluation of short-circuting operators
-- lazy evaluation of `&&` operator
(&&) :: Bool -> Bool -> Bool
True && x = x -- x don't have to be evaluated
False && _ = False

-- if you want x to be evaluated
(&&!) :: Bool -> Bool -> Bool
True  &&! True  = True
True  &&! False = False
False &&! True  = False
False &&! False = False

-- user defined control structures
if' :: Bool -> a -> a -> a
if' True x _ = x
if' False _ y = y

-- define a general tree folder
data Tree a = Nil | Node (Tree a) a (Tree a)

treeFold :: b -> (b -> a -> b -> b) -> Tree a -> b
treeFold n _ Nil = n
treeFold n f (Node l x r) = f (treeFold n f l) x (treeFold n f r)

treeSize :: Tree a -> Integer
treeSize = treeFold 0 (\l _ r -> 1 + l + r)

treeSum :: Tree Integer -> Integer
treeSum = treeFold 0 (\l x r -> l + x + r)

treeDepth :: Tree a -> Integer
treeDepth = treeFold 0 (\l x r -> 1 + max l r)

flatten :: Tree a -> [a]
flatten = treeFold [] (\l x r -> l ++ [x] ++ r)

-- Monoid
class Monoid m where
    mempty :: m
    mappend :: m -> m -> m
    mconcat :: [m] -> m
    mconcat = Prelude.foldr Main.mappend Main.mempty

-- `Monoid m =>` says that a type variable `m`
-- must be an instance of the `Monoid` type class.
(<>) :: Main.Monoid m => m -> m -> m
(<>) = Main.mappend

-- Monoid makes sure no matter how we parenthesize,
-- we always get the same result.
--expr3 = (x Main.<> y) Main.<> z Prelude.== x Main.<> (y Main.<> z)

-- List from a monoid under concatenation:
instance Main.Monoid [a] where
    mempty = []
    mappend = (++)

newtype Sum a = Sum a
    deriving (Eq, Ord, Num, Show)

getSum :: Sum a -> a
getSum (Sum a) = a

instance Num a => Main.Monoid (Sum a) where
    mempty = Sum 0
    mappend = (+)

newtype Product a = Product a
    deriving (Eq, Ord, Num, Show)

getProduct :: Product a -> a
getProduct (Product a) = a

instance Num a => Main.Monoid (Product a) where
    mempty = Product 1
    mappend = (*)

-- Now we can easily sum a list of integers
lst2 :: [Integer]
lst2 = [1,3,5,6,12,56,23]

prod :: Integer
-- `$` is an infix operator, it applies the function on its left 
-- to the value on its right.
prod = getProduct . Main.mconcat . Prelude.map Product $ lst2
-- the above line is amazing

-- perform monoid on pairs
instance (Main.Monoid a, Main.Monoid b) => Main.Monoid (a, b) where
    mempty = (Main.mempty, Main.mempty)
    mappend (a, b) (c, d) = (Main.mappend a c, Main.mappend b d)


-- Week 8 IO
-- Haskell performs all IO computation in the `main` function
-- It can be easily done with lazy evaluation
--
-- define a `(>>)` operator to combine IO
-- (>>) :: IO a -> IO b -> IO b

-- suppose we want the second computation to be able to depend on 
-- the result of the first.
-- (>>=) :: IO a -> (a -> IO b) -> IO b
--main2 :: IO ()
--main2 = putStrLn "Please enter a numer: " >> (readLn >>= (\n -> print n+1))


-- Week 9 Functor
class Functor f where
    fmap :: (a -> b) -> f a -> f b
    (<$>) :: (a -> b) -> f a -> f b
    (<$>) = Main.fmap

instance Main.Functor Maybe where
    fmap _ Nothing = Nothing
    fmap f (Just a) = Just (f a)

instance Main.Functor [] where
    fmap _ [] = []
    fmap f (x:xs) = f x : Main.fmap f xs

instance Main.Functor IO where
    fmap f ioa = ioa >>= (return . f)


instance Main.Functor ((->) e) where
    fmap = (.)

-- Week 10 Applicative Functors
-- To ensure `fmap` works sanely, any instance of `Functor` must 
-- comply with the following two laws:
--  1. `fmap id = id`
--  2. `fmap (g . f) = fmap g . fmap f`
class Main.Functor f => Applicative f where
    pure :: a -> f a
    -- <*> is just an operator of Applicative
    (<*>) :: f (a -> b) -> f a -> f b

instance Main.Applicative Maybe where
    -- `pure` simply wraps the value with `Just`
    pure = Just
    Nothing <*> _ = Nothing
    _ <*> Nothing = Nothing
    (Just f) <*> (Just x) = Just (f x)

liftA2 :: Main.Applicative f => (a -> b -> c) -> f a -> f b -> f c
liftA2 h fa fb = Main.fmap h fa Main.<*> fb

-- `Applicative` has a set of laws which reasonable instance
-- should follow.
-- 1. apply the `pure id` does nothing
-- `pure id <*> v = v`
--
-- 2. apply a `pure` function to a `pure` value is the same as
-- applying the function to the value in the normal way.
-- `pure f <*> pure x = pure (f x)`
--
-- 3. apply a morphism to a `pure` value is the same as applying 
-- `pure ($ y)` to the morphism.
-- `u <*> pure y = pure ($ y) <*> u`
--
-- 4. `pure (.)` composes morphisms similaryly to how `(.)` composes
-- functions: applying the composed morphism `pure (.) <*> u <*> v`
-- `pure (.) <*> u <*> v <*> w = u <*> (v <*> w)`

instance Main.Applicative [] where
    pure a = [a]
    [] <*> _ = []
    (f:fs) <*> as = Main.fmap f as ++ (fs Main.<*> as)

data Employee = Employee { name :: String,  phone :: String } deriving Show

names  = ["Joe", "Sara", "Mae"]
phones = ["555-5555", "123-456-7890", "555-4321"]

employees = Employee Main.<$> names Main.<*> phones

main :: IO ()
main = print employees
