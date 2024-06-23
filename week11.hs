import Prelude (IO, print, Eq, Show, Int, (.))

(++) :: [a] -> [a] -> [a]
(++) [] ys = ys
(++) (x:xs) ys = x : xs ++ ys

repeat :: a -> [a]
repeat x = x : repeat x

class Functor f where
    fmap :: (a -> b) -> f a -> f b 
    (<$>) :: (a -> b) -> f a -> f b 
    (<$>) = fmap

class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

instance Functor [] where
    fmap _ [] = []
    fmap f (x:xs) = f x : fmap f xs

instance Applicative [] where
    pure a = [a]
    [] <*> _ = []
    (f:fs) <*> as = fmap f as ++ (fs <*> as)

newtype ZipList a = ZipList { getZipList :: [a] }
    deriving (Eq, Show, Functor)

instance Applicative ZipList where
    pure = ZipList . repeat
    --ZipList fs <*> ZipList xs = ZipList (zipWidth <$> fs xs)
