-- step1
-- double every second digit in list
-- beginning from the right.

    {-# LANGUAGE NoImplicitPrelude #-}

import Prelude (String, Bool, Int, Integer, Show, undefined, otherwise, (>))

class Functor f where
    fmap :: (a -> b) -> f a -> f b
    (<$>) :: (a -> b) -> f a -> f b
    (<$>) = fmap

instance Functor [] where
    fmap _ [] = []
    fmap f (x:xs) = f x : fmap f xs

data Employee = Employee {name :: String, phone :: String} deriving Show

class Functor f => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b

fmap2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
fmap2 h fa fb = h <$> fa <*> fb

fmap3 :: Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
fmap3 h fa fb fc = h <$> fa <*> fb <*> fc

-- for pure
fmap4 :: Applicative f => (a -> b -> c) -> f a -> b -> f c
fmap4 h fa b = h <$> fa <*> pure b

data Maybe item = Nothing | Just item

instance Functor Maybe where
    fmap _ Nothing = Nothing
    fmap mapper (Just x) = Just (mapper x)

instance Applicative Maybe where
    pure = Just
    Nothing <*> _ = Nothing
    _ <*> Nothing = Nothing
    (Just f) <*> (Just x) = Just (f x)

name1, name2 :: Maybe String
name1 = Nothing
name2 = Just "Barret"

phone1, phone2 :: Maybe String
phone1 = Just "1231241"
phone2 = Nothing

emp1, emp2 :: Maybe Employee
emp1 = Employee <$> name1 <*> phone1
emp2 = Employee <$> name2 <*> phone2
