-- Date:   Mon Jun 24 13:11:39 2024
-- Mail:   lunar_ubuntu@qq.com
-- Author: https://github.com/xiaoqixian

import Prelude(print)
import Data.Data (typeOf)
import Common

class Functor f where 
  (<$>) :: (a -> b) -> f a -> f b

class Functor f => Applicative f where
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b

liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
liftA2 h fa fb = h <$> fa <*> fb

instance Functor Maybe where
  _ <$> Nothing = Nothing
  f <$> (Just a) = Just (f a)

instance Applicative Maybe where
  pure = Just
  Nothing <*> _ = Nothing
  _ <*> Nothing = Nothing
  (Just f) <*> (Just a) = Just (f a)
