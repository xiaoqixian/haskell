-- Date:   Sun Jun 23 20:04:09 2024
-- Mail:   lunar_ubuntu@qq.com
-- Author: https://github.com/xiaoqixian

import Prelude((.))
import Common

class Functor f where
  fmap :: (a -> b) -> f a -> f b

instance Functor Maybe where
  fmap _ Nothing = Nothing
  fmap f (Just a) = Just (f a)

instance Functor ((->) e) where
  fmap = (.)
