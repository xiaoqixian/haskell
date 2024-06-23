data Option a = None | Some a deriving Show

class MyFunctor g where
    gmap :: (a -> b) -> g a -> g b

class MyFunctor g => MyApplicative g where
    clear :: a -> g a
    gapp :: g (a -> b) -> g a -> g b

instance MyFunctor ((->) e) where
    gmap = (.)

instance MyApplicative ((->) e) where
    clear = const
    gapp f x e = f e (x e)

data Employee = Employee { name :: String, phone :: String }
data BigRecord = BR { getName         :: String
                    , getSSN          :: String
                    , getSalary       :: Integer
                    , getPhone        :: String
                    , getLicensePlate :: String
                    , getNumSickDays  :: Int
                    }

getEmp :: BigRecord -> Employee
getEmp = gapp (gmap Employee getName) getPhone
