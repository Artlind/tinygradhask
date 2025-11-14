module Nombres (Nombre (..), createNombre, newNombreWithId, sumNombre, dotProduct, tanH, mse, Parent (..)) where

import Common

data Parent
  = Nid NodeId
  | RawNombre Nombre

instance Show Parent where
  show (Nid nid) = "Nid " ++ show nid
  show (RawNombre n) = "RawNombre(" ++ nombre_id n ++ ")"

instance Eq Parent where
  (Nid id1) == (Nid id2) = id1 == id2
  (RawNombre n1) == (RawNombre n2) = nombre_id n1 == nombre_id n2
  _ == _ = False

data Nombre = Nombre
  { value :: Double,
    grad :: Double,
    nombre_id :: NodeId,
    parents :: [Parent],
    operation :: Operation
  }
  deriving (Eq, Show)

instance Num Nombre where
  Nombre a _ n1 _ _ + Nombre c _ n2 _ _ = Nombre (a + c) 0 (n1 ++ "+" ++ n2) [Nid n1, Nid n2] Add
  Nombre a _ n1 _ _ * Nombre c _ n2 _ _ = Nombre (a * c) 0 ("(" ++ n1 ++ ")" ++ "*" ++ "(" ++ n2 ++ ")") [Nid n1, Nid n2] Mult
  fromInteger n = Nombre (fromIntegral n) 0 "constant" [] Rien
  abs (Nombre a _ n1 _ _) = Nombre (abs a) 0 ("abs(" ++ n1 ++ ")") [Nid n1, RawNombre (Nombre (signum a) 0 "constant" [] Rien)] Mult
  negate (Nombre a _ n1 _ _) = Nombre (-a) 0 ("-(" ++ n1 ++ ")") [Nid n1, RawNombre (Nombre (-1) 0 "constant" [] Rien)] Mult
  signum (Nombre a _ n1 _ _) = Nombre (signum a) 0 ("signum(" ++ n1 ++ ")") [] Rien

instance Fractional Nombre where
  (Nombre a _ n1 _ _) / (Nombre b _ n2 _ _) = Nombre (a / b) 0 (n1 ++ "/" ++ n2) [Nid n1, Nid n2] Div
  recip (Nombre a _ n1 _ _) = Nombre (1 / a) 0 ("1/" ++ n1) [RawNombre (Nombre 1 0 "constant" [] Rien), Nid n1] Div
  fromRational r = Nombre (fromRational r) 0 "constant" [] Rien

newNombreWithId :: (NodeId, Nombre) -> Nombre
newNombreWithId (new_id, n) = Nombre (value n) (grad n) new_id (parents n) (operation n)

createNombre :: (NodeId, Double) -> Nombre
createNombre (nid, x) = Nombre x 0.0 nid [] Rien

sumNombre :: [Nombre] -> Nombre
sumNombre [] = 0 :: Nombre
sumNombre nombres = Nombre (value rawsum) 0 name (fmap Nid all_ids) Sum
  where
    rawsum = head nombres + sumNombre (tail nombres)
    all_ids = [nombre_id n | n <- nombres]
    name = concatMap (++ "+") all_ids

dotProduct :: [Nombre] -> [Nombre] -> Nombre
dotProduct n1s n2s
  | l1 /= l2 = error $ "Lists of different sizes " ++ show l1 ++ " vs " ++ show l2
  | otherwise = Nombre (value computed_nombre) 0 (nombre_id computed_nombre) (concat [[Nid (nombre_id (n1s !! i)), Nid (nombre_id (n2s !! i))] | i <- [0 .. l1 - 1]]) DotProd
  where
    l1 = length n1s
    l2 = length n2s
    minls = min l1 l2
    computed_nombre = sumNombre [n1s !! i * n2s !! i | i <- [0 .. minls - 1]]

tanH :: Nombre -> Nombre
tanH n = Nombre (tanh (value n)) 0 ("tanh(" ++ nombre_id n ++ ")") [Nid (nombre_id n)] TanH

mse :: Nombre -> Nombre -> Nombre
mse n1 n2 = Nombre ((value n1 - value n2) ** 2) 0 ("MSE(" ++ nombre_id n1 ++ "," ++ nombre_id n2) [Nid (nombre_id n1), Nid (nombre_id n2)] MSE
