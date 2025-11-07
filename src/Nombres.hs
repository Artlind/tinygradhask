module Nombres (Nombre (..), createNombre, newNombreWithId, sumNombre) where

import Common

data Nombre = Nombre
  { value :: Double,
    grad :: Double,
    nombre_id :: NodeId,
    parents :: [NodeId],
    operation :: Operation
  }
  deriving (Eq, Show)

instance Num Nombre where
  Nombre a _ n1 _ _ + Nombre c _ n2 _ _ = Nombre (a + c) 0 (n1 ++ "+" ++ n2) [n1, n2] Add
  Nombre a _ n1 _ _ * Nombre c _ n2 _ _ = Nombre (a * c) 0 ("(" ++ n1 ++ ")" ++ "*" ++ "(" ++ n2 ++ ")") [n1, n2] Mult
  fromInteger n = Nombre (fromIntegral n) 0 (show n) [] Rien
  abs (Nombre a _ n1 _ _) = Nombre (abs a) 0 ("abs(" ++ n1 ++ ")") [n1, show (signum a)] Mult
  negate (Nombre a _ n1 _ _) = Nombre (-a) 0 ("-(" ++ n1 ++ ")") [n1, "-1"] Mult
  signum (Nombre a _ n1 _ _) = Nombre (signum a) 0 ("signum(" ++ n1 ++ ")") [] Rien

instance Fractional Nombre where
  (Nombre a _ n1 _ _) / (Nombre b _ n2 _ _) = Nombre (a / b) 0 (n1 ++ "/" ++ n2) [n1, n2] Div
  recip (Nombre a _ n1 _ _) = Nombre (1 / a) 0 ("1/" ++ n1) ["1", n1] Div
  fromRational r = Nombre (fromRational r) 0 (show r) [] Rien

newNombreWithId :: (NodeId, Nombre) -> Nombre
newNombreWithId (new_id, n) = Nombre (value n) (grad n) new_id (parents n) (operation n)

createNombre :: (NodeId, Double) -> Nombre
createNombre (nid, x) = Nombre x 0.0 nid [] Rien

sumNombre :: [Nombre] -> Nombre
sumNombre [] = 0 :: Nombre
sumNombre nombres = Nombre (value rawsum) 0 name all_ids Sum
  where
    rawsum = head nombres + sumNombre (tail nombres)
    all_ids = [nombre_id n | n <- nombres]
    name = concatMap (++ "+") all_ids
