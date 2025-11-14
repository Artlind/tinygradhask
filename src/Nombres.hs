module Nombres (Nombre (..), createNombre, newNombreWithId, sumNombre, dotProduct, tanH, mse) where

import Common

data Nombre = Nombre
  { value :: Double,
    grad :: Double,
    nombre_id :: NodeId,
    parents :: [Nombre],
    operation :: Operation
  }
  deriving (Eq, Show)

instance Num Nombre where
  Nombre a ga n1 pa oa + Nombre b gb n2 pb ob = Nombre (a + b) 0 (n1 ++ "+" ++ n2) [Nombre a ga n1 pa oa, Nombre b gb n2 pb ob] Add
  Nombre a ga n1 pa oa * Nombre b gb n2 pb ob = Nombre (a * b) 0 ("(" ++ n1 ++ ")" ++ "*" ++ "(" ++ n2 ++ ")") [Nombre a ga n1 pa oa, Nombre b gb n2 pb ob] Mult
  fromInteger n = Nombre (fromIntegral n) 0 "constant" [] Rien
  abs (Nombre a ga n1 pa oa) = Nombre (abs a) 0 ("abs(" ++ n1 ++ ")") [Nombre a ga n1 pa oa, Nombre (signum a) 0 "constant" [] Rien] Mult
  negate (Nombre a ga n1 pa oa) = Nombre (-a) 0 ("-(" ++ n1 ++ ")") [Nombre a ga n1 pa oa, Nombre (-1) 0 "constant" [] Rien] Mult
  signum (Nombre a _ n1 _ _) = Nombre (signum a) 0 ("signum(" ++ n1 ++ ")") [] Rien

instance Fractional Nombre where
  Nombre a ga n1 pa oa / Nombre b gb n2 pb ob = Nombre (a / b) 0 (n1 ++ "/" ++ n2) [Nombre a ga n1 pa oa, Nombre b gb n2 pb ob] Div
  recip (Nombre a ga n1 pa oa) = Nombre (1 / a) 0 ("1/" ++ n1) [Nombre 1 0 "constant" [] Rien, Nombre a ga n1 pa oa] Div
  fromRational r = Nombre (fromRational r) 0 "constant" [] Rien

newNombreWithId :: (NodeId, Nombre) -> Nombre
newNombreWithId (new_id, n) = Nombre (value n) (grad n) new_id (parents n) (operation n)

createNombre :: (NodeId, Double) -> Nombre
createNombre (nid, x) = Nombre x 0.0 nid [] Rien

sumNombre :: [Nombre] -> Nombre
sumNombre nombres = Nombre (sum [value n | n <- nombres]) 0 (concatMap (++ "+") [nombre_id n | n <- nombres]) nombres Sum

dotProduct :: [Nombre] -> [Nombre] -> Maybe Nombre
dotProduct n1s n2s
  | l1 /= l2 = Nothing
  | otherwise = Just $ Nombre (value computed_nombre) 0 (nombre_id computed_nombre) (concat [[n1s !! i, n2s !! i] | i <- [0 .. l1 - 1]]) DotProd
  where
    l1 = length n1s
    l2 = length n2s
    minls = min l1 l2
    computed_nombre = sumNombre [n1s !! i * n2s !! i | i <- [0 .. minls - 1]]

tanH :: Nombre -> Nombre
tanH n = Nombre (tanh (value n)) 0 ("tanh(" ++ nombre_id n ++ ")") [n] TanH

mse :: Nombre -> Nombre -> Nombre
mse n1 n2 = Nombre ((value n1 - value n2) ** 2) 0 ("MSE(" ++ nombre_id n1 ++ "," ++ nombre_id n2) [n1, n2] MSE
