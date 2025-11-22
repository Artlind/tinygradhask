module Nombres (Nombre (..), createNombre, newNombreWithId, sumNombre, dotProduct, tanH, mse, nombreNoGrad, softmax) where

import Common

-- Nombre struct
data Nombre = Nombre
  { value :: Double,
    grad :: Double,
    nombre_id :: NodeId,
    parents :: [Nombre],
    operation :: Operation,
    requires_grad :: Bool
  }
  deriving (Eq, Show)

instance Num Nombre where
  Nombre a ga n1 pa oa rga + Nombre b gb n2 pb ob rgb = Nombre (a + b) 0 (n1 ++ "+" ++ n2) [Nombre a ga n1 pa oa rga, Nombre b gb n2 pb ob rgb] Add (rga || rgb)
  Nombre a ga n1 pa oa rga * Nombre b gb n2 pb ob rgb = Nombre (a * b) 0 ("(" ++ n1 ++ ")" ++ "*" ++ "(" ++ n2 ++ ")") [Nombre a ga n1 pa oa rga, Nombre b gb n2 pb ob rgb] Mult (rga || rgb)
  fromInteger n = Nombre (fromIntegral n) 0 "constant" [] Rien False
  abs (Nombre a ga n1 pa oa rga) = Nombre (abs a) 0 ("abs(" ++ n1 ++ ")") [Nombre a ga n1 pa oa rga, Nombre (signum a) 0 "constant" [] Rien False] Mult rga
  negate (Nombre a ga n1 pa oa rga) = Nombre (-a) 0 ("-(" ++ n1 ++ ")") [Nombre a ga n1 pa oa rga, Nombre (-1) 0 "constant" [] Rien False] Mult rga
  signum (Nombre a _ n1 _ _ _) = Nombre (signum a) 0 ("signum(" ++ n1 ++ ")") [] Rien False

instance Fractional Nombre where
  Nombre a ga n1 pa oa rga / Nombre b gb n2 pb ob rgb = Nombre (a / b) 0 (n1 ++ "/" ++ n2) [Nombre a ga n1 pa oa rga, Nombre b gb n2 pb ob rgb] Div (rga || rgb)
  recip (Nombre a ga n1 pa oa rga) = Nombre (1 / a) 0 ("1/" ++ n1) [Nombre 1 0 "constant" [] Rien False, Nombre a ga n1 pa oa rga] Div rga
  fromRational r = Nombre (fromRational r) 0 "constant" [] Rien False

-- Utils for creating numbers
newNombreWithId :: (NodeId, Nombre) -> Nombre
newNombreWithId (new_id, n) = Nombre (value n) (grad n) new_id (parents n) (operation n) (requires_grad n)

createNombre :: (NodeId, Double) -> Nombre
createNombre (nid, x) = Nombre x 0.0 nid [] Rien True

nombreNoGrad :: Nombre -> Nombre
nombreNoGrad (Nombre a ga n1 pa oa _) = Nombre a ga n1 pa oa False

-- More advanced operations
sumNombre :: [Nombre] -> Nombre
sumNombre nombres = Nombre (sum [value n | n <- nombres]) 0 (concatMap (++ "+") [nombre_id n | n <- nombres]) nombres Sum (or [requires_grad n | n <- nombres])

dotProduct :: [Nombre] -> [Nombre] -> Maybe Nombre
dotProduct n1s n2s
  | l1 /= l2 = Nothing
  | otherwise = Just $ Nombre (value computed_nombre) 0 (nombre_id computed_nombre) (concat [[n1s !! i, n2s !! i] | i <- [0 .. l1 - 1]]) DotProd (or [requires_grad n | n <- n1s ++ n2s])
  where
    l1 = length n1s
    l2 = length n2s
    minls = min l1 l2
    computed_nombre = sumNombre [n1s !! i * n2s !! i | i <- [0 .. minls - 1]]

tanH :: Nombre -> Nombre
tanH n = Nombre (tanh (value n)) 0 ("tanh(" ++ nombre_id n ++ ")") [n] TanH (requires_grad n)

mse :: Nombre -> Nombre -> Nombre
mse n1 n2 = Nombre ((value n1 - value n2) ** 2) 0 ("MSE(" ++ nombre_id n1 ++ "," ++ nombre_id n2) [n1, n2] MSE (requires_grad n1 || requires_grad n2)

softmax :: [Nombre] -> Maybe [Nombre]
softmax [] = Nothing
softmax ns =
  Just $
    [ Nombre
        (exp (value n) / s)
        0
        ("softmax" ++ "(" ++ nombre_id n ++ "/" ++ (concatMap (++ ",") [nombre_id ni | ni <- ns, nombre_id ni /= nombre_id n]) ++ ")")
        (n : [ni | ni <- ns, nombre_id ni /= nombre_id n])
        SoftMax
        req_grad
      | n <- ns
    ]
  where
    s = sum [exp (value n) | n <- ns]
    req_grad = or [requires_grad n | n <- ns]
