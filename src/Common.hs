module Common (Operation (..), NodeId) where

-- | Some kind of autograd in Haskell
data Operation = Add | Mult | Rien | Div | Sum | DotProd | TanH
  deriving (Eq, Show)

type NodeId = String
