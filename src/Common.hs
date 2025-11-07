module Common (Operation (..), NodeId) where

-- | Some kind of autograd in Haskell
data Operation = Add | Mult | Rien | Div | Sum
  deriving (Eq, Show)

type NodeId = String
