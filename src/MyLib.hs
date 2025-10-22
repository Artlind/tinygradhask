module MyLib (Operation, Nombre (..), Graph (..), backward, getNombreFromId, createNombre, newNombreWithId) where

import qualified Data.HashMap.Strict as HM
import qualified Data.Set as Set

-- | Some kind of autograd in Haskell
data Operation = Add | Mult | Rien | Div
  deriving (Eq, Show)

type NodeId = String

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

newtype Graph = Graph
  { nodes :: HM.HashMap NodeId Nombre
  }
  deriving (Eq, Show)

getNombreFromId :: NodeId -> Graph -> Nombre
getNombreFromId n_id graph =
  case HM.lookup n_id (nodes graph) of
    Just t -> t
    Nothing -> error $ "No nombre for name: " ++ n_id

overwriteNode :: Graph -> Nombre -> Graph
overwriteNode graph n = Graph (HM.fromList new_nodes)
  where
    id_to_change = nombre_id n
    new_nodes = [if current_id == id_to_change then (id_to_change, n) else (current_id, node) | (current_id, node) <- HM.toList (nodes graph)]

addNodeToGraph :: Nombre -> Graph -> Graph
addNodeToGraph n graph = Graph (HM.fromList new_nodes)
  where
    nid = nombre_id n
    new_nodes = (nid, n) : HM.toList (nodes graph)

removeNodeFromGraph :: Nombre -> Graph -> Graph
removeNodeFromGraph n graph = Graph (HM.fromList new_nodes)
  where
    to_remove_id = nombre_id n
    new_nodes = [(nid, no) | (nid, no) <- HM.toList (nodes graph), nid /= to_remove_id]

backwardParents :: Nombre -> Graph -> [Nombre]
backwardParents n graph = new_node_parents
  where
    op = operation n
    new_node_parents = case op of
      Rien -> []
      Add ->
        let
         in [Nombre (value nod) (grad nod + grad_child) (nombre_id nod) (parents nod) (operation nod) | nod <- nodeparents]
      Mult ->
        let parent_1 :: Nombre
            parent_1 = head nodeparents
            parent_2 :: Nombre
            parent_2 = last nodeparents
         in [ Nombre (value parent_1) (grad parent_1 + value parent_2 * grad_child) (nombre_id parent_1) (parents parent_1) (operation parent_1),
              Nombre (value parent_2) (grad parent_2 + value parent_1 * grad_child) (nombre_id parent_2) (parents parent_2) (operation parent_2)
            ]
      Div ->
        let parent_num :: Nombre
            parent_num = head nodeparents
            parent_den :: Nombre
            parent_den = last nodeparents
         in [ Nombre (value parent_den) (grad parent_den - (grad_child * value parent_num / value parent_den ** 2)) (nombre_id parent_den) (parents parent_den) (operation parent_den),
              Nombre (value parent_num) (grad parent_num + grad_child / value parent_den) (nombre_id parent_num) (parents parent_num) (operation parent_num)
            ]
      where
        parentsn = parents n
        nodeparents = [getNombreFromId nid graph | nid <- parentsn]
        grad_child = grad n

addGradients :: [Nombre] -> Nombre
addGradients nombres = added
  where
    nombre_1 = head nombres
    added = Nombre (value nombre_1) (sum [grad n | n <- nombres]) (nombre_id nombre_1) (parents nombre_1) (operation nombre_1)

mergeGraphs :: [Graph] -> Graph
mergeGraphs graphs = merged_graphs
  where
    all_unique_ids :: Set.Set NodeId
    all_unique_ids = Set.fromList (concat [HM.keys (nodes graph) | graph <- graphs])
    merged_graphs = Graph (HM.fromList [(nid, addGradients [getNombreFromId nid graph | graph <- graphs]) | nid <- Set.toList all_unique_ids])

backwardInner :: NodeId -> Graph -> Graph
backwardInner child_id graph
  | null (parents child) = graph
  | otherwise = addNodeToGraph child (removeNodeFromGraph child (mergeGraphs graph_parents))
  where
    child = getNombreFromId child_id graph
    backwarded_parents = backwardParents child graph
    graph_parents = [backwardInner (nombre_id nod) (overwriteNode graph nod) | nod <- backwarded_parents]

zeroGrad :: Graph -> Graph
zeroGrad graph = new_graph
  where
    new_graph :: Graph
    new_graph = Graph $ HM.fromList [(nid, Nombre (value node) 0.0 nid (parents node) (operation node)) | (nid, node) <- HM.toList (nodes graph)]

backward :: NodeId -> Graph -> Graph
backward child_id graph
  | null (parents child) = overwriteNode graph new_child
  | otherwise = mergeGraphs [graph, backwardInner child_id (overwriteNode (zeroGrad graph) new_child)]
  where
    child = getNombreFromId child_id graph
    new_child = Nombre (value child) 1.0 child_id (parents child) (operation child)
