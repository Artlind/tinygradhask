module Graphs (Graph (..), getNombreFromId, overwriteNode, addNodeToGraph, removeNodeFromGraph, mergeGraphs, makeGradStep) where

import Common
import qualified Data.HashMap.Strict as HM
import qualified Data.Set as Set
import Nombres

newtype Graph = Graph
  { nodes :: HM.HashMap NodeId Nombre
  }
  deriving (Eq, Show)

getNombreFromId :: NodeId -> Graph -> Nombre
getNombreFromId n_id graph =
  case HM.lookup n_id (nodes graph) of
    Just t -> t
    Nothing -> error $ "No nombre for name: " ++ n_id --TODO make that return Nothing so we can use it to fail safely and freeze some layers in MLP

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

makeGradStep :: Graph -> Double -> Graph
makeGradStep graph learning_rate = new_graph
  where
    old_nodes = HM.toList (nodes graph)
    new_graph = Graph (HM.fromList [(nid, Nombre (value n - learning_rate * grad n) (grad n) (nombre_id n) (parents n) (operation n)) | (nid, n) <- old_nodes])
