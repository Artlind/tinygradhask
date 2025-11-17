module Graphs (Graph (..), getNombreFromId, overwriteNode, addNodeToGraph, removeNodeFromGraph, mergeGraphs, makeGradStep, updateGraph) where

import Common
import qualified Data.HashMap.Strict as HM
import Data.Maybe
import qualified Data.Set as Set
import Nombres

-- Graph struct
newtype Graph = Graph
  { nodes :: HM.HashMap NodeId Nombre
  }
  deriving (Eq, Show)

-- Utils
getNombreFromId :: NodeId -> Graph -> Maybe Nombre
getNombreFromId n_id graph = HM.lookup n_id (nodes graph)

overwriteNode :: Graph -> Nombre -> Graph
overwriteNode graph n = Graph (HM.fromList new_nodes)
  where
    id_to_change = nombre_id n
    new_nodes = [if current_id == id_to_change then (id_to_change, n) else (current_id, node) | (current_id, node) <- HM.toList (nodes graph) ++ [(id_to_change, n)]]

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

-- Gradients stuff
addGradients :: [Maybe Nombre] -> Maybe Nombre
addGradients nombres =
  case just_nombres of
    [] -> Nothing
    (nombre_1 : _) ->
      if requires_grad nombre_1
        then
          Just $
            Nombre
              (value nombre_1)
              (sum [grad n | n <- just_nombres])
              (nombre_id nombre_1)
              (parents nombre_1)
              (operation nombre_1)
              (requires_grad nombre_1)
        else Just nombre_1
  where
    just_nombres :: [Nombre]
    just_nombres = catMaybes nombres

mergeGraphs :: [Graph] -> Graph
mergeGraphs graphs = merged_graphs
  where
    all_unique_ids :: Set.Set NodeId
    all_unique_ids = Set.fromList (concat [HM.keys (nodes graph) | graph <- graphs])
    all_grad_summed_nombres :: [(NodeId, Maybe Nombre)]
    all_grad_summed_nombres = [(nid, addGradients [getNombreFromId nid graph | graph <- graphs]) | nid <- Set.toList all_unique_ids]
    merged_graphs = Graph (HM.fromList [(nid, n) | (nid, Just n) <- all_grad_summed_nombres])

updateGraph :: Graph -> Graph -> Graph
updateGraph old new = updated_graph
  where
    ids_old :: Set.Set NodeId
    ids_old = Set.fromList (HM.keys (nodes old))
    all_grad_summed_old_nombres :: [(NodeId, Maybe Nombre)]
    all_grad_summed_old_nombres = [(nid, addGradients [getNombreFromId nid graph | graph <- [old, new]]) | nid <- Set.toList ids_old]
    updated_graph = Graph (HM.fromList [(nid, n) | (nid, Just n) <- all_grad_summed_old_nombres])

makeGradStep :: Graph -> Double -> Graph
makeGradStep graph learning_rate = new_graph
  where
    old_nodes = HM.toList (nodes graph)
    -- We are never too prudent, if requires_grad is True, grad n should be 0 so this if else is not useful on paper.
    new_graph =
      Graph
        ( HM.fromList
            [ ( nid,
                if requires_grad n
                  then
                    Nombre
                      (value n - learning_rate * grad n)
                      (grad n)
                      (nombre_id n)
                      (parents n)
                      (operation n)
                      True
                  else n
              )
              | (nid, n) <- old_nodes
            ]
        )
