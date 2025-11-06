module MyLib (Operation, Nombre (..), Graph (..), backward, getNombreFromId, createNombre, newNombreWithId) where

import Common
import qualified Data.HashMap.Strict as HM
import Graphs
import Nombres

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
