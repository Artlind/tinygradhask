module Tinygrad (Operation, Nombre (..), Graph (..), backward, getNombreFromId, createNombre, newNombreWithId, sumNombre, dotProduct, tanH) where

import Common
import qualified Data.HashMap.Strict as HM
import Data.Maybe
import Graphs
import Nombres

backwardParents :: Nombre -> Graph -> [Nombre]
backwardParents n graph = new_node_parents
  where
    op = operation n
    new_node_parents = case op of
      Rien -> nodeparents
      MSE ->
        let parent_1 :: Nombre
            parent_1 = head nodeparents
            parent_2 :: Nombre
            parent_2 = last nodeparents
         in [ Nombre (value parent_1) (grad parent_1 + 2 * (value parent_1 - value parent_2) * grad_child) (nombre_id parent_1) (parents parent_1) (operation parent_1),
              Nombre (value parent_2) (grad parent_2 + 2 * (value parent_2 - value parent_1) * grad_child) (nombre_id parent_2) (parents parent_2) (operation parent_2)
            ]
      TanH ->
        let
         in [Nombre (value nod) (grad nod + (1 - tanh (value nod) ** 2) * grad_child) (nombre_id nod) (parents nod) (operation nod) | nod <- nodeparents]
      DotProd ->
        let evens = [nod | (i, nod) <- zip [0 ..] nodeparents, even (i :: Int)]
            odds = [nod | (i, nod) <- zip [0 ..] nodeparents, odd (i :: Int)]
            result = concat [[Nombre (value ev) (grad ev + value od * grad_child) (nombre_id ev) (parents ev) (operation ev), Nombre (value od) (grad od + value ev * grad_child) (nombre_id od) (parents od) (operation od)] | (ev, od) <- zip evens odds]
         in result
      Sum ->
        let
         in [Nombre (value nod) (grad nod + grad_child) (nombre_id nod) (parents nod) (operation nod) | nod <- nodeparents]
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
        parentsn :: [Parent]
        parentsn = parents n
        nodeparents_all = [getNombreFromId parent graph | parent <- parentsn] -- Missing a functionality : A+B+C => (A+B)+C and maybe A+B not in graph...
        nodeparents = catMaybes nodeparents_all
        grad_child = grad n

backwardInner :: NodeId -> Graph -> Graph
backwardInner child_id graph =
  case child of
    Nothing -> graph
    Just n ->
      case parents n of
        [] -> graph
        _ ->
          let backwarded_parents = backwardParents n graph
              graph_parents = [backwardInner (nombre_id nod) (overwriteNode graph nod) | nod <- backwarded_parents]
           in addNodeToGraph n (removeNodeFromGraph n (mergeGraphs graph_parents))
  where
    child = getNombreFromId (Nid child_id) graph

zeroGrad :: Graph -> Graph
zeroGrad graph = new_graph
  where
    new_graph :: Graph
    new_graph = Graph $ HM.fromList [(nid, Nombre (value node) 0.0 nid (parents node) (operation node)) | (nid, node) <- HM.toList (nodes graph)]

backward :: NodeId -> Graph -> Graph
backward child_id graph =
  case child of
    Nothing -> graph
    Just n -> mergeGraphs [graph, backwardInner child_id (overwriteNode (zeroGrad graph) new_child)]
      where
        new_child = Nombre (value n) 1.0 child_id (parents n) (operation n)
  where
    child = getNombreFromId (Nid child_id) graph
