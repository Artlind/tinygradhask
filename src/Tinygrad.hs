module Tinygrad (Operation, Nombre (..), Graph (..), backward, getNombreFromId, createNombre, newNombreWithId, sumNombre, dotProduct, tanH) where

import Common
import qualified Data.HashMap.Strict as HM
import Graphs
import Nombres

backwardParents :: Nombre -> [Nombre]
backwardParents n = new_node_parents
  where
    op = operation n
    new_node_parents = case op of
      Rien -> parentsn
      MSE ->
        let parent_1 :: Nombre
            parent_1 = head parentsn
            parent_2 :: Nombre
            parent_2 = last parentsn
         in [ Nombre (value parent_1) (grad parent_1 + 2 * (value parent_1 - value parent_2) * grad_child) (nombre_id parent_1) (parents parent_1) (operation parent_1),
              Nombre (value parent_2) (grad parent_2 + 2 * (value parent_2 - value parent_1) * grad_child) (nombre_id parent_2) (parents parent_2) (operation parent_2)
            ]
      TanH ->
        let
         in [Nombre (value nod) (grad nod + (1 - tanh (value nod) ** 2) * grad_child) (nombre_id nod) (parents nod) (operation nod) | nod <- parentsn]
      DotProd ->
        let evens = [nod | (i, nod) <- zip [0 ..] parentsn, even (i :: Int)]
            odds = [nod | (i, nod) <- zip [0 ..] parentsn, odd (i :: Int)]
            result = concat [[Nombre (value ev) (grad ev + value od * grad_child) (nombre_id ev) (parents ev) (operation ev), Nombre (value od) (grad od + value ev * grad_child) (nombre_id od) (parents od) (operation od)] | (ev, od) <- zip evens odds]
         in result
      Sum ->
        let
         in [Nombre (value nod) (grad nod + grad_child) (nombre_id nod) (parents nod) (operation nod) | nod <- parentsn]
      Add ->
        let
         in [Nombre (value nod) (grad nod + grad_child) (nombre_id nod) (parents nod) (operation nod) | nod <- parentsn]
      Mult ->
        let parent_1 :: Nombre
            parent_1 = head parentsn
            parent_2 :: Nombre
            parent_2 = last parentsn
         in [ Nombre (value parent_1) (grad parent_1 + value parent_2 * grad_child) (nombre_id parent_1) (parents parent_1) (operation parent_1),
              Nombre (value parent_2) (grad parent_2 + value parent_1 * grad_child) (nombre_id parent_2) (parents parent_2) (operation parent_2)
            ]
      Div ->
        let parent_num :: Nombre
            parent_num = head parentsn
            parent_den :: Nombre
            parent_den = last parentsn
         in [ Nombre (value parent_den) (grad parent_den - (grad_child * value parent_num / value parent_den ** 2)) (nombre_id parent_den) (parents parent_den) (operation parent_den),
              Nombre (value parent_num) (grad parent_num + grad_child / value parent_den) (nombre_id parent_num) (parents parent_num) (operation parent_num)
            ]
      where
        parentsn :: [Nombre]
        parentsn = parents n
        grad_child = grad n

backwardInner :: Nombre -> Graph -> Graph
backwardInner n graph
  | null (parents n) = graph
  | otherwise =
      let backwarded_parents = backwardParents n
          graph_parents = [backwardInner p (overwriteNode graph p) | p <- backwarded_parents]
       in addNodeToGraph n (removeNodeFromGraph n (mergeGraphs graph_parents))

zeroGrad :: Graph -> Graph
zeroGrad graph = new_graph
  where
    new_graph :: Graph
    new_graph = Graph $ HM.fromList [(nid, Nombre (value node) 0.0 nid (parents node) (operation node)) | (nid, node) <- HM.toList (nodes graph)]

backward :: NodeId -> Graph -> Graph
backward child_id graph =
  case child of
    Nothing -> graph
    Just n -> updateGraph graph (backwardInner root (overwriteNode (zeroGrad graph) root))
      where
        root = Nombre (value n) 1.0 child_id (parents n) (operation n)
  where
    child = getNombreFromId child_id graph
