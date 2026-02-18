module Transformer (AttentionHead, MultiHeadAttention (..), forwardAttentionHead, forwardMultiHeadAttention, newRandomAttentionHead, newRandomMultiHeadAttentionHead, fitBatchMHA, TokensEmbeddings) where

import qualified Data.HashMap.Strict as HM
import Data.List (transpose)
import Graphs
import Matrices
import Mlp
import System.Random (StdGen, splitGen)
import Tinygrad

-- Structs
data AttentionHead = AttentionHead
  { w_keys :: LinearLayer,
    w_vals :: LinearLayer,
    w_queries :: LinearLayer
  }
  deriving (Eq, Show)

data MultiHeadAttention = MultiHeadAttention
  { heads :: [AttentionHead],
    final_proj :: LinearLayer
  }
  deriving (Eq, Show)

type TokensEmbeddings = Matrix2d

type NumberOfHeads = Int

type HiddenDim = Int

type KeyQueriesDim = Int

type ValsDim = Int

-- Utils
getDk :: AttentionHead -> Maybe Int
getDk layer
  | null (coeffs w) = Nothing
  | otherwise = Just (length (head (coeffs w)))
  where
    (w, _) = w_keys layer

splitn :: StdGen -> NumberOfHeads -> [StdGen]
splitn key n
  | n <= 0 = []
  | n == 1 = [key]
  | otherwise = do
      let (key1, key2) = splitGen key
      key1 : splitn key2 (n - 1)

newRandomAttentionHead :: (HiddenDim, KeyQueriesDim, ValsDim, StdGen) -> Maybe AttentionHead
newRandomAttentionHead (dmodel, dk, dv, masterkey) = do
  let s1 = (dmodel, dk)
  let s2 = (dmodel, dv)
  let (key_keys, new_key) = splitGen masterkey
  let (key_queries, new_key2) = splitGen new_key2
  let (key_values, _) = splitGen new_key
  mlpkeys <- newRandomMlp [(s1, key_keys, False)]
  mlpqueries <- newRandomMlp [(s1, key_queries, False)]
  mlpvalues <- newRandomMlp [(s2, key_values, False)]
  let wkeys = head (layers mlpkeys)
  let wqueries = head (layers mlpqueries)
  let wvals = head (layers mlpvalues)
  Just $ AttentionHead wkeys wvals wqueries

newRandomMultiHeadAttentionHead :: (HiddenDim, KeyQueriesDim, ValsDim, StdGen, NumberOfHeads) -> Maybe MultiHeadAttention
newRandomMultiHeadAttentionHead (dmodel, dk, dv, masterkey, h) = do
  let (masterkeyheads, key_for_fp) = splitGen masterkey
  let keys_for_heads = splitn masterkeyheads h
  let all_heads = [newRandomAttentionHead (dmodel, dk, dv, keyheadi) | keyheadi <- keys_for_heads]
  let s_fp = (h * dv, dmodel)
  fp <- newRandomMlp [(s_fp, key_for_fp, True)]
  heads_proper <- sequence all_heads
  Just $ MultiHeadAttention heads_proper (head (layers fp))

-- Forwards
attend :: Double -> TokensEmbeddings -> TokensEmbeddings -> TokensEmbeddings -> Maybe TokensEmbeddings
attend scale q k v = do
  dp <- multMatrices q (transposeMatrix k)
  e <- divideMatrix dp scale
  sf <- linewiseSoftMax e
  multMatrices sf v

forwardAttentionHead :: AttentionHead -> [TokensEmbeddings] -> Maybe [TokensEmbeddings]
forwardAttentionHead layer embs = do
  keys <- forwardLinearBatch (w_keys layer) embs
  values <- forwardLinearBatch (w_vals layer) embs
  queries <- forwardLinearBatch (w_queries layer) embs
  dk <- getDk layer
  let scale = sqrt (fromIntegral dk)
  sequenceA (zipWith3 (attend scale) queries keys values)

forwardMultiHeadAttention :: MultiHeadAttention -> [TokensEmbeddings] -> Maybe [TokensEmbeddings]
forwardMultiHeadAttention layer embs = do
  head_results <- sequenceA [forwardAttentionHead atthead embs | atthead <- heads layer]
  let per_token_results = transpose head_results
  concated_head_results <-
    sequenceA
      [ concatMatricesColwise token_heads
        | token_heads <- per_token_results
      ]
  res <- sequenceA [forwardLinear (final_proj layer) emb | emb <- concated_head_results]
  Just res

-- Fit
allParamsFromAH :: AttentionHead -> [Nombre]
allParamsFromAH model = concat [allParamsFromLinear (w_keys model), allParamsFromLinear (w_vals model), allParamsFromLinear (w_queries model)]

allParamsFromMHA :: MultiHeadAttention -> [Nombre]
allParamsFromMHA model = concat (allParamsFromLinear (final_proj model) : [allParamsFromAH he | he <- heads model])

updateAHwithGraph :: AttentionHead -> Graph -> AttentionHead
updateAHwithGraph model graph = new_model
  where
    new_model = AttentionHead (updateLinearLayerWithGraph (w_keys model) graph) (updateLinearLayerWithGraph (w_vals model) graph) (updateLinearLayerWithGraph (w_queries model) graph)

updateMHAwithGraph :: MultiHeadAttention -> Graph -> MultiHeadAttention
updateMHAwithGraph model graph = new_model
  where
    new_model = MultiHeadAttention [updateAHwithGraph h graph | h <- heads model] (updateLinearLayerWithGraph (final_proj model) graph)

fitBatchMHA ::
  MultiHeadAttention ->
  ([TokensEmbeddings], [TokensEmbeddings]) ->
  Double ->
  Maybe MultiHeadAttention
fitBatchMHA model (inp, labels) lr =
  case forwardMultiHeadAttention model inp of
    Nothing -> Nothing
    Just ot ->
      do
        labs <- concatMatrices labels
        ots <- concatMatrices ot
        case meanSquaredError ots labs of
          Nothing -> Nothing
          Just mat ->
            let sum_loss =
                  sumNombre (allParamsFromMatrix mat)

                graph =
                  Graph
                    ( HM.fromList
                        [ (nombre_id node, node)
                          | node <- sum_loss : allParamsFromMHA model
                        ]
                    )

                backwarded_graph =
                  backward (nombre_id sum_loss) graph

                grad_steped_graph =
                  makeGradStep backwarded_graph lr

                new_model =
                  updateMHAwithGraph model grad_steped_graph
             in Just new_model
