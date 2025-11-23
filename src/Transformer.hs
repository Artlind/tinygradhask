module Transformer (AttentionHead, MultiHeadAttention (..), forwardAttentionHead, forwardMultiHeadAttention, newRandomAttentionHead, newRandomMultiHeadAttentionHead) where

import Matrices
import Mlp
import System.Random (StdGen, splitGen)

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
forwardAttentionHead :: AttentionHead -> [TokensEmbeddings] -> Maybe [TokensEmbeddings]
forwardAttentionHead layer embs = do
  keys <- forwardLinearBatch (w_keys layer) embs
  values <- forwardLinearBatch (w_vals layer) embs
  queries <- forwardLinearBatch (w_queries layer) embs
  let dot_prods = [multMatrices query (transpose key) | (query, key) <- zip queries keys]
  dps <- sequence dot_prods
  dk <- getDk layer
  let energies = [divideMatrix dp (sqrt (fromIntegral dk)) | dp <- dps]
  ens <- sequence energies
  let softmaxed = [linewiseSoftMax energy | energy <- ens]
  sfs <- sequence softmaxed
  let results = [multMatrices sf value | (sf, value) <- zip sfs values]
  out <- sequence results
  Just out

forwardMultiHeadAttention :: MultiHeadAttention -> [TokensEmbeddings] -> Maybe [TokensEmbeddings]
forwardMultiHeadAttention layer embs = do
  let all_heads_results = [forwardAttentionHead atthead embs | atthead <- heads layer]
  head_results <- sequence all_heads_results
  let concated_head_results = [concatMatrices [head_results !! head_number !! token_number | head_number <- [0 .. length (heads layer)]] | token_number <- [0 .. length embs - 1]]
  res <- sequence concated_head_results
  Just res
