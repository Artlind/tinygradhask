# Haskell Tinygrad

Tinygrad in haskell — inspired by [Karpathy’s Micrograd](https://github.com/karpathy/micrograd).

Allows to **build computation graphs**, **run reverse-mode autodiff**, and **train simple MLPs** — all from scratch, with no external ML frameworks.

---

## Features

- **Backprop**
- **Basic tensor-like operations** (scalars and simple 2D matrices)
- **Multi-Layer Perceptron (MLP)** model
- **Gradient-based optimization (`fitBatch`)**
- **Deterministic randomness** using `mkStdGen`
- **Educational design** — I used this project to learn Haskell.

---

## Core Concepts

The backend revolves around two main abstractions:

- **`Nombre`** — a node in a computation graph, holding:
  - a numeric value  
  - its gradient  
  - references to its parents and operation type  

- **`Graph`** — a mapping of `Nombre`s, used to perform backward passes.

---

## Example: Autodiff

```haskell
import Tinygrad

testMoreComplexBackward :: Bool
testMoreComplexBackward = passed
  where
    a, b, c, d, e, f :: Nombre
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    f = createNombre ("f", 4)
    c = newNombreWithId ("c", a * b)
    d = newNombreWithId ("d", c * c)
    e = newNombreWithId ("e", d / f)

    graph, backwarded_graph :: Graph
    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c, d, e, f]])
    backwarded_graph = backward "e" graph

    computed_a_grad, computed_b_grad, computed_c_grad, computed_d_grad, computed_f_grad :: Double
    computed_a_grad = grad $ getNombreFromId "a" backwarded_graph
    computed_b_grad = grad $ getNombreFromId "b" backwarded_graph
    computed_c_grad = grad $ getNombreFromId "c" backwarded_graph
    computed_d_grad = grad $ getNombreFromId "d" backwarded_graph
    computed_f_grad = grad $ getNombreFromId "f" backwarded_graph

    expected_a_grad, expected_b_grad, expected_c_grad, expected_d_grad, expected_f_grad :: Double
    expected_d_grad = 1 / value f
    expected_f_grad = -(value d / value f ** 2)
    expected_c_grad = expected_d_grad * 2 * value c
    expected_a_grad = value b * expected_c_grad
    expected_b_grad = value a * expected_c_grad

    correct_a_grad = computed_a_grad == expected_a_grad
    correct_b_grad = computed_b_grad == expected_b_grad
    correct_c_grad = computed_c_grad == expected_c_grad
    correct_d_grad = computed_d_grad == expected_d_grad
    correct_f_grad = computed_f_grad == expected_f_grad
    passed = correct_a_grad && correct_b_grad && correct_c_grad && correct_d_grad && correct_f_grad

```

This builds a computation graph for:

$$
e = \frac{(a \times b)^2}{f}
$$

and verifies that all gradients obtained from ```backward e``` match the analytical ones.

---

## Example: One Training Step of an MLP

```haskell
import Matrices
import Mlp
import System.Random (StdGen, mkStdGen)
import Data.Maybe (fromJust, isJust, isNothing)
import Tinygrad

testFitBatch :: Bool
testFitBatch = passed
  where
    shapes :: [Shape]
    shapes = [(3, 5), (5, 1)]
    ranges :: [Range]
    ranges = [(-1, 1), (-1, 1)]
    model :: Mlp
    model = fromJust $ newRandomMlp [(shape, range, key, True :: WithBias) | (shape, range, key) <- zip3 shapes ranges [mkStdGen 42, mkStdGen 43]]

    batch_size :: Int
    batch_size = 3
    shape_input, shape_output :: Shape
    shape_input = (batch_size, fst (head shapes))
    shape_output = (batch_size, snd (last shapes))
    range_input_and_labels :: Range
    range_input_and_labels = (-1, 1)
    key_input, key_output :: StdGen
    key_input = mkStdGen 42
    key_output = mkStdGen 44
    rand_input :: Matrix2d
    (rand_input, _) = fromJust $ randMatrix2d "rand_input" shape_input range_input_and_labels key_input
    rand_labels :: Matrix2d
    (rand_labels, _) = fromJust $ randMatrix2d "rand_labels" shape_output range_input_and_labels key_output
    initial_forward :: MlpOutput
    initial_forward = fromJust $ forwardMlp model rand_input
    old_squared_errors :: Matrix2d
    old_squared_errors = fromJust $ meanSquaredError (output_tensor initial_forward) rand_labels
    old_sumed_squared_errors :: Nombre
    old_sumed_squared_errors = sumNombre (allParamsFromMatrix old_squared_errors)

    lr = 0.01
    new_model :: Mlp
    new_model = fromJust $ fitBatch model (rand_input, rand_labels) lr
    new_forward :: MlpOutput
    new_forward = fromJust $ forwardMlp new_model rand_input
    new_squared_errors :: Matrix2d
    new_squared_errors = fromJust $ meanSquaredError (output_tensor new_forward) rand_labels
    new_sumed_squared_errors :: Nombre
    new_sumed_squared_errors = sumNombre (allParamsFromMatrix new_squared_errors)

    passed = value old_sumed_squared_errors > value new_sumed_squared_errors

```

## Build & test

```bash
git clone https://github.com/Artlind/tinygradhask.git
cd tinygradhask
cabal build
cabal test
```

## License

MIT License — feel free to use, learn from, and modify this project.

