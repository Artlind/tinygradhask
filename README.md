# 🧠 Haskell Nanograd

Micrograd in haskell — inspired by [Karpathy’s Micrograd](https://github.com/karpathy/micrograd).

Allows to **build computation graphs**, **run reverse-mode autodiff**, and **train simple MLPs** — all from scratch, with no external ML frameworks.

---

## Features

- **Reverse-mode autodiff** (backpropagation)
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
    a = createNombre ("a", 2)
    b = createNombre ("b", 3)
    f = createNombre ("f", 4)
    c = newNombreWithId ("c", a * b)
    d = newNombreWithId ("d", c * c)
    e = newNombreWithId ("e", d / f)

    graph = Graph (HM.fromList [(nombre_id node, node) | node <- [a, b, c, d, e, f]])
    backwarded_graph = backward "e" graph

    -- Extract gradients
    computed_a_grad = grad $ getNombreFromId "a" backwarded_graph
    computed_b_grad = grad $ getNombreFromId "b" backwarded_graph
    computed_c_grad = grad $ getNombreFromId "c" backwarded_graph
    computed_d_grad = grad $ getNombreFromId "d" backwarded_graph
    computed_f_grad = grad $ getNombreFromId "f" backwarded_graph

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
