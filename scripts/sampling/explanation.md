## node structure

- essential data from `get_next()`
  -  prompt, input_ids, key_values, score  
- tree-related data
  - parent, children, depth
- specialized data
  - visits: for calculating uct1, used in `search()`
  - is_terminal, terminal_rank: for writing and keeping track of results
  - active: for **caching** `get_next()` calls during rollout(); set to `False` during `rollout()`, `True` during `expand()`
  - `uct1()`: standard uct1 formula, only we set really low exploration weight to incentivize deeper branches


## algorithm

- ### main loop
  run-of-the-mill mcts main loop  

  ``` 
  create root node

  while not enough variants:
    best_leaf_node <- search() best leaf node from root
    
    if best_leaf_node contains terminal character: 
      write() results
      
    else:
      if best_leaf_node is new (hasn't been visited):
        rollout() from best leaf node
        update branch with backpropagate()
        
      else:
        expand() one layer from best_leaf_node
  ```

- ### search()
  greedy search for layer's best child (highest uct1), until arriving at valid leaf (no active children, not terminal)  
  
  ```
  while current_node is not leaf:
  
    if current_node has valid children:
      current_node <- child with highest uct1
    
    else:
      remove current_node from tree
      current_node <- search() from root
  
  return current_node
  ```

- ### rollout()
  from specified node, simulate downward until either reaching terminal node or max depth if set, then get the `score` for `backpropagate()`  
  **note**: for our use case, we immediately call `write()` on reaching terminal nodes in simulation  
  **note**: simulated nodes are made with `get_next()` calls, we keep them but set `active` to `False` to prevent `search()` from finding them  
  **note**: it's actually best to limit max depth, because:
  1. deep simulation calls `get_next()` a lot
  2.  calling `backpropagate()` more often helps finding better branches  
    <br>  
    
  ```
  while node haven't reached max depth:
    simulate a child one level down (inactive child)
    
    if the child is terminal:
      write() child to results
      return child.score
    
    else:
      node <- child
  ```
  
- ### backpropagate()
  update from rollout node upward to root  
  things updated: `visits` and `score`
  ```
  while node has parents (root doesn't):
    node.visits += 1
    node.score += score
    node <- node.parent
  ```

- ### expand()
  create children one level down for specified node, if there were already inactive children, activate them  
  **note**: like rollout(), we immediately call `write()` on reaching terminal nodes upon creation
  ```
  while current_node doesn't have enough children:
    create a child
    
    if child is terminal:
      write() child to results

  activate all children
  ```
  
- ### write()
  this is where each result entry is written, so we can also decide what gets written, as of now we:  
  1. check if we got dupe results
  2. if the results are good enough (check w/ `score`)  
   
  we can also use this to make sure we don't get stuck finding the same results by repeatedly decreasing `score` on each repeated visit to terminal nodes  
  **note**: we do not simply remove used terminal nodes as calling `backpropagate()` with their penalized `score` helps improve result
  ```
  if node.score is good enough:
    results += [node]
  
  decrease node.score and backpropagate()  
  ```