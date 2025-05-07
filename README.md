# The Witness Solver

**Disclaimer:  This solver is for those moments of extreme frustration when you've already spent way too long trying to solve a puzzle in The Witness.  We've all been there.**

This project aims to automatically solve puzzles from the game *The Witness*.  The idea for this project came from a desire to combine two passions: video games and coding.

## Puzzle Types

Here's a list of puzzle types and their current completion status:

| Puzzle Type | Working On |  Completed |
|---|---|---|
| Basic Panels (No Symbols) | [X] | [ ] |
| Hexagons | [ ] | [ ] |
| Black & White Squares | [ ] | [ ] |
| Multicolor Squares | [ ] | [ ] |
| Tetris | [ ] | [ ] |
| Stars | [ ] | [ ] |
| Triangles | [ ] | [ ] |
| Elimination Marks | [ ] | [ ] |
| Dots | [ ] | [ ] |
| Combinations | [ ] | [ ] |
|---|---|---|
| Symmetry | [ ] | [ ] |


**Note:** Some puzzles in *The Witness* are "environmental" puzzles. These puzzles are integrated into the game's environment and require the player to observe and interact with the surroundings in a specific way.  This type of solver, which typically processes image input of the puzzle board, cannot solve environmental puzzles.


Here's a breakdown of the logic behind each puzzle type:

* **Basic Panels (No Symbols):** These are the simplest puzzles, requiring you to draw a continuous line from the start to the end point. The path cannot cross itself.
* **Hexagons:** These puzzles involve routing the path through all the hexagon tiles.
* **Black & White Squares:** These puzzles involve separating the path to go through white and black squares in specific ways, often requiring alternating path segments.
* **Multicolor Squares:** Similar to Black & White Squares, but with more than two colors, adding complexity to the path separation rules.
* **Tetris:** These puzzles feature Tetris-like shapes that must be placed in designated areas of the grid, with the path often needing to navigate around them.
* **Stars:** These puzzles involve pairing star symbols. The path must separate each star from its pair.
* **Triangles:** These puzzles involve directing the path to pass through a specific number of triangles. Each triangle indicates how many edges of that tile must be used by the path.
* **Elimination Marks:** These puzzles feature special symbols that negate other symbols on the board, adding a layer of logical deduction to the pathfinding.
* **Dots:** These puzzles require the path to pass through all the dots in a specific order or grouping.
* **Combinations:** This likely refers to puzzles that combine multiple puzzle elements, requiring the solver to handle several rulesets simultaneously.
* **Symmetry:** These puzzles require the path to exhibit symmetry, either rotational or reflective.


## How to Use

1.  Clone the repository: `git clone https://github.com/Opeiz/TheWitnessSolver.git`
<!-- 2.  Install the required dependencies: `pip install -r requirements.txt`
3.  Run the solver: `python solve.py <puzzle_image>` -->

## Sumamry of Features

1. For the moment i dont want to apply any AI solution, so the detection of the Start point and the End point it's clicked by the user. Its basic, i know, but works