# cs5100sp18
CS5100 Foundation of AI assignment

## Goal
Run and compare all search strategies under a home map, and print out all solutions if found

## set up
Install packages: `python3 -m pip install bisect tabulate`

## run
To run: `python3 main.py`

## Result
| Search Strategy       | First Solution Path                                | Set of All Solution Paths                                    |
|:----------------------|:---------------------------------------------------|:-------------------------------------------------------------|
| Depth-first           | <o103, b3, b4, o109, o119, o123, o125, d2, d3, d1> | {<o103, b3, b4, o109, o119, o123, o125, d2, d3, d1>          |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, o125, d2, d3, d1>  |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, r123>              |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, storage>                 |
|                       |                                                    |  <o103, o109, o119, o123, o125, d2, d3, d1>                  |
|                       |                                                    |  <o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, o109, o119, storage>}                                |
| Breadth-first         | <o103, o109, o119, storage>                        | {<o103, o109, o119, storage>                                 |
|                       |                                                    |  <o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, o109, o119, o123, o125, d2, d3, d1>                  |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, storage>                 |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, r123>              |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, o125, d2, d3, d1>          |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, o125, d2, d3, d1>} |
| Lowest-cost-first     | <o103, o109, o119, storage>                        | {<o103, o109, o119, storage>                                 |
|                       |                                                    |  <o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, storage>                 |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, r123>              |
|                       |                                                    |  <o103, o109, o119, o123, o125, d2, d3, d1>                  |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, o125, d2, d3, d1>          |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, o125, d2, d3, d1>} |
| Depth-limited         | <o103, b3, b4, o109, o119, o123, r123>             | {<o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, o109, o119, storage>}                                |
| Iterative deepening   | <o103, o109, o119, storage>                        | {<o103, o109, o119, storage>                                 |
|                       |                                                    |  <o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, storage>                 |
|                       |                                                    |  <o103, o109, o119, o123, o125, d2, d3, d1>                  |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, r123>              |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, o125, d2, d3, d1>          |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, o125, d2, d3, d1>} |
| Heuristic depth-first | <o103, o109, o119, storage>                        | {<o103, o109, o119, storage>                                 |
|                       |                                                    |  <o103, o109, o119, o123, o125, d2, d3, d1>                  |
|                       |                                                    |  <o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, o125, d2, d3, d1>          |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, storage>                 |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, o125, d2, d3, d1>  |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, r123>}             |
| Greedy best-first     | <o103, o109, o119, o123, r123>                     | {<o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, o109, o119, storage>                                 |
|                       |                                                    |  <o103, o109, o119, o123, o125, d2, d3, d1>                  |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, o125, d2, d3, d1>          |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, r123>              |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, storage>                 |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, o125, d2, d3, d1>} |
| A*                    | <o103, o109, o119, o123, r123>                     | {<o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, o109, o119, storage>                                 |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, r123>              |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, storage>                 |
|                       |                                                    |  <o103, o109, o119, o123, o125, d2, d3, d1>                  |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, o125, d2, d3, d1>          |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, o125, d2, d3, d1>} |
| Bidirectional         | <o103, o109, o119, storage>                        | {<o103, o109, o119, storage>                                 |
|                       |                                                    |  <o103, o109, o119, o123, r123>                              |
|                       |                                                    |  <o103, b3, b4, o109, o119, storage>                         |
|                       |                                                    |  <o103, o109, o119, o123, o125, d2, d3, d1>                  |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, r123>                      |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, storage>                 |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, r123>              |
|                       |                                                    |  <o103, b3, b1, b2, b4, o109, o119, o123, o125, d2, d3, d1>  |
|                       |                                                    |  <o103, b3, b4, o109, o119, o123, o125, d2, d3, d1>}         |
