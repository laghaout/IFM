
# Table of Contents

1.  [Observations ⅓ ½ ⅔ <code>[2/3]</code>](#org3903dc1)
        1.  [More&#x2026;](#org33c0754)
2.  [Review <code>[0/4]</code>](#orgba5a056)
3.  [To-do <code>[1/13]</code>](#org07fcefa)

Saturday, May 25, 2024 at 11:40


<a id="org3903dc1"></a>

# Observations ⅓ ½ ⅔ <code>[2/3]</code>

-   [X] 1 bomb, N-1 clear:
    -   Main mode:
        -   Steering towards 0. Weaker steering as N increases.
        -   Maintaining a unit purity.
    -   Dark mode: Total collapse (steering?) onto 1 (and therefore unit purity)
-   [ ] 2 identical bombs, N-2 clear:
    -   Main mode:
        -   Steering towards 0. Weaker steering as N increases.
        -   Losing purity.
    -   Dark mode: Steering towards 1 but losing purity with <some parity> collapsing totally onto 50/50.
-   [ ] N-2 identical bombs, 2 clear
-   [ ] N-1 identical bombs, 1 clear
-   [X] N identical bombs, 0 clear:
    -   Main mode:
        -   Steering towards 0. Weaker steering as N increases
        -   Losing purity. The purity loss is higher for (i) higher weight on 1 in the initial state, and for (ii) smaller N.
    -   Dark mode:
        -   Partial collapse towards 50/50 regardless of initial state. Total collapse onto 50/50 if N=2, gets weaker as N increases.
        -   Purity independent of the initial state. Increases with N.


<a id="org33c0754"></a>

### More&#x2026;

-   Complete collapse to 1 in the dark modes
-   Phase loss (i.e., mixing) in the auxiliary modes
-   The purity indeed matches the pairwise mixture?


<a id="orgba5a056"></a>

# Review <code>[0/4]</code>

1.  [ ] Sinha et al.
2.  [ ] Nested interferometers and Elitzur's lecture
3.  [ ] Asking photons where they have been
4.  [ ] Photons are lying about where they have been
5.  [ ] Photons are lying about where they have been, again
6.  [ ] The two archive papers mentioned by Larsson
7.  [ ] Hardy's paradox


<a id="org07fcefa"></a>

# To-do <code>[1/13]</code>

1.  [ ] `weight * prior * probability`
2.  [ ] Try the heuristic from page 58.
3.  [ ] Try a constrained solver from scipy. See ChatGPT from March 17.
4.  [X] sha256 to see the different states that are generated from the components
5.  [ ] Try other distance measures: Quantum Relative Entropy, Trace Distance, and Bures Distance. Only use them if available from QuTiP.
6.  [ ] What happens with N-choose-k, for k > 2 when doing the Born decomposition?
7.  [ ] In addition to `born` and `linear`, try heuristic decomposition I had on page 58.
8.  [ ] Check that it works for non-equal superpositions
9.  [ ] Check the `predicted_purity()` with `bee` and not just `eee`. Also, why is the "main" different? How can we predict its purity?
10. [ ] Verify all the math by hand with a general expression and write an alternative `=__call__()` class.
11. [ ] Check all the code
12. [ ] Order of the interactions
13. [ ] Graph over N
14. [-] Report data frame with
    -   [X] Purities
    -   [X] Probabilities
    -   [X] Diagonal measurements for the bombs
    -   [ ] Fidelity
    -   [ ] Hilbert-Schmidt inner product
15. [ ] Comment
16. [ ] Vanilla-ize. Rename \`ifm\` to \`IFM\` for the repo name?

