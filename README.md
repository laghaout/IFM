
# Table of Contents

1.  [Observations ⅓ ½ ⅔](#org071ca7b)
        1.  [More&#x2026;](#org20c6893)
2.  [Review <code>[0/4]</code>](#orge5b5e33)
3.  [To-do <code>[1/13]</code>](#org5b12948)


<a id="org071ca7b"></a>

# Observations ⅓ ½ ⅔

-   1 bomb, N-1 clear:
    -   Main mode: Steering towards 0 while maintaining a unit purity. Weaker steering as N increases.
    -   Dark mode: Total collapse onto 1
-   n identical bombs, N-n clear:
    -   Main mode: Steering towards 0 but losing purity.
    -   Dark mode: Steering towards 1 but losing purity with <some parity> collapsing totally onto 50/50.
-   N identical bombs
    -   Main mode:
        -   Steering towards 0 but losing purity.
        -   The purity loss is higher for (i) higher weight on 1 in the initial state, and for (ii) smaller N.
    -   Dark mode:
        -   Partial collapse towards 50/50 regardless of initial state. Total collapse onto 50/50 if N=2.
        -   Purity independent of the initial state.


<a id="org20c6893"></a>

### More&#x2026;

-   Complete collapse to 1 in the dark modes
-   Phase loss (i.e., mixing) in the auxiliary modes
-   The purity indeed matches the pairwise mixture?


<a id="orge5b5e33"></a>

# Review <code>[0/4]</code>

1.  [ ] Sinha et al.
2.  [ ] Nested interferometers and Elitzur's lecture
3.  [ ] Asking photons where they have been
4.  [ ] Photons are lying about where they have been
5.  [ ] Photons are lying about where they have been, again
6.  [ ] The two archive papers mentioned by Larsson
7.  [ ] Hardy's paradox


<a id="org5b12948"></a>

# To-do <code>[1/13]</code>

1.  [ ] Try the heuristic from page 58.
2.  [ ] Try a constrained solver from scipy. See ChatGPT from March 17.
3.  [X] sha256 to see the different states that are generated from the components
4.  [ ] Try other distance measures: Quantum Relative Entropy, Trace Distance, and Bures Distance. Only use them if available from QuTiP.
5.  [ ] What happens with N-choose-k, for k > 2 when doing the Born decomposition?
6.  [ ] In addition to `born` and `linear`, try heuristic decomposition I had on page 58.
7.  [ ] Check that it works for non-equal superpositions
8.  [ ] Check the `predicted_purity()` with `bee` and not just `eee`. Also, why is the "main" different? How can we predict its purity?
9.  [ ] Verify all the math by hand with a general expression and write an alternative `=__call__()` class.
10. [ ] Check all the code
11. [ ] Order of the interactions
12. [ ] Graph over N
13. [-] Report data frame with
    -   [X] Purities
    -   [X] Probabilities
    -   [X] Diagonal measurements for the bombs
    -   [ ] Fidelity
    -   [ ] Hilbert-Schmidt inner product
14. [ ] Comment
15. [ ] Vanilla-ize. Rename \`ifm\` to \`IFM\` for the repo name?

