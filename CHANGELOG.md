
# v0.0.6

-   fixed \`minimize\` api issue
-   fixed meaningless warning in line mod when no line search


# <span class="timestamp-wrapper"><span class="timestamp">[2019-01-06 Sun] </span></span> v0.0.5

-   added \`CachedProblem\` struct to cache function evaluations.
-   improved the performance of VelocityVerlet MD scheme
-   removed ForwardEuler scheme from MD options for its bad performance.
-   new parameter \`dt\_min\` to avoid stagnation in minimization.
-   new builder option: with\_line\_search()


# <span class="timestamp-wrapper"><span class="timestamp">[2019-01-02 Wed] </span></span> v0.0.4

-   simplify the callback function in LineSearch.
-   add options to set MD integration scheme.


# <span class="timestamp-wrapper"><span class="timestamp">[2018-12-27 Thu] </span></span> v0.0.3

-   add line mod for line search


# <span class="timestamp-wrapper"><span class="timestamp">[2018-12-26 Wed] </span></span> v0.0.2

-   fix a bug in MD step
-   add lj mod for testing

