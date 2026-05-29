# Filter

ObsPy-backed signal filtering. Every method returns a **new** DataFrame and
never mutates `Record.df`; chain filters by passing the previous result via
`df=`.

::: apeQuake.filters.filters.Filter
