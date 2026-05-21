# MATLAB Golden Fixture Checklist

Before retiring MATLAB, run the V4 scripts and export:

- match/alliance tables
- team index maps
- split masks
- target stats
- baseline coefficients, predictions, and metrics
- initialized model weights
- fixed-batch forward outputs
- epoch-1 shuffled batch order
- loss components
- evidence packet CSV schemas

Use `scipy.io.loadmat`, `pandas.testing`, and
`numpy.testing.assert_allclose` from Python tests to compare against these
fixtures.
