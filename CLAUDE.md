## About this project
 
This is a learning project: I'm implementing classic machine learning models by hand (e.g. DBSCAN) with NumPy to understand how they work. Writing the model code myself is the whole point.
 
Tests are the exception. I don't learn much from writing them, so you write them.
 
## Rules
 
### Model code is off-limits
- Never write, edit, refactor or "fix" model implementation code, even when a test exposes a bug.
- Don't suggest code snippets for model logic either. If a test fails because of a bug in a model, explain in plain words what's wrong and where it probably is, so I can fix it myself.
- Keep those explanations short and objective.
### Tests are yours
- You can freely create and edit test files and test utilities.
- Use pytest.
- Where a scikit-learn equivalent exists, use it as the reference implementation and compare results against it.
- Also cover edge cases: empty input, a single point, all noise, all points in one cluster, duplicate points, different dimensionalities, and so on.
- Use fixed random seeds so tests are reproducible.
