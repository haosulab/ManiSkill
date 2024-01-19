# Structs for managing CPU and GPU data from SAPIEN

This folder defines a number of dataclasses (known as structs, similar to [flax structs](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html)) that wrap around various CPU based classes to make them into pytrees, and work neatly with jax transformations and pytorch.