def before_gpu_init(func):
    """
    Decorator to throw an error if a function is called when gpu sim has been initialized already.
    Used for functions such as setting friction values which currently cannot be changed once the
    gpu simulation has started.
    """

    def wrapper(self, *args, **kwargs):
        assert not self.sim._gpu_sim_initialized, (
            f"{func} can only be called when the GPU simulation has not been initialized "
            "yet"
        )
        return func(self, *args, **kwargs)

    return wrapper
