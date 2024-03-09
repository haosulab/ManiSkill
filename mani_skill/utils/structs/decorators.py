def before_gpu_init(func):
    """
    decorator to throw an error if a function is called when gpu sim has been initialized already. Used for functions such as setting friction values which currently
    cannot be changed once the gpu simulation has started.
    """

    def wrapper(self, *args, **kwargs):
        assert (
            self._scene._gpu_sim_initialized == False
        ), f"{func} can only be called when the GPU simulation has not been initialized yet"
        return func(self, *args, **kwargs)

    return wrapper
