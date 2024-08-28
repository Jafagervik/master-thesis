class BaseAE(ABC):
    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]: pass

    @abstractmethod
    def criterion(self, x: Tensor) -> Tensor: pass

    @TinyJit
    def predict(self, x: Tensor) -> Tensor: pass