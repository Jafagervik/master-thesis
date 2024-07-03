class BaseAE(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def convolutional(self) -> bool:
        """Helper property to help with how we should reshape before"""
        pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tuple[Tensor, ...]:
        pass

    @abstractmethod
    def criterion(self, x: Tensor) -> Dict[str, Tensor]:
        pass

    @TinyJit
    def predict(self, x: Tensor) -> Tensor:
        pass