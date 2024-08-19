class Trainer:
    def __init__(self, model: BaseAE, dataloader: DataLoader,
        optimizer: Optimizer, **kwargs): ...

    @TinyJit
    def train_step(self, x: Tensor, f: Callable[[Tensor], Tensor]) -> Tensor:
        self.optim.zero_grad()
        loss = self.model.criterion(f(x))["tot_loss"]
        loss.backward()
        self.optim.step()
        return loss.realize()

    def train(self):
        Tensor.training = True

        print(colored(f"Starting training {self.model.__class__.__name__} with {self.epochs} epochs", 'yellow'))
        reshape_fn = lambda x: x.reshape(-1, 1, self.shape[0], self.shape[1]) if self.model.convolutional else x.reshape(-1, self.shape[0] * self.shape[1])
 
        for epoch in range(self.epochs):
            GlobalCounters.reset()
            running_loss = 0.0
            with Timer() as t:
                for data in self.dataloader:
                    running_loss += self.train_step(data, reshape_fn).item()
                    
            self.losses[epoch] = running_loss
            printing(epoch, self.epochs, running_loss, t.interval)

            if running_loss < self.best_loss:
                self.best_loss = running_loss
                save_model(self.model)

            self.early_stopping(running_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                self.losses = self.losses[:epoch + 1]
                break
    ...
