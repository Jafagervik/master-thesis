class Trainer:
    def __init__(
        self,
        model: BaseAE,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[LR_Scheduler] = None,
        **kwargs):
        # initialize class properties

    @TinyJit
    @Tensor.train()
    def train_step(self, input_batch: Tensor):
        self.optim.zero_grad()
        loss = self.model.criterion(input_batch)
        loss.backward()
        self.optim.step()
        return loss

    @TinyJit
    def validate_step(self, input_batch: Tensor):
        Tensor.no_grad = True
        loss = self.model.criterion(input_batch)
        Tensor.no_grad = False
        return loss

    def run_epoch(self, dataloader: DataLoader, step_fn):
        total_loss = 0.0
        num_batches = 0
        gflops = 0
        batch_time = time.perf_counter()

        for batch in dataloader:
            step_start_time = time.perf_counter()
            GlobalCounters.reset()
            loss = step_fn(batch)
            step_end_time = time.perf_counter() - step_start_time
            gflops += GlobalCounters.global_ops / (1e9 * step_end_time)
            total_loss += loss.float().item() 
            num_batches += 1

        batch_end_time = time.perf_counter() - batch_time
        gflops /= num_batches
        avg_loss = total_loss / num_batches  
        return avg_loss, gflops, batch_end_time

    def train(self):
        print(colored(f"Starting training {self.model.__class__.__name__} with {self.epochs} epochs", 'yellow'))
 
        for epoch in range(self.epochs):
            train_loss, train_gflops, train_time = self.run_epoch(self.train_dataloader, self.train_step)
            self.train_losses.append(train_loss)
        
            val_loss, val_gflops, val_time = self.run_epoch(self.val_dataloader, self.validate_step)
            self.val_losses.append(val_loss)
        
            printing(epoch, self.epochs, train_loss, val_loss, train_gflops, val_gflops, self.optim.lr.float().item(), train_time, val_time)

            self.scheduler.step(val_loss)
        
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.model.save()

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break