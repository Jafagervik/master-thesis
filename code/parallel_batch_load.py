class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, devices: List[str],
        num_workers: int = 1,
        shuffle: bool = False):
        self.indices = list(range(len(dataset)))
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        if self.shuffle: rnd.shuffle(self.indices)
        return self

    def __len__(self) -> int: return len(self.dataset)

    def __next__(self) -> Tensor:
        if self.current_index >= len(self.indices): raise StopIteration
        end_index = min(self.current_index + self.batch_size, len(self.indices))
        batch_indices = self.indices[self.current_index:self.current_index + self.batch_size]
        batch_data = self._load_batch_data(batch_indices)
        
        self.current_index = end_index
        batch_tensor = Tensor.stack(*batch_data, dim=0)
        if len(self.devices) == 1: batch_tensor.shard(self.devices, axis=0)
        else: return batch_tensor

    def _load_batch_data(self, batch_indices: List[int]) -> List[Tensor]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(lambda x: self.dataset[x], idx) for idx in batch_indices]
            batch_data = [future.result().realize() for future in as_completed(futures)]
        return batch_data