def _load_batch_data(self, batch_indices: List[int]) 
    -> List[Tensor]:
    with ThreadPoolExecutor(self.num_workers) as ex:
        futures = [ex.submit(self._load_single_data, idx) 
            for idx in batch_indices]

        batch_data = [future.result().realize() 
            for future in as_completed(futures)]
    
    return batch_data

def _load_single_data(self, idx: int) -> Tensor:
    return self.dataset[idx]