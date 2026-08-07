package torchrec.data

/**
 * Stub factory for Java DataLoaders.
 *
 * NOTE: This is a stub that satisfies compilation after the upstream
 * org.bytedeco.pytorch 2.13.0-1.5.14-SNAPSHOT package dropped the
 * generic Example-based JavaDataset/JavaRandomDataLoader family.
 *
 * The Scala DataLoader type in torchrec.data is the canonical loader;
 * the factory methods here return DataLoader instances. Custom logic
 * from the original adapters can be reintroduced here when the upstream
 * JavaDataset support is restored.
 */
object JavaDataLoaderFactory {

  // Non-tensor (Example-based) DataLoaders — stubs
  def random(backing: Dataset, batchSize: Long = 256L, numWorkers: Long = 0L, dropLast: Boolean = false): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = true, numWorkers.toInt, dropLast)

  def sequential(backing: Dataset, batchSize: Long = 256L, numWorkers: Long = 0L, dropLast: Boolean = false): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast)

  def stateful(backing: Dataset, batchSize: Long = 256L, numWorkers: Long = 0L): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast = false)

  def stream(backing: Dataset, batchSize: Long = 256L, numWorkers: Long = 0L): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast = false)

  // Tensor-based DataLoaders — stubs
  def randomTensor(backing: Dataset, batchSize: Long = 256L, numWorkers: Long = 0L, dropLast: Boolean = false): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = true, numWorkers.toInt, dropLast)

  def sequentialTensor(backing: Dataset, batchSize: Long = 256L, numWorkers: Long = 0L, dropLast: Boolean = false): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast)

  def statefulTensor(backing: Dataset, batchSize: Long = 256L, numWorkers: Long = 0L): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast = false)

  def streamTensor(backing: Dataset, batchSize: Long = 256L, numWorkers: Long = 0L): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast = false)

  // Distributed DataLoaders — stubs
  def distributedRandom(backing: Dataset, rank: Int, worldSize: Int, batchSize: Long = 256L, numWorkers: Long = 0L, dropLast: Boolean = false): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = true, numWorkers.toInt, dropLast)

  def distributedSequential(backing: Dataset, rank: Int, worldSize: Int, batchSize: Long = 256L, numWorkers: Long = 0L, dropLast: Boolean = false): DataLoader =
    new DataLoader(backing, batchSize.toInt, shuffle = false, numWorkers.toInt, dropLast)
}