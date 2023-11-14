import multiprocessing
import time

class ParallelProcessing:
    def run_parallel(self, function, arguments):
        # Create a pool of worker processes
        pool = multiprocessing.Pool()

        # Apply the function to each argument in parallel
        results = pool.map(function, arguments)

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

        return results


if __name__ == "__main__":
    def square(number):
        return number ** 2

    # Example usage
    parallel_processing = ParallelProcessing()
    numbers = [1, 2, 3, 4, 5]
    results = parallel_processing.run_parallel(square, numbers)
    print(results)