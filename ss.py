from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()    
size = comm.Get_size()    

if rank == 0:
    numbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    # Split the list into chunks for each process
    chunks = [numbers[i::size] for i in range(size)]
else:
    numbers = None
    chunks = None

chunk = comm.scatter(chunks, root=0)

# Perform local calculations on each process's chunk
local_sum = sum(chunk)
local_product = 1
for num in chunk:
    local_product *= num
local_min = min(chunk)
local_max = max(chunk)

global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
global_product = comm.reduce(local_product, op=MPI.PROD, root=0)
global_min = comm.reduce(local_min, op=MPI.MIN, root=0)
global_max = comm.reduce(local_max, op=MPI.MAX, root=0)

if rank == 0:
    print(f"Global Sum: {global_sum}")
    print(f"Global Product: {global_product}")
    print(f"Global Min: {global_min}")
    print(f"Global Max: {global_max}")
