require Hok

Hok.defmodule_rts Saxpy do

defk saxpy_kernel(a,b,c,n) do
   index = blockIdx.x * blockDim.x + threadIdx.x;
   stride = blockDim.x * gridDim.x;

  for i in range(index,n,stride) do
     c[i] = 2 * a[i] + b[i]
   end
 end
end

Hok.include_rts [Saxpy]

n = 10000000

#list = [Enum.to_list(1..n)]

#mat1 = Matrex.new(list)
#mat2 = Matrex.new(list)

a = Nx.tensor(Enum.to_list(1..n),type: {:f, 32})
b = Nx.tensor(Enum.to_list(1..n),type: {:f, 32})

gnx1= Hok.new_gnx(a)
gnx2 = Hok.new_gnx(b)
gnxr= Hok.new_gnx(n, {:f,32})

threadsPerBlock = 128;
numberOfBlocks = div(n + threadsPerBlock - 1, threadsPerBlock)

Hok.spawn_rts(&Saxpy.saxpy_kernel/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[gnx1,gnx2,gnxr,n])


result = Hok.get_gmatrex(gnxr)

Hok.end_hok 


IO.inspect result
