require PolyHok
#Nx.default_backend(EXLA.Backend)
#import Nx
PolyHok.defmodule DP do
#include CAS_Double
  defk map_2kernel(a1,a2,a3,size,f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    if(id < size) do
      a3[id] = f(a1[id],a2[id])
    end
  end
  def map2(t1,t2,func) do

    {l,c} = PolyHok.get_shape_gnx(t1)
    type = PolyHok.get_type_gnx(t2)
     size = l*c
     result_gpu = PolyHok.new_gnx({l,c}, type)

      threadsPerBlock = 256;
      numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

      PolyHok.spawn(&DP.map_2kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[t1,t2,result_gpu,size,func])


      result_gpu
  end
  def reduce(ref, initial, f) do

     threadsPerBlock = 256
    numberOfBlocks  = 64
    size = case PolyHok.get_shape_gnx(ref) do
        {l,c} -> l *c
        {size} -> size   
    end 
     type = PolyHok.get_type_gnx(ref)
      result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))
    d_partials = PolyHok.new_gnx( {numberOfBlocks} ,  type)
    
    PolyHok.spawn(&DP.reduce_kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,d_partials, f, initial, size])
    PolyHok.spawn(&DP.reduce_kernel/5,{1,1,1},{threadsPerBlock,1,1},[d_partials, result_gpu, f, initial, numberOfBlocks])

      result_gpu
  end

#eft reduce_kernel (arr a) ~> (arr a) ~> [a ~> a ~> a ] ~> a ~> integer
defk reduce_kernel(ra, rout, f ,identity, n) do
   
    __shared__ cache[256];
    cacheIndex = threadIdx.x;
    tid = threadIdx.x + (blockIdx.x * blockDim.x);

    temp = identity;
    while (tid < n) do
        temp = f(ra[tid], temp);
        tid = tid + (blockDim.x * gridDim.x);
    end
    cache[cacheIndex] = temp;
    __syncthreads();

    type i int
    i = blockDim.x / 2;
    while (i > 0) do
        if (cacheIndex < i) do
            cache[cacheIndex] = f(cache[cacheIndex + i], cache[cacheIndex]);
        end    
        __syncthreads();
        i = i / 2;
    end

    if (cacheIndex == 0) do
        rout[blockIdx.x] = cache[0];
    end    
 end
  def replicate(n, x), do: (for _ <- 1..n, do: x)
end

#PolyHok.include [DP]

[arg] = System.argv()

n = String.to_integer(arg)


#{vet1,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {1, n}, type: :f32)
#{vet2,_} = Nx.Random.uniform(Nx.Random.key(1), shape: {1, n}, type: :f32)

vet1 = PolyHok.new_nx_from_function(1,n,{:f,64},fn -> 1.0 end )
vet2 = PolyHok.new_nx_from_function(1,n,{:f,64},fn -> 0.1 end )

#vet1 = Nx.iota({1,n}, type: :f32)
#vet2 = Nx.iota({1,n}, type: :f32)

#vet1 = PolyHok.new_nx_from_function(1,n,{:f,32},fn -> :rand.uniform(1000) end )
#vet2 = PolyHok.new_nx_from_function(1,n,{:f,32},fn -> :rand.uniform(1000) end)

#vet1 = PolyHok.new_nx_from_function(1,n,{:f,32},fn -> 1.0 end )
#vet2 = Nx.tensor([Enum.to_list(1..n)], type: {:f,32})

prev = System.monotonic_time()


ref1 = PolyHok.new_gnx(vet1)

ref2 = PolyHok.new_gnx(vet2)


_result = ref1
    |> DP.map2(ref2, PolyHok.phok fn (a,b) -> a * b end)
    |> DP.reduce(0.0,PolyHok.phok fn (a,b) -> a + b end)
    |> PolyHok.get_gnx

#IO.inspect result

next = System.monotonic_time()


IO.puts "PolyHok\t#{n}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
