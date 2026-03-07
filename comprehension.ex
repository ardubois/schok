require Hok
Hok.defmodule_rts Comp do

  #defh soma(x,y) do
  #  x + y
  #end
  deft map_kernel (arr a) ~> (arr b) ~> integer ~> [a ~> b] ~> unit
  defk map_kernel(a1,r,size,f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    if(id < size) do
      r[id] = f(a1[id])
    end
  end

  def map(array,func) do

    array_gpu = Hok.new_gnx(array)

    shape = Hok.get_shape_gnx(array_gpu)
    type = Hok.get_type_gnx(array_gpu)
    size = Tuple.product(shape)
    result_gpu = Hok.new_gnx(shape, type)

    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)
    Hok.spawn_rts(&Comp.map_kernel/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[array_gpu,result_gpu,size,func])

   # Comp.map(array_gpu, result_gpu, size,func)

    result_gpu
    
  end

def replicate(n, x), do: (for _ <- 1..n, do: x)
end

#Hok.include_rts [Comp]
Hok.include_rts {Comp, %{ a: :float, b: :float }}
#Hok.set_default_type(%{default:  :float})
#Hok.include_rts {Comp, %{ default: :float}}

size = 10000

array = Nx.tensor(Enum.to_list(1..size),type: {:f, 32})


prev = System.monotonic_time()

#result = Comp.comp(array, Hok.hok (fn (a) ->  a + 10.0 end))

result = Hok.gpufor x<- array,  do: x + 10

Hok.end_hok

next = System.monotonic_time()

IO.puts "Hok\t#{size}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"

IO.inspect result
