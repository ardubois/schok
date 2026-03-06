require Hok

Hok.defmodule_rts Mapp do
      deft map_2kernel (arr a) ~> (arr b) ~> (arr c) ~> integer ~> [a ~> b ~> c] 
      defk map_2kernel(a1,a2,a3,size,f) do
        id = blockIdx.x * blockDim.x + threadIdx.x
        if(id < size) do
          a3[id] = f(a1[id],a2[id])
        end
      end
      def map2(t1,t2,func) do
    
        shape = Hok.get_shape_gnx(t1)
        type = Hok.get_type_gnx(t2)
         size = Tuple.product(shape)
         result_gpu = Hok.new_gnx(shape, type)
    
          threadsPerBlock = 256;
          numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)
    
          Hok.spawn_rts(&Mapp.map_2kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[t1,t2,result_gpu,size,func])
    
    
          result_gpu
      end
end   


#Hok.include_rts [Mapp]
Hok.include_rts {Mapp, %{ a: :float, b: :float, c: :float }}
#Hok.set_default_type(%{default:  :float})
#Hok.set_default_type(%{default:  :float, a: :float, b: :float, c: :float })

n = 10000000

#list = [Enum.to_list(1..n)]

#mat1 = Matrex.new(list)
#mat2 = Matrex.new(list)
a = Hok.new_nx_from_function(1,n,{:f,32},fn -> 1 end )
b = Hok.new_nx_from_function(1,n,{:f,32},fn -> 1 end )

gnx1= Hok.new_gnx(a)
gnx2 = Hok.new_gnx(b)

c = Mapp.map2(gnx1,gnx2, Hok.hok_rts fn (x,y) -> x+y end)

result = Hok.get_gnx(c)

Hok.end_hok 


IO.inspect result