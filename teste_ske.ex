require Ske

[arg] = System.argv()


m = String.to_integer(arg)


Hok.include_rts [Ske]
Hok.set_default_type(%{default:  :float, a: :float, b: :float})

vet1 = Hok.new_nx_from_function(1,m,{:f,32},fn -> 1.0 end)

ref1 = Hok.new_gnx(vet1)

resp = Ske.map(ref1, Hok.hok_rts fn a -> a+1 end)

Hok.end_hok

IO.inspect resp