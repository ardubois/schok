Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA, client: :cuda)

defmodule DotProductNx do
  @moduledoc """
  Elixir/Nx port of the CUDA dot-product benchmark (map_2kernel + reduce_kernel).

  Structural correspondence with the CUDA source:

    CUDA                          Nx
    ----                          --
    map_2kernel(a1, a2, f)   ->   map_2kernel(a1, a2, f)   (defn, f applied elementwise)
    reduce_kernel(a, f)      ->   reduce_kernel(a, f)       (defn, f folded over a)
    anonymous_mult           ->   anonymous_mult (defn)
    anonymous_sum            ->   anonymous_sum  (defn)
    get_anonymous_*_ptr()    ->   &anonymous_*/2 (Elixir function capture, passed as "pointer")
    main()                   ->   run/1
  """

  import Nx.Defn

  # ---------------------------------------------------------------
  # "map_2kernel": elementwise binary op over two tensors, driven
  # by a function argument, exactly like the CUDA kernel takes a
  # `float (*f)(float, float)` device function pointer.
  # ---------------------------------------------------------------
  defn map_2kernel(a1, a2, f) do
    f.(a1, a2)
  end

  # ---------------------------------------------------------------
  # "reduce_kernel": fold a over f, seeded with the tensor's
  # first-element-compatible zero, mirroring the CUDA tree-reduction
  # kernel that starts from ref4[0] = 0 and repeatedly applies f.
  # ---------------------------------------------------------------
  defn reduce_kernel(a, f) do
    zero = Nx.tensor(0.0, type: Nx.type(a))
    Nx.reduce(a, zero, fn x, acc -> f.(x, acc) end)
  end

  # ---------------------------------------------------------------
  # The two "device functions" exposed as function pointers in the
  # CUDA program (anonymous_mult / anonymous_sum), here just plain
  # defn functions captured with &/2 and passed around as values.
  # ---------------------------------------------------------------
  defn anonymous_mult(a, b), do: a * b
  defn anonymous_sum(a, b), do: a + b

  @doc """
  Equivalent of CUDA's main(argc, argv): builds two random float
  arrays of size N, runs map then reduce through the function
  pointers, times the whole device-facing portion (allocation +
  transfer + kernels + transfer back, same as the CUDA cudaEvent
  timers bracket), and prints "NX\\tN\\ttime_ms".
  """
  def run(n) do
    # host arrays (CUDA: a = malloc(...); a[i] = rand();)
    a_host = for _ <- 1..n, do: :rand.uniform() * 32_767
    b_host = for _ <- 1..n, do: :rand.uniform() * 32_767

    f1 = &anonymous_mult/2
    f2 = &anonymous_sum/2

    start = System.monotonic_time()

    # cudaMalloc + cudaMemcpy HostToDevice equivalent: build device
    # tensors (EXLA backend places them on GPU if available)
    dev_a = Nx.tensor(a_host, type: {:f, 32})
    dev_b = Nx.tensor(b_host, type: {:f, 32})

    # map_2kernel<<<...>>>(dev_a, dev_b, dev_resp, N, f1)
    dev_resp = map_2kernel(dev_a, dev_b, f1)

    # reduce_kernel<<<...>>>(dev_resp, d_final, f2, N)
    d_final = reduce_kernel(dev_resp, f2)

    # cudaMemcpy DeviceToHost
    final = Nx.to_number(d_final)

    stop = System.monotonic_time()
    time_ms = System.convert_time_unit(stop - start, :native, :microsecond) / 1000.0

    IO.puts("Nx\t#{n}\t#{time_ms}")

    final
  end
end

# equivalent of: int N = atoi(argv[1]);
[n_arg | _] = System.argv()
n = String.to_integer(n_arg)

DotProductNx.run(n)