Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA, client: :cuda)

defmodule NearestNeighborNx do
  @moduledoc """
  Elixir/Nx port of the CUDA nearest-neighbor benchmark
  (map_step_2para_1resp_kernel + reduce_kernel, with euclid/menor
  as the two function pointers).

  Structural correspondence with the CUDA source:

    CUDA                                   Nx
    ----                                   --
    euclid(d_locations, lat, lng)     ->   euclid(loc_lat, loc_lng, par1, par2)  (defn, per-record distance)
    menor(x, y)                       ->   menor(x, y)                           (defn, min-with-sentinel fold)
    map_step_2para_1resp_kernel       ->   map_step_2para_1resp_kernel(loc_lat, loc_lng, par1, par2, f)
    reduce_kernel                     ->   reduce_kernel(a, f)
    get_euclid_ptr() / get_menor_ptr()->   &euclid/4 / &menor/2 (Elixir function captures, the "pointers")
    loadData()                        ->   built inline in run/1
    main()                            ->   run/1

  Note on `menor`: the CUDA version isn't a plain min — it treats an
  accumulator value of exactly 0.0 as "not set yet" and returns the
  new element unconditionally in that case (`if (y == 0.0) return x;`).
  This relies on the reduction seed (ref4[0]) starting at 0.0, and is
  fragile if a genuine distance is ever exactly 0.0 — but it's the
  actual behavior of the program being compared against, so it's
  reproduced exactly rather than "fixed" to a real min.
  """

  import Nx.Defn

  # ---------------------------------------------------------------
  # "euclid": per-record Euclidean distance from a fixed (lat, lng)
  # reference point to every location at once, vectorized over the
  # whole loc_lat/loc_lng arrays instead of CUDA's one-thread-per-
  # record launch (`map_step_2para_1resp_kernel<<<numRecords, 1>>>`).
  # ---------------------------------------------------------------
  defn euclid(loc_lat, loc_lng, lat, lng) do
    dlat = lat - loc_lat
    dlng = lng - loc_lng
    Nx.sqrt(dlat * dlat + dlng * dlng)
  end

  # ---------------------------------------------------------------
  # "menor": min-with-sentinel fold, same branching as the CUDA
  # device function — not a plain Nx.min.
  # ---------------------------------------------------------------
  defn menor(x, y) do
    min_val = Nx.select(x < y, x, y)
    Nx.select(y == 0.0, x, min_val)
  end

  # ---------------------------------------------------------------
  # "map_step_2para_1resp_kernel": generic map driven by a function
  # argument, exactly like the CUDA kernel takes a
  # `float (*f)(float*, float, float)` device function pointer.
  # step/pointer arithmetic (`d_array + id`) has no Nx equivalent —
  # here the per-record lat/lng columns are just passed directly.
  # ---------------------------------------------------------------
  defn map_step_2para_1resp_kernel(loc_lat, loc_lng, par1, par2, f) do
    f.(loc_lat, loc_lng, par1, par2)
  end

  # ---------------------------------------------------------------
  # "reduce_kernel": fold a over f, seeded with 0.0 — same seed value
  # the CUDA kernel implicitly relies on via ref4[0], and the same
  # generic reduce-through-a-function-pointer shape used in the
  # dot-product benchmark.
  # ---------------------------------------------------------------
  defn reduce_kernel(a, f) do
    zero = Nx.tensor(0.0, type: Nx.type(a))
    Nx.reduce(a, zero, fn x, acc -> f.(x, acc) end)
  end

  @doc """
  Equivalent of CUDA's main(argc, argv): builds numRecords random
  (lat, lng) pairs, computes each one's distance from (0.0, 0.0)
  through the map kernel, finds the minimum via the reduce kernel,
  times the whole device-facing portion (allocation + transfer +
  map + reduce + transfer back, same as the CUDA cudaEvent bracket),
  and prints "NX\\tnumRecords\\ttime_ms".
  """
  def run(n) do
    # loadData(locations, numRecords) equivalent:
    #   locations[0] = (7 + rand()%63) + fraction in [0,1)   -> lat
    #   locations[1] = (rand()%358)    + fraction in [0,1)   -> lng
    lat_host = for _ <- 1..n, do: (7 + :rand.uniform(63) - 1) + :rand.uniform()
    lng_host = for _ <- 1..n, do: :rand.uniform(358) - 1 + :rand.uniform()

    f1 = &euclid/4
    f2 = &menor/2

    start = System.monotonic_time()

    # cudaMalloc(&d_locations, ...) + cudaMemcpy HostToDevice equivalent
    loc_lat = Nx.tensor(lat_host, type: {:f, 32})
    loc_lng = Nx.tensor(lng_host, type: {:f, 32})

    # map_step_2para_1resp_kernel<<<numRecords, 1>>>(d_locations, d_distances, 2, 0.0, 0.0, numRecords, f1)
    distances = map_step_2para_1resp_kernel(loc_lat, loc_lng, 0.0, 0.0, f1)

    # reduce_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_distances, d_resp, f2, numRecords)
    nearest = reduce_kernel(distances, f2)

    # cudaMemcpy(resp, d_resp, sizeof(float), cudaMemcpyDeviceToHost)
    final = Nx.to_number(nearest)

    stop = System.monotonic_time()
    time_ms = System.convert_time_unit(stop - start, :native, :microsecond) / 1000.0

    IO.puts("Nx\t#{n}\t#{time_ms}")

    final
  end
end

# equivalent of: int numRecords = atoi(argv[1]);
[n_arg | _] = System.argv()
n = String.to_integer(n_arg)

NearestNeighborNx.run(n)