Nx.global_default_backend(EXLA.Backend)
Nx.Defn.global_default_options(compiler: EXLA, client: :cuda)

defmodule BMP do
  @on_load :load_nifs
  def load_nifs do
    :erlang.load_nif("./priv/bmp_nifs", 0)
  end

  def gen_bmp_int_nif(_string, _dim, _mat) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def gen_bmp_float_nif(_string, _dim, _mat) do
    :erlang.nif_error(:nif_not_loaded)
  end

  def gen_bmp_int(string, dim, binary) do
    gen_bmp_int_nif(to_charlist(string), dim, binary)
  end
end

defmodule JuliaSetNx do
  @moduledoc """
  Elixir/Nx port of the CUDA Julia-set benchmark
  (mapgen2D_xy_1para_noret_ker + julia + julia_function).

  Structural correspondence with the CUDA source:

    CUDA                                 Nx
    ----                                 --
    julia(x, y, dim)                ->   julia(x, y, dim)              (defn, escape-time test)
    julia_function(ptr, x, y, dim)  ->   julia_function(x, y, dim)     (defn, builds RGBA per pixel)
    mapgen2D_xy_1para_noret_ker     ->   mapgen2d_xy_1para_noret_ker(x, y, dim, f)
    get_julia_function_ptr()        ->   &julia_function/3 (Elixir function capture, the "pointer")
    main()                          ->   run/1

  The CUDA kernel launches one thread per (x, y) pixel and has julia()
  `return` early the moment |z| escapes. Nx has no per-lane early exit,
  so the escape test is vectorized over the whole (dim, dim) grid at
  once: every pixel still runs all 200 iterations, but a `still_running`
  mask freezes a pixel's (ar, ai) the instant it escapes, so further
  iterations are no-ops for that pixel — same result, different
  execution model (SIMD-over-pixels instead of one CUDA thread each).
  """

  import Nx.Defn

  @iterations 200
  @scale 0.1
  @cr -0.8
  @ci 0.156
  @escape_radius_sq 1.0e3

  # ---------------------------------------------------------------
  # "julia": escape-time test, vectorized over the full (x, y) grid.
  # Mirrors the CUDA device function's math and 200-iteration bound
  # exactly; returns 1 where the point never escaped (CUDA's
  # `return 1`), 0 where it did (CUDA's early `return 0`).
  # ---------------------------------------------------------------
  defn julia(x, y, dim) do
    jx = @scale * (dim - x) / dim
    jy = @scale * (dim - y) / dim
    zero = Nx.broadcast(0, Nx.shape(x))

    {_ar, _ai, escaped} =
      while {ar = jx, ai = jy, escaped = zero}, _i <- 0..(@iterations - 1) do
        nar = ar * ar - ai * ai + @cr
        nai = ai * ar + ar * ai + @ci

        still_running = escaped == 0
        diverged = nar * nar + nai * nai > @escape_radius_sq
        newly_escaped = still_running and diverged

        escaped = Nx.select(newly_escaped, 1, escaped)
        # freeze (ar, ai) once escaped, same effect as CUDA's early `return`
        ar = Nx.select(still_running, nar, ar)
        ai = Nx.select(still_running, nai, ai)

        {ar, ai, escaped}
      end

    escaped == 0
  end

  # ---------------------------------------------------------------
  # "julia_function": builds the 4 RGBA channels for every pixel at
  # once from julia_value, mirroring the CUDA device function that
  # writes ptr[offset*4 + 0..3] = {255*juliaValue, 0, 0, 255} per pixel.
  # ---------------------------------------------------------------
  defn julia_function(x, y, dim) do
    julia_value = julia(x, y, dim)

    r = 255 * julia_value
    zeros = Nx.broadcast(0, Nx.shape(julia_value))
    full = Nx.broadcast(255, Nx.shape(julia_value))

    # stacked on the last axis so flattening (y, x, channel) matches
    # the CUDA layout exactly: ptr[(x + y*dim)*4 + channel]
    Nx.stack([r, zeros, zeros, full], axis: 2)
  end

  # ---------------------------------------------------------------
  # "mapgen2D_xy_1para_noret_ker": generic 2D map kernel driven by a
  # function argument, exactly like the CUDA kernel takes a
  # `void (*f)(int*, int, int, int)` device function pointer — here f
  # is applied over the whole (x, y) coordinate grid at once instead
  # of one CUDA thread per pixel.
  # ---------------------------------------------------------------
  defn mapgen2d_xy_1para_noret_ker(x, y, dim, f) do
    f.(x, y, dim)
  end

  @doc """
  Equivalent of CUDA's main(argc, argv): builds the (x, y) pixel grid
  for a DIM x DIM image, runs the map kernel through the function
  pointer, times the whole device-facing portion (allocation +
  kernel + transfer back, same as the CUDA cudaEvent bracket), and
  prints "NX\\tDIM\\ttime_ms".
  """
  def run(dim) do
    f = &julia_function/3

    start = System.monotonic_time()

    # cudaMalloc(&d_pixelbuffer, ...) equivalent: build the (x, y)
    # coordinate grid, one entry per pixel (CUDA: one thread per
    # pixel via blockIdx/threadIdx, here vectorized instead)
    x = Nx.iota({dim, dim}, axis: 1, type: {:s, 32})
    y = Nx.iota({dim, dim}, axis: 0, type: {:s, 32})

    # mapgen2D_xy_1para_noret_ker<<<grid, block>>>(d_pixelbuffer, DIM, DIM, f)
    pixelbuffer = mapgen2d_xy_1para_noret_ker(x, y, dim, f)

    # cudaMemcpy(h_pixelbuffer, d_pixelbuffer, ..., DeviceToHost)
    h_pixelbuffer = Nx.to_binary(pixelbuffer)

    stop = System.monotonic_time()
    time_ms = System.convert_time_unit(stop - start, :native, :microsecond) / 1000.0

    IO.puts("Nx\t#{dim}\t#{time_ms}")

    h_pixelbuffer
  end
end

# equivalent of: size_t usr_value = (size_t)atol(argv[1]);
[dim_arg | _] = System.argv()
dim = String.to_integer(dim_arg)

pixelbuffer = JuliaSetNx.run(dim)

BMP.gen_bmp_int("julia_set_nx.bmp", dim, pixelbuffer)