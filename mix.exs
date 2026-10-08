defmodule Hok.MixProject do
  use Mix.Project

  def project do
    [
      app: :hok,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      #compilers:   Mix.compilers()++ [:my_task] ,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:matrex, "~> 0.6"},
      {:nx, "== 0.10.0"},
      {:exla, "== 0.10.0"}
    ]
  end
end
