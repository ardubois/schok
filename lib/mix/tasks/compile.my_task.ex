defmodule Mix.Tasks.Compile.MyTask do
    use Mix.Task
    @recursive true
    @moduledoc "Runs a task after all other compilation."
  
    def run(_args) do
      IO.puts("Compiling Generated CUDA code for Hok application!!!")
      # Your custom execution logic here
      :ok
    end
  end