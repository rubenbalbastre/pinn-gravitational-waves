@info "No multiprocessing time"

function compute()
    x::Int = 1000000
    arr = Array{Float64}(x)
    for i in range(1,x)
        arr[i] = i
    end
end

@time compute()

using Distributed;
using SharedArrays;
addprocs(4)

@everywhere using Pkg;
@everywhere Pkg.activate("."); 

function compute_mp()
    x::Int = 1000000
    arr = SharedArray{Float64}(x)
    @sync @distributed for i in range(1,x)
        arr[i] = i
    end
end

@info "Multiprocessing time"
@time compute_mp()
