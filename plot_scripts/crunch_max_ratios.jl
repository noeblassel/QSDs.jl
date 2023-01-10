#!/libre/blasseln/julia-1.8.2/bin/julia

using Base.Threads

path = "/libre2/blasseln/QSD_data/dirichlet_data"

println("Usage: dir βmin dβ βmax N istart")
dir = ARGS[1]
βmin,dβ,βmax = parse.(Float64,ARGS[2:4])
N = parse(Int64,ARGS[5])
istart = parse(Int64,ARGS[6])

include(joinpath(path,dir,"potential.out"))

βrange=βmin:dβ:βmax

argmaxs_threaded = [zero(βrange) for i=1:nthreads()]
maxs_threaded = [zero(βrange) for i=1:nthreads()]

@threads for (i,β)=enumerate(βrange)
    lines=readlines(joinpath(path,dir,"eigen_β$(β)_N$(N).out"))
    ratios = parse.(Float64,split(match(r"ratios=\[(.+)\]",last(lines),","))
    imax = argmax(ratios)
    argmaxs_threaded[threadid()][i] = domain[istart + imax]
    maxs_threaded[threadid()][i] = maximum(ratios)
end

argmaxs = sum(argmaxs_threaded)
maxs = sum(maxs_threaded)

output_file=open("max_ratios_$(dir).out","w")
println(output_file,"argmaxs = ",argmaxs)
println(output_file,"maxs = ",maxs)
close(output_file)