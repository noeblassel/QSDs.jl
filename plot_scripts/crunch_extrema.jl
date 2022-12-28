#!/libre/blasseln/julia-1.8.2/bin/julia

using ProgressMeter


println("Usage: dir βmin dβ βmax N istart iend")
dir = ARGS[1]
βmin,dβ,βmax = parse.(Float64,ARGS[2:4])
N = parse(Int64,ARGS[5])
istart = parse(Int64,ARGS[6])
iend = parse(Int64,ARGS[7])

path = "/libre2/blasseln/QSD_data/dirichlet_data"
include(joinpath(path,dir,"potential.out"))
qs=Float64[]
output_filename="extrema_$(dir).out"
output_file=open(output_filename,"w")
println(output_file,"β ix argmins mins argmaxs maxs")
@showprogress for β=βmin:dβ:βmax
    for ix=istart:iend
        include(joinpath(path,dir,"beta$(β)_N$(N)_ix$(ix).out"))
        min_ixs = [i for i=2:length(qsd)-1 if (qsd[i]< qsd[i-1])&&(qsd[i]<qsd[i+1])]
        argmins = domain[min_ixs]
        mins = qsd[min_ixs]

        max_ixs = [i for i=2:length(qsd)-1 if (qsd[i] > qsd[i-1])&&(qsd[i] > qsd[i+1])]
        argmaxs = domain[max_ixs]
        maxs = qsd[max_ixs]

        println(output_file, "$(β) $(ix) $(argmins) $(mins) $(argmaxs) $(maxs)")
    end
end
close(output_file)