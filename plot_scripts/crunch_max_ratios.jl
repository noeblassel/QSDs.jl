#!/libre/blasseln/julia-1.8.2/bin/julia 

path = "/libre2/blasseln/QSD_data/dirichlet_data"

println("Usage: dir βmin dβ βmax N istart")
dir = ARGS[1]
βmin,dβ,βmax = parse.(Float64,ARGS[2:4])
N = parse(Int64,ARGS[5])
istart = parse(Int64,ARGS[6])

include(joinpath(path,dir,"potential.out"))

βrange=βmin:dβ:βmax

argmaxs=zero(βrange)
maxs = zero(argmaxs)

for (i,β)=enumerate(βrange)
    lines=readlines(joinpath(path,dir,"eigen_β$(β)_N$(N).out"))
    ratios = parse.(Float64,split(match(r"ratios=\[(.+)\]",last(lines)),","))
    imax = argmax(ratios)
    argmaxs[i] = domain[istart + imax]
    maxs[i] = maximum(ratios)
end


output_file=open("max_ratios_$(dir).out","w")
println(output_file,"argmaxs = ",argmaxs)
println(output_file,"maxs = ",maxs)
close(output_file)