#!/libre/blasseln/julia-1.8.2/bin/julia

dir = ARGS[1]
βmin,dβ,βmax = parse.(Float64,ARGS[2:4])
ix_start,ix_end = parse.(Int64,ARGS[5:6])
N = parse(Int64,ARGS[7])
output_file_prefix = ARGS[8]
output_dir = ARGS[9]

println("usage: dir βmin dβ βmax ixstart ixend N output_file_prefix output_dir")

h = 1/N

for β=βmin:dβ:βmax
    println(β)
    l_normal_derivative_u1 = Float64[]
    r_normal_derivative_u1 = Float64[]

    l_normal_derivative_u2 = Float64[]
    r_normal_derivative_u2=Float64[]

    for ix = ix_start:ix_end
        include(joinpath("/libre2/blasseln/QSD_data/dirichlet_data",dir,"beta$(β)_N$(N)_ix$(ix).out"))
        u1 = us[:,1]
        u2 = us[:,2]

        n = length(u1)

        push!(l_normal_derivative_u1,(u1[2]-u1[1])/h)
        push!(r_normal_derivative_u1,(u1[n-1]-u1[n])/h)

        push!(l_normal_derivative_u2,(u2[2]-u2[1])/h)
        push!(r_normal_derivative_u2,(u2[n-1]-u2[n])/h)
    end

    output_file = open(joinpath(output_dir,"$(output_file_prefix)_normal_derivatives_β$(β)_$(dir).out"))
    println(output_file,"l_normal_derivative_u1=",l_normal_derivative_u1)
    println(output_file,"r_normal_derivative_u1=",r_normal_derivative_u1)
    println(output_file,"l_normal_derivative_u2=",l_normal_derivative_u2)
    println(output_file,"r_normal_derivative_u2=",r_normal_derivative_u2)
    close(output_file)
end