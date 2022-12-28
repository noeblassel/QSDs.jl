#!/libre/blasseln/julia-1.8.2/bin/julia

using Plots, ProgressMeter

if !isdir(joinpath("figures/dirichlet_eigen",dir))
    mkdir(joinpath("figures/dirichlet_eigen",dir))
end

dir = ARGS[1]
node = "clustern14"
path = "/libre2/blasseln/QSD_data/dirichlet_data"

scp_cmd(file) = `scp $(node):$(path)/$(file) .`
run(pipeline(`ssh $(node) -C "ls $(path)/$(dir)"`,stdout="flist.txt")) # get file list

run(scp_cmd("potential.out"))
include("potential.out")
files=readlines(flist.txt)

datapoint_regex=r"beta(.+)_N(\d+)_ix(\d+).out"



@showprogress for f in files
    m_dp = match(datapoint_regex,f)

    if m_dp!==nothing

    end

end

rm("potential.out")