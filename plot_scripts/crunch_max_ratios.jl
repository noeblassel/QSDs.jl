#!/bin/env julia

dir = ARGS[1]
node = "clustern14"
path = "/libre2/blasseln/QSD_data/dirichlet_data"

scp_cmd(file) = `scp $(node):$(path)/$(file) tmp_potential.out`
run(pipeline(`ssh $(node) -C "ls $(path)/$(dir)"`,stdout="flist.txt")) # get file list

include("tmp_potential.out")

files=readlines(flist.txt)


# cleanup
rm("flist.txt")
rm("tmp_potential.out")
rm("tmp.out")