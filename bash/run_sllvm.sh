#!/bin/bash
start=$SECONDS 
DIR=$(pwd)
###############################################################
## Script for running stochastic lattice Lotka-Volterra model #
###############################################################
# Set parameters using flags 
while [[ $# -gt 0 ]]
    do 
    key="$1"
    case $key in 
        -ssh)           # Run parallel on multiple nodes through ssh
        SSH=true
        shift
        ;; 
        -n|--numseeds)  # Number of seeds
        NSEEDS="$2"
        shift 
        shift 
        ;;
        *)              # Unknown option
        shift 
        ;; 
    esac 
done 
# Specify action
DISTRIBUTE=false         # Distribute code among nodes, if using -ssh
EXECUTE=false            # Run the code in parallel using GNU parallel
GETDATA=true            # Retrieve data from the modes through rsync

## Extract variables
if [ $SSH ]; then 
    echo "Executing script using -ssh"
    # Extract nodes to run on from the relevant file
    . $DIR/bash/variables.sh -ssh
    # DISTRIBUTE
    if $DISTRIBUTE; then 
        echo "Distributing code on available nodes..."
        # Archive the codebase
        cd; echo $CODEDIR
        tar --exclude='out/*' --exclude='data/*' --exclude='population/*' --exclude='results/*' --exclude='supplementary/data/' --exclude='.git/*' -czf sllvm-fragmented.tar.gz sllvm-fragmented/
        # Copy the archived codebase
        for node in ${nodes[@]}; do 
            if [ "$node" != "$current" ]; then    # Do not copy to the current node
                echo $node; scp -q ${CODEDIR}.tar.gz ${node}:${LOCALDIR}
            fi 
        done 
        # Unpack the codebase
	echo "Unpacking..." 
        parallel --nonall -S $noboss_nodes_string 'tar -xzf {1}.tar.gz' ::: ${CODEDIR}
    fi 
else
    echo "Executing script locally"
    . $DIR/bash/variables.sh 
fi 

## DEFINE variables and sequences
#  Additionally store these variables in files for later use (e.g. analysis, plotting)
seeds=$(seq 1 1 $NSEEDS)
alpha=$(seq 1.1 0.1 3.0)
python -c 'import numpy as np; np.savetxt("H.txt", np.logspace(-2,0,25), fmt="%.4f")'
#H=(0.010000 0.012115 0.014678 0.017783 0.021544 0.026102 0.031623 0.038312 0.046416 0.056234 0.068129 0.082540 0.100000 0.121153 0.146780 0.177828 0.215443 0.261016 0.316228 0.383119 0.464159 0.562341 0.681292 0.825404 1.000000)
rho=(0.2)
#python -c 'import numpy as np; np.savetxt("lambda.txt", np.logspace(-3,0,25), fmt="%.4e")'
lambda=(0.05 0.025 0.0125)
mkdir -p $DATADIR
echo "${seeds[@]}" > $DATADIR/seeds.txt
echo "${alpha[@]}" > $DATADIR/alpha.txt
echo "${rho[@]}" > $DATADIR/rho.txt
echo "${H[@]}" > $DATADIR/H.txt
mapfile -t H < H.txt; mv H.txt $DATADIR
#mapfile -t lambda < lambda.txt; mv lambda.txt $DATADIR

if [ $SSH ]; then 
    ## EXECUTE Python script in parallel on all available CPU threads
    if $EXECUTE; then 
	echo "Executing code, #seeds $NSEEDS"
        parallel -S $nodes_string --sshdelay 0.1 --delay 0.1 "
        cd {1};
        python run_system.py --H {2} --alpha {3} --rho {4} --lambda {5} --seed {6};
        " ::: $CODEDIR ::: ${H[@]} ::: ${alpha[@]} ::: ${rho[@]} ::: ${lambda[@]} ::: ${seeds[@]}
    fi 
    ## RETRIEVE data 
    if $GETDATA; then 
        for node in ${noboss_nodes[@]}; do 
            echo $node; 
	    for h in ${H[@]}; do
                HDIR=${DATADIR}H$h/
                rsync -avz --include='*Lambda-1*.npy' --exclude='*' $node:${HDIR} ${HDIR}/
            done 
        done
    fi
fi 

# Print approximate total runtime
duration=$(( SECONDS - start ))
echo "Simulation finished after approx. $duration s."
