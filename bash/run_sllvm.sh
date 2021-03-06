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
DISTRIBUTE=true         # Distribute code among nodes, if using -ssh
EXECUTE=true            # Run the code in parallel using GNU parallel
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
# alpha=$(seq 1.01 0.01 1.01; seq 1.05 0.05 2.0; seq 2.1 0.1 4.0)
# alpha=(-1 1.01)
# alpha=(1.1 1.2 1.5 2.0 2.5 3.0)
alpha=$(seq 1.1 0.1 3.0)
H=$(seq 0.0100 1 0.0100; seq 0.0500 0.0500 1)
# H=(0.0100 0.5000 0.9999)
# sigma=$(seq 0.05 0.05 1)
# sigma=(0.05 0.10 0.20 0.50 1.00)
# python -c 'import numpy as np; np.savetxt("H.txt", np.logspace(-2,0,25), fmt="%.4f")'
# python -c 'import numpy as np; np.savetxt("lambda.txt", np.logspace(-3,0,25), fmt="%.4e")'
mkdir -p $DATADIR
echo "${seeds[@]}" > $DATADIR/seeds.txt
echo "${alpha[@]}" > $DATADIR/alpha.txt
# echo "${sigma[@]}" > $DATADIR/sigma.txt
echo "${H[@]}" > $DATADIR/H.txt
# mapfile -t H < H.txt; mv H.txt $DATADIR
# mapfile -t lambda < lambda.txt; mv lambda.txt $DATADIR

if [ $SSH ]; then 
    ## EXECUTE Python script in parallel on all available CPU threads
    if $EXECUTE; then 
	echo "Executing code, #seeds $NSEEDS"
        parallel -j 30 -S $nodes_string --sshdelay 0.1 --delay 0.1 "
        cd {1};
        python run_system.py --H {2} --alpha {3} --seed {4};
        " ::: $CODEDIR ::: ${H[@]} ::: ${alpha[@]} ::: ${seeds[@]}
    fi 
    ## RETRIEVE data 
    if $GETDATA; then 
        for node in ${noboss_nodes[@]}; do 
            for h in ${H[@]}; do
                echo $node; 
                HDIR=${DATADIR}H$h/
                rsync -avz --include='*population*T10000*.npy' --exclude='*' $node:${HDIR} ${HDIR}/
            done 
        done
    fi
fi 

# Print approximate total runtime
duration=$(( SECONDS - start ))
echo "Simulation finished after approx. $duration s."
