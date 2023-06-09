#!/usr/bin/env bash

dset_dir=/srv/beegfs/scratch/shares/dmml/eo4eu/datasets
shre_dir=/share/users/${SLURM_JOB_USER:0:1}/${SLURM_JOB_USER}
beg_lock="${shre_dir}/beg_lock"
end_lock="${shre_dir}/end_lock"

# check if copy is already completed
[[ -f "${end_lock}" ]] && exit 0

# check if copy has started; if not, create a lock file with my name in it
[[ ! -f "${beg_lock}" ]] && echo "$SLURM_JOB_ID" > "${beg_lock}"

echo "$(cat "${beg_lock}")"
echo "$SLURM_JOB_ID"

# check whether I'm the one who created the `beg_lock` file
if [[ "$(cat "${beg_lock}")" == "$SLURM_JOB_ID" ]]; then
    # if my name is in it, it means that the copy has not started
    # and more importantly that I am responsible for the copy

    # copy and uncompress the dataset at the location
    echo "cp"
    time cp ${dset_dir}/BigEarthNet-S2-v1.0.tar.gz ${shre_dir}
    echo "tar"
    time tar -xzf ${shre_dir}/BigEarthNet-S2-v1.0.tar.gz -C ${shre_dir}
    # create the `end_lock` file with my name in it
    echo "$SLURM_JOB_ID" > "${end_lock}"
else
    # if my name is not in it, it means that until there is a `end_lock` file,
    # the copy is still in progress, so I must wait for this file to appear
    while [[ ! -f "${end_lock}" ]]; do
        echo "i sleep"
        sleep 60
    done
fi

# say goodbyes
echo "we done here. bye."
