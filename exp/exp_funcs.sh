#!/bin/bash

function status {
    echo "[*] $1"
}

function positive {
    echo "[+] $1"
}

function negative {
    echo "[!] $1"
}

function exp_completed {
    positive "Experiment run completed."
    exit 0
}

function exp_failed {
    negative "Experiment run FAILED."
    exit 1
}

function save_secondary_input {
    git diff $1 > $2/changes.diff
    git rev-parse HEAD > $2/revision.txt
    git ls-files --others --exclude-standard $1 > $2/untracked.txt
}

function ensure_spy_not_running {
    if [ "$(pidof spy)" ]; then
        negative "Spy process is already running. Kill it first!"
        exp_failed
    fi
}

clean_env () {
    sleep 1
    echo "Killing processes quickhpc, sensitive[1-9], spy, gnupg"
    ps -ef | grep "quickhpc" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "sensitive[1-9]" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "spy" | awk '{print $2;}' | xargs -r kill
    ps -ef | grep "encrypt_" | awk '{print $2;}' | xargs -r kill
    sleep 1
}
