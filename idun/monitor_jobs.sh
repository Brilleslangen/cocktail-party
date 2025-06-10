#!/bin/bash
# monitor_jobs.sh - Utilities for monitoring SLURM jobs

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display job summary
show_jobs() {
    echo -e "${BLUE}=== Your Current Jobs ===${NC}"
    squeue -u $USER --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %.6C %.6G %R"
    echo ""
}

# Function to show detailed job info
job_info() {
    if [ -z "$1" ]; then
        echo "Usage: job_info <job_id>"
        return 1
    fi

    echo -e "${BLUE}=== Job Details for $1 ===${NC}"
    scontrol show job $1
    echo ""
}

# Function to tail job output
job_output() {
    if [ -z "$1" ]; then
        echo "Usage: job_output <job_id>"
        return 1
    fi

    # Find the output file
    output_file=$(find logs -name "*_$1.out" 2>/dev/null | head -1)
    error_file=$(find logs -name "*_$1.err" 2>/dev/null | head -1)

    if [ -f "$output_file" ]; then
        echo -e "${GREEN}=== Output from Job $1 ===${NC}"
        tail -n 50 "$output_file"
        echo ""
    fi

    if [ -f "$error_file" ] && [ -s "$error_file" ]; then
        echo -e "${RED}=== Errors from Job $1 ===${NC}"
        tail -n 20 "$error_file"
        echo ""
    fi
}

# Function to check GPU utilization
gpu_usage() {
    if [ -z "$1" ]; then
        echo "Usage: gpu_usage <job_id>"
        return 1
    fi

    # Get the node where job is running
    node=$(squeue -j $1 -h -o %N)

    if [ -n "$node" ]; then
        echo -e "${BLUE}=== GPU Usage on $node ===${NC}"
        ssh $node 'nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'
    else
        echo "Job $1 is not running or node information not available"
    fi
}

# Function to cancel all jobs
cancel_all() {
    echo -e "${YELLOW}Are you sure you want to cancel all your jobs? (y/n)${NC}"
    read -r response
    if [[ "$response" == "y" ]]; then
        scancel -u $USER
        echo -e "${GREEN}All jobs cancelled${NC}"
    fi
}

# Function to show job efficiency
job_efficiency() {
    if [ -z "$1" ]; then
        echo "Usage: job_efficiency <job_id>"
        return 1
    fi

    echo -e "${BLUE}=== Efficiency Report for Job $1 ===${NC}"
    seff $1
}

# Main menu
case "$1" in
    "")
        show_jobs
        ;;
    "info")
        job_info $2
        ;;
    "output")
        job_output $2
        ;;
    "gpu")
        gpu_usage $2
        ;;
    "cancel-all")
        cancel_all
        ;;
    "efficiency")
        job_efficiency $2
        ;;
    "watch")
        # Continuously monitor jobs
        watch -n 5 "squeue -u $USER --format='%.18i %.9P %.30j %.8u %.2t %.10M %.6D %.6C %.6G %R'"
        ;;
    *)
        echo "Usage: $0 [command] [job_id]"
        echo ""
        echo "Commands:"
        echo "  (no command)     Show all your jobs"
        echo "  info <job_id>    Show detailed job information"
        echo "  output <job_id>  Show job output/error logs"
        echo "  gpu <job_id>     Show GPU usage for running job"
        echo "  cancel-all       Cancel all your jobs"
        echo "  efficiency <id>  Show job efficiency report"
        echo "  watch            Continuously monitor jobs"
        ;;
esac