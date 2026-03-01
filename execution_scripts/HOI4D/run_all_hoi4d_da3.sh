#!/bin/bash
# HOI4D Master Control Script - Run all 5 videos with DA3 depth
# This script orchestrates training on all HOI4D videos with proper GPU/resource management

eval "$(conda shell.bash hook)"
conda activate Gaussians4D

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

cd "$PROJECT_ROOT"

echo "================================================================================"
echo "HOI4D MASTER TRAINING SCRIPT - DA3 DEPTH PREDICTIONS"
echo "================================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Scripts location: $SCRIPT_DIR"
echo "================================================================================"

# Trap errors
set -e

# ============================================================================
# FUNCTIONS
# ============================================================================

run_single_video() {
    local video_num=$1
    local video_dir="$SCRIPT_DIR/Video$video_num"
    local script="$video_dir/run_video${video_num}_hoi4d_da3.sh"
    
    if [ ! -f "$script" ]; then
        echo "❌ Script not found: $script"
        return 1
    fi
    
    echo ""
    echo "================================================================================"
    echo "▶️  STARTING VIDEO $video_num"
    echo "================================================================================"
    echo "Script: $script"
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
    
    bash "$script"
    
    echo "✅ VIDEO $video_num COMPLETED"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
}

run_all_sequential() {
    echo ""
    echo "🎬 SEQUENTIAL MODE: Running videos one after another"
    echo "This will take longer but is gentler on system resources"
    
    for video_num in 1 2 3 4 5; do
        run_single_video $video_num || echo "⚠️  Video $video_num encountered an error (continuing...)"
    done
}

run_all_parallel() {
    echo ""
    echo "⚡ PARALLEL MODE: Running videos in background"
    echo "WARNING: This requires significant GPU memory. Monitor with 'nvidia-smi'"
    
    pids=()
    for video_num in 1 2 3 4 5; do
        run_single_video $video_num &
        pids+=($!)
        sleep 2  # Stagger startup by 2 seconds
    done
    
    # Wait for all background jobs
    for pid in "${pids[@]}"; do
        wait $pid || echo "⚠️  Job $pid failed"
    done
}

# ============================================================================
# MAIN
# ============================================================================

if [ $# -eq 0 ]; then
    MODE="sequential"
    VIDEOS="1 2 3 4 5"
elif [ "$1" = "seq" ]; then
    MODE="sequential"
    VIDEOS="${@:2:-1}"
    [ -z "$VIDEOS" ] && VIDEOS="1 2 3 4 5"
elif [ "$1" = "par" ]; then
    MODE="parallel"
    VIDEOS="${@:2:-1}"
    [ -z "$VIDEOS" ] && VIDEOS="1 2 3 4 5"
elif [ "$1" = "help" ]; then
    cat << EOF
Usage: $0 [MODE] [VIDEO_NUMBERS]

Modes:
  seq        Sequential: Run videos one after another (default)
  par        Parallel: Run videos simultaneously (requires high GPU memory)
  help       Show this help message

Video numbers:
  1 2 3 4 5  Individual video numbers to run

Examples:
  $0                      # Run all videos sequentially
  $0 seq 1 2 3            # Run videos 1,2,3 sequentially
  $0 par 3 4              # Run videos 3,4 in parallel
  $0 help                 # Show this help

Notes:
  - Sequential: Recommended for stable training
  - Parallel: Use only if you have multiple GPUs or sufficient VRAM
  - Each video uses different port (6256-6260)
  - Results saved in output/ folder

EOF
    exit 0
else
    echo "Unknown option: $1"
    echo "Use '$0 help' for usage information"
    exit 1
fi

# Summary
echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Videos: $VIDEOS"
echo "  Batch size: 16 per video"
echo "  Total iterations: ~15,000 per video"
echo "  Expected time: 2-4 hours per video (depending on GPU)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Execute
if [ "$MODE" = "sequential" ]; then
    run_all_sequential
else
    run_all_parallel
fi

echo ""
echo "================================================================================"
echo "✅ ALL VIDEOS COMPLETED"
echo "================================================================================"
echo "Check output/ folder for results"
echo "Completion time: $(date '+%Y-%m-%d %H:%M:%S')"
