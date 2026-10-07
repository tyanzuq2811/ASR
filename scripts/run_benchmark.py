import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset import ASRDataset
from src.benchmark import ASRBenchmarkRunner
from src.visualize import plot_benchmark_results

def main():
    parser = argparse.ArgumentParser(description="Run ASR Whisper-Medusa Comparative Benchmark")
    parser.add_argument("--medusa-model", type=str, default="aiola/whisper-medusa-v1", help="HuggingFace model ID or path for Medusa")
    parser.add_argument("--baseline-model", type=str, default="openai/whisper-large-v2", help="HuggingFace model ID for Vanilla Whisper baseline")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline model inference")
    parser.add_argument("--language", type=str, default=None, choices=["en", "vi", "all"], help="Filter by language")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of audio samples (e.g. 5, 10 for quick testing)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save benchmark metrics and reports")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision (use FP32)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    lang = None if (args.language is None or args.language == "all") else args.language
    output_path = root_dir / args.output_dir

    print("=" * 70)
    print("   ASR BENCHMARK: WHISPER-MEDUSA vs VANILLA WHISPER")
    print(f"   Output Directory: {output_path}")
    print(f"   Language Filter : {lang or 'All (EN + VI)'}")
    print(f"   Sample Limit    : {args.limit or 'All 200 samples'}")
    print("=" * 70)

    dataset = ASRDataset(base_dir=str(root_dir / "real200"))
    
    runner = ASRBenchmarkRunner(
        medusa_model_name=args.medusa_model,
        baseline_model_name=None if args.no_baseline else args.baseline_model,
        output_dir=str(output_path),
        fp16=not args.no_fp16,
    )

    runner.run(dataset, language=lang, limit=args.limit)

    if not args.no_plot:
        print("\nGenerating presentation-ready charts...")
        plot_benchmark_results(results_dir=str(output_path))

    print("\nBenchmark successfully completed!")

if __name__ == "__main__":
    main()
