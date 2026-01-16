from classification_icl_subspace import ExperimentConfig, LinearTransformer, GaussianMixtureDataset
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import asdict
from typing import List, Dict, Tuple, Optional
import pandas as pd


class CheckpointEvaluator:
    """Evaluator class for analyzing trained model checkpoints"""
    
    def __init__(self, checkpoint_dir: str, label_flips: Optional[List[float]] = [0.0, 0.2]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.label_flips = label_flips
        
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[LinearTransformer, ExperimentConfig]:
        """Load model and config from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = ExperimentConfig(**asdict(checkpoint['config']))
        model = LinearTransformer(config.d)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, config

    def evaluate_risk_curves(
        self,
        model: LinearTransformer,
        d: int,
        k: int,
        max_seq_length: int,
        R: Optional[float] = None,
        num_samples: int = 2500,
        label_flip_ps: Optional[List[float]] = None,
        device: str = 'cpu',
        include_memorization: bool = False, #new flag
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Evaluate model's performance curves including both means and standard errors.
        """
        if label_flip_ps is None:
            label_flip_ps = self.label_flips
            
        model = model.to(device)
        results = {}
        if R is None:
            R = d ** 0.3
        
        for label_flip_p in label_flip_ps:
            print(f"\nEvaluating curves for d={d}, label_flip_p={label_flip_p}")
            
            # Create test set
            dataset = GaussianMixtureDataset(
                d=d,
                k=k,
                N=max_seq_length,
                B=num_samples,
                R=R,
                is_validation=True,
                label_flip_p=label_flip_p
            )
            
            context_x, context_y, _, _ = [t.to(device) for t in dataset[0]]
            
            memorization_accuracies = np.zeros((max_seq_length-1, num_samples))
            test_accuracies = np.zeros((max_seq_length-1, num_samples))
            
            with torch.no_grad():
                for k in range(1, max_seq_length):
                    curr_context_x = context_x[:, :k]
                    curr_context_y = context_y[:, :k]

                    if include_memorization:
                        # Memorization accuracy (per example)
                        mem_preds = model.compute_in_context_preds(curr_context_x, curr_context_y)
                        mem_correct = (mem_preds == curr_context_y).float()
                        memorization_accuracies[k-1] = mem_correct.mean(dim=1).cpu().numpy()
                    
                    # Test accuracy (per example)
                    next_x = context_x[:, k:k+1, :]
                    next_y = context_y[:, k:k+1]
                    
                    pred_logits = model(curr_context_x, curr_context_y, next_x.squeeze(1))
                    test_preds = (pred_logits > 0).float()
                    test_correct = (test_preds == next_y.squeeze(1)).float()
                    test_accuracies[k-1] = test_correct.cpu().numpy()
                    
                    if k % 20 == 0:
                        print(f"Position {k}: memorization = {memorization_accuracies[k-1].mean():.3f}, test = {test_accuracies[k-1].mean():.3f}")

            results[label_flip_p] = {
                'test': {
                    'mean': test_accuracies.mean(axis=1),
                    'stderr': test_accuracies.std(axis=1) / np.sqrt(num_samples - 1)
                }
            }

            if include_memorization:
                results[label_flip_p]['memorization'] = {
                    'mean': memorization_accuracies.mean(axis=1),
                    'stderr': memorization_accuracies.std(axis=1) / np.sqrt(num_samples - 1)
                }
            
        return results

    def plot_dimension_curves(self, results_by_d: Dict[int, Dict[float, Dict[str, Dict[str, np.ndarray]]]],  R_d_to_power: float = 0.3,
                            sequence_length: int = 40, save_path: Optional[str] = None,
                            label_flip_ps: Optional[List[float]] = None, force_y_range = False):
        """Plot separate accuracy vs dimension curves for each label flip probability"""
        if label_flip_ps is None:
            label_flip_ps = self.label_flips

        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'

        dimensions = sorted(results_by_d.keys())
        seq_idx = sequence_length - 2
        
        mem_color = 'red'
        test_color = 'blue'
        opt_color = 'green'
        
        # Create separate plot for each label flip probability
        for label_flip_p in label_flip_ps:
            plt.figure(figsize=(8, 6))
            
            mem_means = []
            mem_errs = []
            test_means = []
            test_errs = []
            
            for d in dimensions:
                curves = results_by_d[d][label_flip_p]
                mem_means.append(curves['memorization']['mean'][seq_idx])
                mem_errs.append(curves['memorization']['stderr'][seq_idx])
                test_means.append(curves['test']['mean'][seq_idx])
                test_errs.append(curves['test']['stderr'][seq_idx])
            
            mem_means = np.array(mem_means)
            mem_errs = np.array(mem_errs)
            test_means = np.array(test_means)
            test_errs = np.array(test_errs)
            plt.errorbar(dimensions, mem_means, yerr=1.96*mem_errs, 
                        color=mem_color, linestyle='--', linewidth=2,
                        label='In-context train', capsize=3)
            plt.errorbar(dimensions, test_means, yerr=1.96*test_errs,
                        color=test_color, linestyle='-', linewidth=2,
                        label='Test', capsize=3)
            
            optimal_acc = 1.0 - label_flip_p
            plt.plot(dimensions, [optimal_acc] * len(dimensions), 
                    color=opt_color, linestyle='-.', linewidth=2, 
                    label=f'Optimal Test ({optimal_acc:.2f})')
            
            base_font = 18
            if force_y_range:
                plt.ylim(0.49, 1.01)
            plt.xlabel('Input Dimension (d)', fontsize = base_font+1)
            plt.ylabel('Accuracy', fontsize = base_font+1)

            plt.title(f'Performance vs Dimension ($\\tilde R=d^{{{R_d_to_power}}}$)\n(Seq. Length = {sequence_length}, Label Flip = {label_flip_p})', fontsize=base_font+2)
            plt.grid(True, alpha=0.4, color='gray', linewidth=0.5)
            plt.xticks(fontsize=base_font)
            plt.yticks(fontsize=base_font)
            plt.xscale('log')

            # Check if bottom right area is crowded by looking at the final values
            final_mem = mem_means[-1]  # Last memorization value
            final_test = test_means[-1]  # Last test value
            legend_threshold = 0.75  # Adjust this value to change sensitivity

            if final_mem > legend_threshold and final_test > legend_threshold:
                # If both lines are above threshold in bottom right, place legend there
                plt.legend(loc='lower right', fontsize=base_font)
            else:
                # Otherwise use the default center right position
                plt.legend(loc='center right', fontsize=base_font)
            
            if save_path:
                base_path = Path(save_path)
                # Ensure the parent directory exists
                base_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Add .png extension if no extension is provided
                if not base_path.suffix:
                    base_path = base_path.with_suffix('.png')
                    
                flip_specific_path = base_path.parent / f"{base_path.stem}_N{sequence_length}_R{R_d_to_power}_p{label_flip_p}{base_path.suffix}"
                plt.savefig(flip_specific_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot for label_flip_p={label_flip_p} to {flip_specific_path}")
            else:
                plt.show()
                
            plt.close()
    
    def evaluate_batch_sizes(self, 
                           dimension: int = 1000,
                           max_seq_length: int = 40,
                           R_d_to_power: Optional[float] = 0.3,
                           num_samples: int = 2500) -> Dict[int, Dict[float, Dict[str, Dict[str, np.ndarray]]]]:
        """
        Evaluate checkpoints for different batch sizes with fixed dimension.
        
        Args:
            dimension: Fixed dimension to analyze (default 1000)
            max_seq_length: Maximum sequence length to evaluate
            R: Optional radius parameter
            num_samples: Number of samples for evaluation
        """
        results_by_batch = {}

        if R_d_to_power is None:
            R = dimension**0.3
        else: 
            R = dimension**R_d_to_power
        
        # Find all checkpoints matching the dimension
        matches = list(self.checkpoint_dir.glob(f"checkpoint_d{dimension}*.pt"))
        
        for checkpoint_path in matches:
            # Extract batch size from filename
            # Assuming filename format contains "B{batch_size}"
            batch_str = str(checkpoint_path)
            batch_size = int(batch_str[batch_str.find('B')+1:].split('_')[0])
            print(f"\nEvaluating batch size {batch_size}")
            
            model, config = self.load_checkpoint(str(checkpoint_path))
            curves = self.evaluate_risk_curves(
                model=model,
                d=dimension,
                R=R,
                max_seq_length=max_seq_length + 1,
                num_samples=num_samples,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            results_by_batch[batch_size] = curves
            
        return results_by_batch

    def plot_batch_size_curves(self,
                             results_by_batch: Dict[int, Dict[float, Dict[str, Dict[str, np.ndarray]]]],
                             sequence_length: int = 40,
                             save_path: Optional[str] = None,
                             R_d_to_power: Optional[float] = 0.3,
                             label_flip_ps: Optional[List[float]] = None):
        """
        Plot accuracy vs batch size curves.
        """
        if label_flip_ps is None:
            label_flip_ps = self.label_flips

        if R_d_to_power is None:
            R_d_to_power = 0.3
            
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'
        
        batch_sizes = sorted(results_by_batch.keys())
        seq_idx = sequence_length - 2
        
        # Create separate plot for each label flip probability
        for label_flip_p in label_flip_ps:
            plt.figure(figsize=(8, 6))
            
            mem_means = []
            mem_errs = []
            test_means = []
            test_errs = []
            
            for batch_size in batch_sizes:
                curves = results_by_batch[batch_size][label_flip_p]
                mem_means.append(curves['memorization']['mean'][seq_idx])
                mem_errs.append(curves['memorization']['stderr'][seq_idx])
                test_means.append(curves['test']['mean'][seq_idx])
                test_errs.append(curves['test']['stderr'][seq_idx])
            
            mem_means = np.array(mem_means)
            mem_errs = np.array(mem_errs)
            test_means = np.array(test_means)
            test_errs = np.array(test_errs)
            
            plt.errorbar(batch_sizes, mem_means, yerr=1.96*mem_errs,
                        color='red', linestyle='--', linewidth=2,
                        label='In-context train', capsize=3)
            plt.errorbar(batch_sizes, test_means, yerr=1.96*test_errs,
                        color='blue', linestyle='-', linewidth=2,
                        label='Test', capsize=3)
            
            optimal_acc = 1.0 - label_flip_p
            plt.plot(batch_sizes, [optimal_acc] * len(batch_sizes),
                    color='green', linestyle='-.', linewidth=2,
                    label=f'Optimal Test ({optimal_acc:.2f})')
            
            base_font = 18
            plt.ylim(0.49, 1.01)
            plt.xlabel('Tasks (B)', fontsize=base_font+1)
            plt.ylabel('Accuracy', fontsize=base_font+1)
            plt.title(f'Performance vs Number of Tasks ($\\tilde R=d^{{{R_d_to_power}}}$)\n(d=1000, Seq. Length = {sequence_length}, Label Flip = {label_flip_p})', fontsize=base_font+2)
            plt.grid(True, alpha=0.4, color='gray', linewidth=0.5)
            plt.xticks(fontsize=base_font)
            plt.yticks(fontsize=base_font)
            plt.xscale('log')
            plt.legend(fontsize=base_font)
            
            if save_path:
                base_path = Path(save_path)
                base_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not base_path.suffix:
                    base_path = base_path.with_suffix('.png')
                    
                flip_specific_path = base_path.parent / f"{base_path.stem}_N{sequence_length}_R{R_d_to_power}_p{label_flip_p}{base_path.suffix}"
                plt.savefig(flip_specific_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot for label_flip_p={label_flip_p} to {flip_specific_path}")
            else:
                plt.show()
                
            plt.close()

    def evaluate_checkpoint(self, checkpoint_file: str, max_seq_length: int, R: Optional[float]=None,
                          label_flip_ps: Optional[List[float]]=None,  include_memorization: bool = False) -> Dict[int, Dict[float, Dict[str, np.ndarray]]]:
        """Evaluate a single checkpoint with specified maximum sequence length."""
        if label_flip_ps is None:
            label_flip_ps = self.label_flips
            
        print(f"\nEvaluating {checkpoint_file}")
        model, config = self.load_checkpoint(checkpoint_file)
        results = self.evaluate_risk_curves(
            model=model,
            k = config.k,
            d=config.d,
            R=R,
            max_seq_length=max_seq_length + 1,  # Add 1 to get desired sequence length
            num_samples=2500,
            label_flip_ps=label_flip_ps,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            include_memorization=include_memorization  # Pass flag
        )
        return {config.d: results}

    def plot_sequence_length_curves(
            self,
            results_by_N: Dict[int, Dict[float, Dict[str, Dict[str, np.ndarray]]]],
            fixed_d: int,
            R_d_to_power: float,
            save_path: Optional[str] = None,
            label_flip_ps: Optional[List[float]] = None,
            force_y_range: bool = True
    ):
        """Plot accuracy vs. sequence length (N) for fixed dimension d."""
        if label_flip_ps is None:
            label_flip_ps = self.label_flips

        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'

        sequence_lengths = sorted(results_by_N.keys())

        mem_color = 'red'
        test_color = 'blue'
        opt_color = 'green'

        for label_flip_p in label_flip_ps:
            plt.figure(figsize=(8, 6))

            mem_means = []
            mem_errs = []
            test_means = []
            test_errs = []

            for N in sequence_lengths:
                curves = results_by_N[N][label_flip_p]
                idx = N - 2  # accuracy at position N-1 (last token before prediction)
                if 'memorization' in curves:
                    mem_means.append(curves['memorization']['mean'][idx])
                    mem_errs.append(curves['memorization']['stderr'][idx])
                test_means.append(curves['test']['mean'][idx])
                test_errs.append(curves['test']['stderr'][idx])

            if mem_means:
                mem_means = np.array(mem_means)
                mem_errs = np.array(mem_errs)
                plt.errorbar(sequence_lengths, mem_means, yerr=1.96 * mem_errs,
                             color=mem_color, linestyle='--', linewidth=2,
                             label='In-context train', capsize=3)

            test_means = np.array(test_means)
            test_errs = np.array(test_errs)
            plt.errorbar(sequence_lengths, test_means, yerr=1.96 * test_errs,
                         color=test_color, linestyle='-', linewidth=2,
                         label='Test', capsize=3)

            optimal_acc = 1.0 - label_flip_p
            plt.plot(sequence_lengths, [optimal_acc] * len(sequence_lengths),
                     color=opt_color, linestyle='-.', linewidth=2,
                     label=f'Optimal Test ({optimal_acc:.2f})')

            base_font = 18
            if force_y_range:
                plt.ylim(0.49, 1.01)
            plt.xlabel('Sequence Length (N)', fontsize=base_font + 1)
            plt.ylabel('Accuracy', fontsize=base_font + 1)
            plt.title(
                f'Performance vs Sequence Length ($\\tilde R=d^{{{R_d_to_power}}}$)\n(d={fixed_d}, Label Flip = {label_flip_p})',
                fontsize=base_font + 2)
            plt.grid(True, alpha=0.4, color='gray', linewidth=0.5)
            plt.xticks(fontsize=base_font)
            plt.yticks(fontsize=base_font)
            plt.legend(fontsize=base_font)

            if save_path:
                base_path = Path(save_path)
                base_path.parent.mkdir(parents=True, exist_ok=True)
                if not base_path.suffix:
                    base_path = base_path.with_suffix('.png')
                flip_specific_path = base_path.parent / f"{base_path.stem}_d{fixed_d}_R{R_d_to_power}_p{label_flip_p}{base_path.suffix}"
                plt.savefig(flip_specific_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot for label_flip_p={label_flip_p} to {flip_specific_path}")
            else:
                plt.show()
            plt.close()

    def plot_sequence_length_curves_by_k(
            self,
            results_by_k: Dict[int, Dict[int, Dict[float, Dict[str, Dict[str, np.ndarray]]]]],
            fixed_d: int,
            R_d_to_power: float,
            save_path: Optional[str] = None,
            label_flip_ps: Optional[List[float]] = None,
            force_y_range: bool = True
    ):
        """Plot accuracy vs. sequence length (N) for different values of k."""
        if label_flip_ps is None:
            label_flip_ps = self.label_flips

        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.default'] = 'regular'

        for label_flip_p in label_flip_ps:
            plt.figure(figsize=(8, 6))

            for k in sorted(results_by_k):
                results_by_N = results_by_k[k]
                sequence_lengths = sorted(results_by_N.keys())

                test_means = []
                test_errs = []

                for N in sequence_lengths:
                    curves = results_by_N[N][label_flip_p]
                    idx = N - 2
                    test_means.append(curves['test']['mean'][idx])
                    test_errs.append(curves['test']['stderr'][idx])

                test_means = np.array(test_means)
                test_errs = np.array(test_errs)

                plt.errorbar(sequence_lengths, test_means, yerr=1.96 * test_errs,
                             label=f'Test (k={k})', linewidth=2, capsize=3)

            optimal_acc = 1.0 - label_flip_p
            plt.plot(sequence_lengths, [optimal_acc] * len(sequence_lengths),
                     color='gray', linestyle='--', linewidth=2,
                     label=f'Optimal Test ({optimal_acc:.2f})')

            base_font = 18
            if force_y_range:
                plt.ylim(0.49, 1.01)
            plt.xlabel('Sequence Length (N)', fontsize=base_font + 1)
            plt.ylabel('Accuracy', fontsize=base_font + 1)
            plt.title(
                f'Accuracy vs Sequence Length (d={fixed_d}, $\\tilde R=d^{{{R_d_to_power}}}$, flip={label_flip_p})',
                fontsize=base_font + 2)
            plt.grid(True, alpha=0.4, color='gray', linewidth=0.5)
            plt.xticks(fontsize=base_font)
            plt.yticks(fontsize=base_font)
            plt.legend(fontsize=base_font)

            if save_path:
                base_path = Path(save_path)
                base_path.parent.mkdir(parents=True, exist_ok=True)
                if not base_path.suffix:
                    base_path = base_path.with_suffix('.png')
                flip_specific_path = base_path.parent / f"{base_path.stem}_d{fixed_d}_R{R_d_to_power}_p{label_flip_p}{base_path.suffix}"
                plt.savefig(flip_specific_path, bbox_inches='tight', dpi=300)
                print(f"Saved plot for label_flip_p={label_flip_p} to {flip_specific_path}")
            else:
                plt.show()
            plt.close()

def main():
    evaluator = CheckpointEvaluator('checkpoints/', label_flips = [0.2])
    max_seq_length = 20

    # first run batch size results

    R_d_to_powers = [0.1, 0.3, 0.6]

    for R_d_to_power in R_d_to_powers:
        batch_results = evaluator.evaluate_batch_sizes(
            dimension=1000,
            max_seq_length=max_seq_length,
            num_samples=2500,
            R_d_to_power=R_d_to_power,
        )
        evaluator.plot_batch_size_curves(
            batch_results,
            sequence_length=max_seq_length,
            save_path='plots/batch_size_curves.png',
            R_d_to_power = R_d_to_power,
        )

    # then run high-dimensionality results

    all_results = {}
    dimensions = [10, 50, 100, 200, 400, 600, 800, 1000, 1250, 1500, 2000]

    R_d_to_powers = [0.1, 0.3, 0.6]

    
    for R_d_to_power in R_d_to_powers:
        for d in dimensions:
            matches = list(Path('checkpoints/').glob(f"checkpoint_d{d}*.pt"))
            if matches:
                for checkpoint_path in matches: 
                    batch_str = str(checkpoint_path)
                    batch_size = int(batch_str[batch_str.find('B')+1:].split('_')[0])
                    R = d**R_d_to_power
                    if batch_size == d: # only considering those where B=d
                        results = evaluator.evaluate_checkpoint(checkpoint_path, max_seq_length, R)
                        all_results.update(results)
        
        evaluator.plot_dimension_curves(all_results, sequence_length=max_seq_length, save_path="plots/dimension_curves.png", R_d_to_power=R_d_to_power, force_y_range = True)


# def subspace_different_k_experiments():
#     import matplotlib.pyplot as plt
#     import pandas as pd
#     from pathlib import Path
#     import torch
#     import numpy as np
#
#
#     d = 2000
#     # tilda_R = 2.5
#     # Define R and k values to evaluate
#     #R_d_to_powers = [0.5, 0.7]
#     #R_list = [int(d**power) for power in R_d_to_powers]
#     R_list = [5,7,9]
#     k_list = [500,1000,1500]
#
#     max_M = 500
#     step = 50
#     # M_values = list(range(step, max_M + 1, step))
#     M_values = (1, 5,20,50,150,250,500)
#
#     label_flip_p = 0.0
#     num_samples = 1000
#     results_by_k = {}
#
#     evaluator = CheckpointEvaluator('checkpoints/', label_flips=[0.0])
#     checkpoint_paths = list(Path('checkpoints/').glob(f"checkpoint_d{d}*.pt"))
#
#     if not checkpoint_paths:
#         raise FileNotFoundError("No checkpoint files found in 'checkpoints/'")
#
#     def extract_timestamp(ckpt_name):
#         parts = ckpt_name.split('_')
#         for i, part in enumerate(parts):
#             if part == 'step' and i >= 2:
#                 return f"{parts[i - 2]}_{parts[i - 1]}"
#         return "00000000_000000"
#
#     # Iterate over all (R, k) pairs
#     for R in R_list:
#         for k in k_list:
#             tilda_R = R
#             R_tag = f"R{int(R)}"
#             k_tag = f"k{k}"
#             d_tag = f"d{d}"
#             matching_ckpts = [
#                 p for p in checkpoint_paths
#                 if R_tag in p.name.split('_') and k_tag in p.name.split('_') and d_tag in p.name.split('_')
#             ]
#
#             if not matching_ckpts:
#                 print(f"❌ No checkpoint found for R={R}, k={k}")
#                 continue
#
#             latest_ckpt = max(matching_ckpts, key=lambda p: extract_timestamp(p.name))
#             print(f"\n✅ Using checkpoint: {latest_ckpt.name} for R={R}, k={k}")
#
#             # Load model and data
#             model, config = evaluator.load_checkpoint(str(latest_ckpt))
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#             model = model.to(device)
#             model.eval()
#
#             dataset = GaussianMixtureDataset(
#                 d=d, k=k, N=max_M + 1, B=num_samples, R=tilda_R,
#                 is_validation=False, label_flip_p=label_flip_p
#             )
#             context_x, context_y, _, _ = [t.to(device) for t in dataset[0]]
#
#             acc = []
#             with torch.no_grad():
#                 for M in M_values:
#                     cx = context_x[:, :M,:]
#                     cy = context_y[:, :M]
#                     tx = context_x[:, M,:]
#                     ty = context_y[:, M]
#
#                     logits = model(cx, cy, tx)
#                     preds = (logits > 0).float()
#                     accuracy = (preds == ty).float().cpu().numpy().mean()
#                     acc.append(accuracy)
#                     print(f"   R={R}, k={k}, M={M}: acc={accuracy:.4f}")
#
#             # Store results under this R, then by k
#             if R not in results_by_k:
#                 results_by_k[R] = {}
#             results_by_k[R][k] = acc
#
#     # Plot and save results for each R
#     plot_dir = Path("plots/subspace_by_k")
#     plot_dir.mkdir(parents=True, exist_ok=True)
#
#     for R, k_to_acc in results_by_k.items():
#         plt.figure(figsize=(8, 6))
#         for k, acc_list in sorted(k_to_acc.items()):
#             plt.plot(M_values, acc_list, label=f'k={k}', linewidth=2)
#
#
#         plt.title(f'Accuracy vs Sequence Length (d={d}, R={R})', fontsize=18)
#         plt.xlabel('Sequence Length (M)', fontsize=16)
#         plt.ylabel('Test Accuracy', fontsize=16)
#         plt.ylim(0.49, 1.01)
#         plt.grid(True, alpha=0.3)
#         plt.legend()
#         plt.tight_layout()
#
#         plot_path = plot_dir / f"subspace_k_vs_seq_R{int(R)}_Rtest{int(tilda_R)}.png"
#         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#         print(f"✅ Saved plot to {plot_path}")
#         plt.close()
#
#         # Save CSV
#         df = pd.DataFrame(k_to_acc, index=M_values)
#         df.index.name = 'M'
#         csv_path = plot_dir / f"subspace_k_vs_seq_R{int(R)}_Rtest{int(tilda_R)}.csv"
#         df.to_csv(csv_path)
#         print(f"📊 Saved accuracy data to {csv_path}")


def subspace_different_k_experiments():
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path
    import torch
    import numpy as np
    from scipy.stats import sem

    d = 2000
    R_list = [5, 7, 9]
    k_list = [500, 1000, 1500]
    max_M = 700
    M_values = (1, 5, 20, 50, 150, 250, 500,700)
    label_flip_p = 0.0
    num_samples = 1200
    results_by_k = {}

    evaluator = CheckpointEvaluator('checkpoints/', label_flips=[0.0])
    checkpoint_paths = list(Path('checkpoints/').glob(f"checkpoint_d{d}*.pt"))

    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoint files found in 'checkpoints/'")

    def extract_timestamp(ckpt_name):
        parts = ckpt_name.split('_')
        for i, part in enumerate(parts):
            if part == 'step' and i >= 2:
                return f"{parts[i - 2]}_{parts[i - 1]}"
        return "00000000_000000"

    for R in R_list:
        for k in k_list:
            tilda_R = R
            R_tag = f"R{int(R)}"
            k_tag = f"k{k}"
            d_tag = f"d{d}"
            matching_ckpts = [
                p for p in checkpoint_paths
                if R_tag in p.name.split('_') and k_tag in p.name.split('_') and d_tag in p.name.split('_')
            ]

            if not matching_ckpts:
                print(f"❌ No checkpoint found for R={R}, k={k}")
                continue

            latest_ckpt = max(matching_ckpts, key=lambda p: extract_timestamp(p.name))
            print(f"\n✅ Using checkpoint: {latest_ckpt.name} for R={R}, k={k}")

            model, config = evaluator.load_checkpoint(str(latest_ckpt))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)
            model.eval()

            dataset = GaussianMixtureDataset(
                d=d, k=k, N=max_M + 1, B=num_samples, R=tilda_R,
                is_validation=False, label_flip_p=label_flip_p
            )
            context_x, context_y, _, _ = [t.to(device) for t in dataset[0]]

            acc = [[] for _ in M_values]
            with torch.no_grad():
                for i, M in enumerate(M_values):
                    cx = context_x[:, :M, :]
                    cy = context_y[:, :M]
                    tx = context_x[:, M, :]
                    ty = context_y[:, M]

                    logits = model(cx, cy, tx)
                    preds = (logits > 0).float()
                    accuracies = (preds == ty).float().cpu().numpy()
                    acc[i].extend(accuracies)
                    print(f"   R={R}, k={k}, M={M}: acc={np.mean(accuracies):.4f}")

            if R not in results_by_k:
                results_by_k[R] = {}
            results_by_k[R][k] = acc

    # Compute mean and CI
    results_by_k_with_ci = {}
    for R in results_by_k:
        results_by_k_with_ci[R] = {}
        for k, acc_lists in results_by_k[R].items():
            means = [np.mean(runs) for runs in acc_lists]
            sems = [sem(runs) if len(runs) > 1 else 0.0 for runs in acc_lists]
            cis = [(m - 1.96 * s, m + 1.96 * s) for m, s in zip(means, sems)]
            results_by_k_with_ci[R][k] = (means, cis)

    # Plotting
    plot_dir = Path("plots/subspace_by_k")
    plot_dir.mkdir(parents=True, exist_ok=True)

    for R, k_to_data in results_by_k_with_ci.items():
        plt.figure(figsize=(8, 6))
        for k, (means, cis) in sorted(k_to_data.items()):
            lower = [c[0] for c in cis]
            upper = [c[1] for c in cis]
            plt.plot(M_values, means, label=f'k={k}', linewidth=2)
            plt.fill_between(M_values, lower, upper, alpha=0.2)

        plt.title(f"Accuracy vs Context Size ($R=\\tilde{{R}} = {R}$, $d={d}$)", fontsize=18)
        plt.xlabel('Sequence Length (M)', fontsize=16)
        plt.ylabel('Test Accuracy', fontsize=16)
        plt.ylim(0.49, 1.01)
        plt.grid(True, alpha=0.4)
        plt.legend(fontsize=16)
        plt.tight_layout()

        plot_path = plot_dir / f"subspace_k_vs_seq_R{int(R)}_Rtest{int(R)}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved plot to {plot_path}")
        plt.close()

        # Save to CSV
        df = pd.DataFrame(index=M_values)
        df.index.name = 'M'
        for k, (means, cis) in sorted(k_to_data.items()):
            df[f'k={k}_mean'] = means
            df[f'k={k}_lower'] = [c[0] for c in cis]
            df[f'k={k}_upper'] = [c[1] for c in cis]

        csv_path = plot_dir / f"subspace_k_vs_seq_R{int(R)}_Rtest{int(R)}.csv"
        df.to_csv(csv_path)
        print(f"📊 Saved accuracy data with CI to {csv_path}")


from sklearn.svm import LinearSVC


# def compare_linear_vs_mle(target_k: int = 30, max_context_length: int = 250, step_size: int = 50):
#     from scipy.stats import sem
#     evaluator = CheckpointEvaluator('checkpoints/', label_flips=[0.0])
#     checkpoint_paths = list(Path('checkpoints/').glob("checkpoint*.pt"))
#     if not checkpoint_paths:
#         raise FileNotFoundError("No checkpoint files found.")
#
#     # R_d_to_powers = [0.1, 0.2, 0.3]
#     d = 500
#     # R_values = [d ** p for p in R_d_to_powers]
#     R_values = [2,3,4]
#     M_values = [1,5,10,20,30,40,50]
#     max_context_length = max(M_values)+1
#
#     for R in R_values:
#         R_tag = f"R{int(R)}"
#         matching_ckpts = [
#             p for p in checkpoint_paths
#             if f'k{target_k}' in p.name.split('_') and R_tag in p.name.split('_') and f'd{d}' in p.name.split('_')
#         ]
#
#         if not matching_ckpts:
#             print(f"❌ No checkpoint found for k={target_k}, R={R_tag}")
#             continue
#
#         def extract_timestamp(ckpt_name):
#             try:
#                 parts = ckpt_name.split('_')
#                 for i, part in enumerate(parts):
#                     if part == 'step' and i >= 2:
#                         return f"{parts[i - 2]}_{parts[i - 1]}"
#             except:
#                 return "00000000_000000"
#
#         latest_ckpt = max(matching_ckpts, key=lambda p: extract_timestamp(p.name))
#         print(f"\n✅ Using latest checkpoint with k={target_k}, R={R_tag}: {latest_ckpt.name}")
#
#         model, config = evaluator.load_checkpoint(str(latest_ckpt))
#         d = config.d
#
#         dataset = GaussianMixtureDataset(
#             d=d, N=max_context_length + 1, B=1000, R=R,
#             is_validation=True, label_flip_p=0.0, k=target_k
#         )
#         context_x, context_y, _, _ = [t.to('cuda' if torch.cuda.is_available() else 'cpu') for t in dataset[0]]
#
#         test_acc_linear, test_acc_mle, test_acc_proj_mle, test_acc_svm = [], [], [], []
#         context_lengths = []
#
#         with torch.no_grad():
#             for M in M_values:
#                 cx = context_x[:, :M]
#                 cy = context_y[:, :M]
#                 tx = context_x[:, M]
#                 ty = context_y[:, M]
#
#                 logits = model(cx, cy, tx)
#                 preds = (logits > 0).float().cpu()
#                 acc_linear = (preds == ty).float().cpu().numpy()
#                 test_acc_linear.append(acc_linear.mean())
#
#                 mu_hat = torch.sum((2 * cy - 1).unsqueeze(-1) * cx, dim=1) / M
#                 logits_mle = torch.sum(mu_hat * tx, dim=1)
#                 preds_mle = (logits_mle > 0).float()
#                 acc_mle = (preds_mle == ty).float().cpu().numpy()
#                 test_acc_mle.append(acc_mle.mean())
#
#                 cx_proj = cx[:, :, :target_k]
#                 tx_proj = tx[:, :target_k]
#                 mu_hat_proj = torch.sum((2 * cy - 1).unsqueeze(-1) * cx_proj, dim=1) / M
#                 logits_proj = torch.sum(mu_hat_proj * tx_proj, dim=1)
#                 preds_proj = (logits_proj > 0).float()
#                 acc_proj = (preds_proj == ty).float().cpu().numpy()
#                 test_acc_proj_mle.append(acc_proj.mean())
#
#                 svm_preds = []
#                 for i in range(context_x.shape[0]):
#                     try:
#                         clf = LinearSVC(max_iter=1000, tol=1e-3)
#                         clf.fit(cx[i].cpu().numpy(), cy[i].cpu().numpy().astype(int))
#                         y_pred = clf.predict(tx[i].cpu().numpy().reshape(1, -1))
#                         svm_preds.append(y_pred[0])
#                     except:
#                         svm_preds.append(0)
#
#                 svm_preds = torch.tensor(svm_preds, device=ty.device).float()
#                 acc_svm = (svm_preds == ty).float().cpu().numpy()
#                 test_acc_svm.append(acc_svm.mean())
#
#                 context_lengths.append(M)
#
#                 print(f"M={M} | LT={test_acc_linear[-1]:.4f} | MLE={test_acc_mle[-1]:.4f} | ProjMLE={test_acc_proj_mle[-1]:.4f} | SVM={acc_svm:.4f}")
#
#         plot_dir = Path(f'plots/compare_lt_mle_svm_k{target_k}')
#         plot_dir.mkdir(parents=True, exist_ok=True)
#
#         plt.figure(figsize=(10, 6))
#         plt.plot(context_lengths, test_acc_linear, label='LinearTransformer', linewidth=2)
#         plt.plot(context_lengths, test_acc_mle, label='MLE (full)', linewidth=2)
#         plt.plot(context_lengths, test_acc_proj_mle, label=f'MLE (proj {target_k})', linewidth=2)
#         plt.plot(context_lengths, test_acc_svm, label='SVM', linewidth=2)
#         plt.axhline(1.0, linestyle='--', color='gray', label='Optimal')
#         plt.xlabel('Context Size (M)')
#         plt.ylabel('Test Accuracy')
#         plt.title(f"Accuracy vs Context Size ($R=\\tilde{{R}} = {R}$, $d={d}$, $k={target_k}$)", fontsize=18)
#         plt.ylim(0.49, 1.01)
#         plt.legend()
#         plt.grid(True, alpha=0.4)
#
#         plot_path = plot_dir / f'lt_vs_mle_vs_svm_d{d}_k{target_k}_R{int(R)}.png'
#         plt.savefig(plot_path, bbox_inches='tight', dpi=300)
#         print(f"📈 Saved plot to {plot_path}")
#
#         results_df = pd.DataFrame({
#             'context_size': context_lengths,
#             'linear_transformer': test_acc_linear,
#             'mle': test_acc_mle,
#             'proj_mle': test_acc_proj_mle,
#             'svm': test_acc_svm
#         })
#         csv_path = plot_dir / f'accuracy_data_d{d}_k{target_k}_R{int(R)}.csv'
#         results_df.to_csv(csv_path, index=False)
#         print(f"📊 Saved accuracy data to {csv_path}")
#

def compare_linear_vs_mle(target_k: int = 30, max_context_length: int = 250, step_size: int = 50):
    from scipy.stats import sem
    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path
    import torch
    import numpy as np
    from sklearn.svm import LinearSVC

    evaluator = CheckpointEvaluator('checkpoints/', label_flips=[0.0])
    checkpoint_paths = list(Path('checkpoints/').glob("checkpoint*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoint files found.")

    d = 500
    R_values = [2]
    M_values = [1, 5, 10, 20, 30, 40, 50,100]
    max_context_length = max(M_values) + 1

    for R in R_values:
        R_tag = f"R{int(R)}"
        matching_ckpts = [
            p for p in checkpoint_paths
            if f'k{target_k}' in p.name.split('_') and R_tag in p.name.split('_') and f'd{d}' in p.name.split('_')
        ]

        if not matching_ckpts:
            print(f"❌ No checkpoint found for k={target_k}, R={R_tag}")
            continue

        def extract_timestamp(ckpt_name):
            try:
                parts = ckpt_name.split('_')
                for i, part in enumerate(parts):
                    if part == 'step' and i >= 2:
                        return f"{parts[i - 2]}_{parts[i - 1]}"
            except:
                return "00000000_000000"

        latest_ckpt = max(matching_ckpts, key=lambda p: extract_timestamp(p.name))
        print(f"\n✅ Using latest checkpoint with k={target_k}, R={R_tag}: {latest_ckpt.name}")

        model, config = evaluator.load_checkpoint(str(latest_ckpt))
        d = config.d

        dataset = GaussianMixtureDataset(
            d=d, N=max_context_length + 1, B=1000, R=R,
            is_validation=True, label_flip_p=0.0, k=target_k
        )
        context_x, context_y, _, _ = [t.to('cuda' if torch.cuda.is_available() else 'cpu') for t in dataset[0]]

        test_acc_linear, test_acc_mle = [], []
        test_acc_proj_mle, test_acc_svm = [], []

        linear_cis, mle_cis, proj_cis, svm_cis = [], [], [], []
        context_lengths = []

        with torch.no_grad():
            for M in M_values:
                cx = context_x[:, :M]
                cy = context_y[:, :M]
                tx = context_x[:, M]
                ty = context_y[:, M]

                # Linear Transformer
                logits = model(cx, cy, tx)
                preds = (logits > 0).float().cpu()
                acc_linear = (preds == ty).float().cpu().numpy()
                test_acc_linear.append(acc_linear.mean())
                linear_cis.append(1.96 * sem(acc_linear))

                # MLE full
                mu_hat = torch.sum((2 * cy - 1).unsqueeze(-1) * cx, dim=1) / M
                logits_mle = torch.sum(mu_hat * tx, dim=1)
                preds_mle = (logits_mle > 0).float()
                acc_mle = (preds_mle == ty).float().cpu().numpy()
                test_acc_mle.append(acc_mle.mean())
                mle_cis.append(1.96 * sem(acc_mle))

                # MLE proj
                cx_proj = cx[:, :, :target_k]
                tx_proj = tx[:, :target_k]
                mu_hat_proj = torch.sum((2 * cy - 1).unsqueeze(-1) * cx_proj, dim=1) / M
                logits_proj = torch.sum(mu_hat_proj * tx_proj, dim=1)
                preds_proj = (logits_proj > 0).float()
                acc_proj = (preds_proj == ty).float().cpu().numpy()
                test_acc_proj_mle.append(acc_proj.mean())
                proj_cis.append(1.96 * sem(acc_proj))

                # SVM
                svm_preds = []
                for i in range(context_x.shape[0]):
                    try:
                        clf = LinearSVC(max_iter=1000, tol=1e-3)
                        clf.fit(cx[i].cpu().numpy(), cy[i].cpu().numpy().astype(int))
                        y_pred = clf.predict(tx[i].cpu().numpy().reshape(1, -1))
                        svm_preds.append(y_pred[0])
                    except:
                        svm_preds.append(0)
                svm_preds = torch.tensor(svm_preds, device=ty.device).float()
                acc_svm = (svm_preds == ty).float().cpu().numpy()
                test_acc_svm.append(acc_svm.mean())
                svm_cis.append(1.96 * sem(acc_svm))

                context_lengths.append(M)

                print(f"M={M} | LT={test_acc_linear[-1]:.4f} | MLE={test_acc_mle[-1]:.4f} | "
                      f"ProjMLE={test_acc_proj_mle[-1]:.4f} | SVM={test_acc_svm[-1]:.4f}")

        plot_dir = Path(f'plots/compare_lt_mle_svm_k{target_k}')
        plot_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))

        # Linear Transformer
        plt.plot(context_lengths, test_acc_linear, label='LinearTransformer', linewidth=2)
        plt.fill_between(context_lengths,
                         np.array(test_acc_linear) - np.array(linear_cis),
                         np.array(test_acc_linear) + np.array(linear_cis),
                         alpha=0.2)

        # MLE Full
        plt.plot(context_lengths, test_acc_mle, label='MLE (full)', linewidth=2)
        plt.fill_between(context_lengths,
                         np.array(test_acc_mle) - np.array(mle_cis),
                         np.array(test_acc_mle) + np.array(mle_cis),
                         alpha=0.2)

        # MLE Projected
        plt.plot(context_lengths, test_acc_proj_mle, label=f'MLE (proj {target_k})', linewidth=2)
        plt.fill_between(context_lengths,
                         np.array(test_acc_proj_mle) - np.array(proj_cis),
                         np.array(test_acc_proj_mle) + np.array(proj_cis),
                         alpha=0.2)

        # SVM
        plt.plot(context_lengths, test_acc_svm, label='SVM', linewidth=2)
        plt.fill_between(context_lengths,
                         np.array(test_acc_svm) - np.array(svm_cis),
                         np.array(test_acc_svm) + np.array(svm_cis),
                         alpha=0.2)


        plt.xlabel('Context Size (M)',fontsize=20)
        plt.ylabel('Test Accuracy',fontsize=20)
        plt.title(f"Accuracy vs Context Size ($R=\\tilde{{R}} = {R}$, $d={d}$, $k={target_k}$)", fontsize=18)
        plt.ylim(0.48, 1.01)
        plt.legend(fontsize=22)
        plt.grid(True, alpha=0.4)

        plot_path = plot_dir / f'lt_vs_mle_vs_svm_d{d}_k{target_k}_R{int(R)}.png'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        print(f"📈 Saved plot to {plot_path}")

        results_df = pd.DataFrame({
            'context_size': context_lengths,
            'linear_transformer': test_acc_linear,
            'mle': test_acc_mle,
            'proj_mle': test_acc_proj_mle,
            'svm': test_acc_svm
        })
        csv_path = plot_dir / f'accuracy_data_d{d}_k{target_k}_R{int(R)}.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"📊 Saved accuracy data to {csv_path}")


if __name__ == "__main__":
    subspace_eval_different_k = True
    run_comparison_experiment = False
    if subspace_eval_different_k:
        subspace_different_k_experiments()
    if run_comparison_experiment:
        compare_linear_vs_mle()
 