"""
Optuna-based Hyperparameter Tuner for VCM
Automatically searches 30+ trials to find optimal hyperparameters
Tracks accuracy, latency, and model size metrics

Hardware-aware: runs efficiently on both Jetson Orin Nano and DGX A100
"""

import optuna
from optuna.pruners import MedianPruner, PatientePruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from datetime import datetime
import numpy as np

from src.vcm_model import VCMModel, create_vcm_model
from src.vcm_data_loader import create_dataloaders, create_intent_mapping, AudioProcessor, compute_class_weights
from src.vcm_data.common import check_intent_contract
from src.vcm_trainer import Trainer
from src.vcm_hardware import HardwareProfiler, HardwareAwareConfig
from src.vcm_performance import LatencyMeasurer, ModelProfiler

logger = logging.getLogger(__name__)


class VCMObjective:
    """Optuna objective function for VCM hyperparameter tuning
    
    Tracks multiple metrics:
      - Accuracy (secondary, logged)
      - Latency (secondary, soft constraint <500ms)
      - Model size (secondary, track efficiency)
      - F1-score weighted (primary optimization metric)
    """
    
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        intent_mapping: Dict[str, int],
        device: torch.device,
        hardware_config: HardwareAwareConfig,
        max_epochs: int = 30,
        early_stopping_patience: int = 5,
        latency_constraint_ms: float = 500,
    ):
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.intent_mapping = intent_mapping
        self.device = device
        self.hardware_config = hardware_config
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.latency_constraint_ms = latency_constraint_ms
        
        # Performance measurement
        self.latency_measurer = LatencyMeasurer(device, n_warmup=10, n_samples=100)
        
        # Trial counter for logging
        self.trial_count = 0
    
    def __call__(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function: returns weighted F1 to maximize
        
        Optuna will:
        1. Suggest hyperparameters from search space
        2. Train model with those hyperparameters
        3. Measure accuracy + latency + size
        4. Return validation accuracy (primary metric)
        5. Prune unpromising trials early
        """
        self.trial_count += 1
        trial_id = trial.number
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Trial {trial_id + 1}: Starting optimization")
        logger.info(f"{'='*70}")
        
        try:
            # ===== SUGGEST HYPERPARAMETERS =====
            hyperparams = self._suggest_hyperparams(trial)
            
            logger.info("Suggested hyperparameters:")
            for key, val in hyperparams.items():
                logger.info(f"  {key}: {val}")
            
            # ===== CREATE DATA LOADERS =====
            train_loader, val_loader = create_dataloaders(
                train_manifest=self.train_manifest,
                val_manifest=self.val_manifest,
                intent_mapping=self.intent_mapping,
                batch_size=hyperparams['batch_size'],
                num_workers=self.hardware_config.get_num_workers(hyperparams['batch_size']),
                sample_rate=16000,
            )
            
            # ===== CREATE MODEL WITH CUSTOM ARCHITECTURE =====
            model = self._create_model(hyperparams)
            model = model.to(self.device)
            
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {param_count:,}")
            
            # Counter class imbalance (e.g. UNKNOWN-heavy composite) per trial
            class_weights = compute_class_weights(train_loader.dataset.samples, self.intent_mapping)

            # ===== CREATE TRAINER =====
            trainer = Trainer(
                model=model,
                device=self.device,
                mixed_precision=self.hardware_config.config['mixed_precision'],
                gradient_clip=1.0,
                class_weights=class_weights.to(self.device),
            )
            
            # ===== TRAINING LOOP WITH PRUNING =====
            best_val_acc = 0.0
            
            for epoch in range(1, self.max_epochs + 1):
                # Train one epoch
                train_loss, train_acc = trainer.train_epoch(
                    train_loader,
                    self._get_optimizer(model, hyperparams),
                )
                
                # Validate
                val_loss, val_acc = trainer.validate(val_loader)
                
                best_val_acc = max(best_val_acc, val_acc)
                
                logger.info(
                    f"Epoch {epoch}/{self.max_epochs} | "
                    f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Best: {best_val_acc:.4f}"
                )
                
                # Report to Optuna for pruning
                trial.report(val_acc, epoch)
                
                # Prune unpromising trials
                if trial.should_prune():
                    logger.info(f"Trial {trial_id + 1} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
                
                # Early stopping
                if epoch > 5 and val_acc < 0.5:  # Very poor performance
                    logger.info(f"Trial {trial_id + 1} stopped: validation accuracy too low")
                    raise optuna.TrialPruned()
            
            # ===== MEASURE PERFORMANCE METRICS =====
            latency_ms = self.latency_measurer.measure_latency(model)
            model_size_kb = ModelProfiler.get_model_size_on_disk(model)
            f1_weighted = self._compute_f1_weighted(model, val_loader)
            
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  Accuracy: {best_val_acc:.4f}")
            logger.info(f"  Latency: {latency_ms:.2f}ms (constraint: <{self.latency_constraint_ms}ms)")
            logger.info(f"  Model Size: {model_size_kb:.2f}KB")
            logger.info(f"  F1-Score (weighted): {f1_weighted:.4f}")
            
            # ===== STORE METRICS FOR LATER ANALYSIS =====
            trial.set_user_attr('accuracy', float(best_val_acc))
            trial.set_user_attr('latency_ms', float(latency_ms))
            trial.set_user_attr('model_size_kb', float(model_size_kb))
            trial.set_user_attr('f1_score_weighted', float(f1_weighted))
            trial.set_user_attr('latency_constraint_met', latency_ms < self.latency_constraint_ms)
            trial.set_user_attr('rtf', float(latency_ms / 4000))  # Real-time factor
            
            logger.info(f"Trial {trial_id + 1} completed: Best Acc = {best_val_acc:.4f}")
            
            # Return weighted F1 (to maximize). Raw accuracy is inflated by the
            # UNKNOWN-heavy class imbalance, so it must NOT be the objective.
            return f1_weighted
            
        except optuna.TrialPruned():
            logger.info(f"Trial {trial_id + 1} was pruned")
            raise
        
        except Exception as e:
            logger.error(f"Trial {trial_id + 1} failed: {str(e)}")
            raise
    
    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> Dict:
        """Suggest hyperparameters from hardware-appropriate search space"""
        space = self.hardware_config.get_search_space()
        
        hyperparams = {
            'batch_size': trial.suggest_categorical('batch_size', space['batch_size']),
            'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][-1], log=True),
            'weight_decay': trial.suggest_float('weight_decay', space['weight_decay'][0], space['weight_decay'][-1], log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', space['dropout_rate'][0], space['dropout_rate'][-1]),
            'n_filters_block1': trial.suggest_categorical('n_filters_block1', space['n_filters_block1']),
            'n_filters_block2': trial.suggest_categorical('n_filters_block2', space['n_filters_block2']),
            'n_filters_block3': trial.suggest_categorical('n_filters_block3', space['n_filters_block3']),
        }
        
        return hyperparams
    
    def _create_model(self, hyperparams: Dict) -> nn.Module:
        """Create VCM model with hyperparameter-specific architecture"""
        model = VCMModel(
            n_mel_bins=80,
            n_intents=len(self.intent_mapping),
            dropout_rate=hyperparams['dropout_rate'],
        )
        return model
    
    def _get_optimizer(self, model: nn.Module, hyperparams: Dict):
        """Create optimizer with hyperparameters"""
        return torch.optim.Adam(
            model.parameters(),
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay'],
        )
    
    def _compute_f1_weighted(self, model: nn.Module, val_loader: DataLoader) -> float:
        """Compute weighted F1-score across all classes"""
        from sklearn.metrics import f1_score
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for mel_specs, labels in val_loader:
                mel_specs = mel_specs.to(self.device)
                logits = model(mel_specs)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return float(f1_weighted)


class HyperparameterTuner:
    """Optuna-based hyperparameter tuning orchestrator"""
    
    def __init__(
        self,
        train_manifest: str,
        val_manifest: str,
        intent_mapping: Dict[str, int],
        device: torch.device,
        hardware_config: HardwareAwareConfig,
        storage_path: str = 'tuning_results',
        latency_constraint_ms: float = 500,
    ):
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.intent_mapping = intent_mapping
        self.device = device
        self.hardware_config = hardware_config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.latency_constraint_ms = latency_constraint_ms
        
        # Database path for Optuna
        self.db_path = self.storage_path / 'optuna_studies.db'
    
    def optimize(
        self,
        n_trials: int = 30,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        study_name: str = 'vcm_hyperparameter_tuning',
    ) -> Dict:
        """
        Run hyperparameter optimization
        
        Args:
            n_trials: Number of trials to run (recommend 30+)
            n_jobs: Number of parallel jobs (set to 1 for single GPU)
            timeout: Maximum time in seconds (None = unlimited)
            study_name: Name of Optuna study
        
        Returns:
            Best hyperparameters and study statistics
        """
        
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING HYPERPARAMETER OPTIMIZATION")
        logger.info(f"{'='*70}")
        logger.info(f"Trials: {n_trials}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Hardware Tier: {self.hardware_config.tier}")
        logger.info(f"Latency Constraint: <{self.latency_constraint_ms}ms (soft)")
        logger.info(f"Study database: {self.db_path}")
        logger.info(f"{'='*70}\n")
        
        # Create Optuna study with pruning
        sampler = TPESampler(
            seed=42,
            n_startup_trials=10,  # Random trials before TPE kicks in
        )
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )
        
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.db_path}"
        )
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction='maximize',  # Maximize weighted F1 (not raw accuracy -- imbalanced classes)
            load_if_exists=True,  # Resume if interrupted
        )
        
        # Create objective function
        objective = VCMObjective(
            train_manifest=self.train_manifest,
            val_manifest=self.val_manifest,
            intent_mapping=self.intent_mapping,
            device=self.device,
            hardware_config=self.hardware_config,
            max_epochs=30,
            early_stopping_patience=5,
            latency_constraint_ms=self.latency_constraint_ms,
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True,
            catch=(RuntimeError,),  # Continue on CUDA errors
        )
        
        # ===== RESULTS ANALYSIS =====
        logger.info(f"\n{'='*70}")
        logger.info(f"OPTIMIZATION COMPLETED")
        logger.info(f"{'='*70}")
        
        best_trial = study.best_trial
        logger.info(f"\nBest Trial: #{best_trial.number}")
        logger.info(f"Best Validation F1 (weighted): {best_trial.value:.4f}")
        logger.info(f"Best Latency: {best_trial.user_attrs.get('latency_ms', 'N/A'):.2f}ms")
        logger.info(f"Best Model Size: {best_trial.user_attrs.get('model_size_kb', 'N/A'):.2f}KB")
        logger.info(f"\nBest Hyperparameters:")
        for key, val in best_trial.params.items():
            logger.info(f"  {key}: {val}")
        
        # Save detailed results
        results = {
            'best_trial': best_trial.number,
            'best_accuracy': float(best_trial.user_attrs.get('accuracy', -1)),
            'best_latency_ms': float(best_trial.user_attrs.get('latency_ms', -1)),
            'best_model_size_kb': float(best_trial.user_attrs.get('model_size_kb', -1)),
            'best_f1_score': float(best_trial.value),
            'latency_constraint_met': best_trial.user_attrs.get('latency_constraint_met', False),
            'best_params': best_trial.params,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat(),
            'hardware_tier': self.hardware_config.tier,
        }
        
        results_path = self.storage_path / f'best_hyperparams_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {results_path}")
        
        # Print trial statistics
        self._print_trial_statistics(study)
        
        return results
    
    def _print_trial_statistics(self, study: optuna.study.Study):
        """Print statistics about all trials"""
        trials = study.trials
        completed = [t for t in trials if t.state == optuna.TrialState.COMPLETE]
        pruned = [t for t in trials if t.state == optuna.TrialState.PRUNED]
        failed = [t for t in trials if t.state == optuna.TrialState.FAIL]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Trial Statistics:")
        logger.info(f"  Total trials: {len(trials)}")
        logger.info(f"  Completed: {len(completed)}")
        logger.info(f"  Pruned: {len(pruned)}")
        logger.info(f"  Failed: {len(failed)}")
        
        if completed:
            accs = [t.value for t in completed if t.value is not None]
            latencies = [t.user_attrs.get('latency_ms') for t in completed if 'latency_ms' in t.user_attrs]
            latency_constraint_met = sum(1 for t in completed if t.user_attrs.get('latency_constraint_met', False))
            
            logger.info(f"\nCompleted Trial Accuracies:")
            logger.info(f"  Best: {max(accs):.4f}")
            logger.info(f"  Mean: {np.mean(accs):.4f}")
            logger.info(f"  Std: {np.std(accs):.4f}")
            
            if latencies:
                logger.info(f"\nLatency Statistics:")
                logger.info(f"  Best: {min(latencies):.2f}ms")
                logger.info(f"  Mean: {np.mean(latencies):.2f}ms")
                logger.info(f"  Std: {np.std(latencies):.2f}ms")
                logger.info(f"  <{self.latency_constraint_ms}ms: {latency_constraint_met}/{len(completed)} trials")
        
        logger.info(f"{'='*70}\n")
    
    def load_best_params(self, results_file: str) -> Dict:
        """Load best hyperparameters from saved results"""
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results['best_params']
