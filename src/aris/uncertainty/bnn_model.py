import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ReasoningTrace:
    """Store traceable reasoning information"""
    triplet: Tuple[str, str, str]
    llm_embedding: torch.Tensor
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    prediction_mean: float
    prediction_std: float
    confidence: float
    kl_divergence: float
    layer_activations: Dict[str, torch.Tensor]

class BayesianLinear(nn.Module):
    """Bayesian Linear Layer with weight uncertainty"""
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        
        # Prior distribution
        self.prior_std = prior_std
        self.prior = Normal(0, prior_std)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_logvar, -5)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_logvar, -5)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, float]:
        """Forward pass with optional sampling"""
        if sample:
            # Sample weights from posterior
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean weights (for deterministic inference)
            weight = self.weight_mu
            bias = self.bias_mu
        
        # Compute KL divergence
        kl = self._kl_divergence()
        
        return F.linear(x, weight, bias), kl
    
    def _kl_divergence(self) -> float:
        """Compute KL divergence between posterior and prior"""
        weight_posterior = Normal(self.weight_mu, torch.exp(0.5 * self.weight_logvar))
        bias_posterior = Normal(self.bias_mu, torch.exp(0.5 * self.bias_logvar))
        
        weight_kl = kl_divergence(weight_posterior, self.prior).sum()
        bias_kl = kl_divergence(bias_posterior, self.prior).sum()
        
        return (weight_kl + bias_kl).item()

class BayesianReasoningModule(nn.Module):
    """Bayesian Neural Network for supervised LLM reasoning"""
    def __init__(
        self,
        llm_embedding_dim: int = 768,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 1,
        prior_std: float = 1.0,
        n_samples: int = 10
    ):
        super().__init__()
        self.llm_embedding_dim = llm_embedding_dim
        self.n_samples = n_samples
        
        # Build Bayesian network
        layers = []
        in_dim = llm_embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(BayesianLinear(in_dim, hidden_dim, prior_std))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(BayesianLinear(in_dim, output_dim, prior_std))
        
        self.layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        
        # Store traces for interpretability
        self.reasoning_traces: List[ReasoningTrace] = []
    
    def forward(
        self,
        llm_embedding: torch.Tensor,
        triplet: Optional[Tuple[str, str, str]] = None,
        sample: bool = True,
        store_trace: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty quantification
        
        Args:
            llm_embedding: Embedding from LLM (batch_size, embedding_dim)
            triplet: Knowledge graph triplet (head, relation, tail)
            sample: Whether to sample from posterior
            store_trace: Whether to store reasoning trace
        
        Returns:
            Dictionary containing predictions and uncertainties
        """
        batch_size = llm_embedding.shape[0]
        
        if sample:
            # Monte Carlo sampling for uncertainty estimation
            predictions = []
            total_kl = 0.0
            layer_activations = {}
            
            for i in range(self.n_samples):
                x = llm_embedding
                sample_kl = 0.0
                activations = {}
                
                for idx, layer in enumerate(self.layers):
                    x, kl = layer(x, sample=True)
                    sample_kl += kl
                    
                    if idx < len(self.layers) - 1:
                        x = self.activation(x)
                    
                    activations[f'layer_{idx}'] = x.detach()
                
                predictions.append(x)
                total_kl += sample_kl
                
                if i == 0:  # Store first sample activations
                    layer_activations = activations
            
            predictions = torch.stack(predictions, dim=0)  # (n_samples, batch_size, output_dim)
            
            # Compute statistics
            pred_mean = predictions.mean(dim=0)
            pred_std = predictions.std(dim=0)
            
            # Epistemic uncertainty (model uncertainty)
            epistemic = pred_std.mean().item()
            
            # Aleatoric uncertainty (data uncertainty) - approximated
            aleatoric = torch.abs(pred_mean).mean().item() * 0.1
            
            avg_kl = total_kl / self.n_samples
            
        else:
            # Deterministic forward pass
            x = llm_embedding
            total_kl = 0.0
            layer_activations = {}
            
            for idx, layer in enumerate(self.layers):
                x, kl = layer(x, sample=False)
                total_kl += kl
                layer_activations[f'layer_{idx}'] = x.detach()
                
                if idx < len(self.layers) - 1:
                    x = self.activation(x)
            
            pred_mean = x
            pred_std = torch.zeros_like(pred_mean)
            epistemic = 0.0
            aleatoric = 0.0
            avg_kl = total_kl
        
        # Compute confidence (inverse of total uncertainty)
        total_uncertainty = epistemic + aleatoric
        confidence = 1.0 / (1.0 + total_uncertainty)
        
        # Store reasoning trace
        if store_trace and triplet is not None:
            for i in range(batch_size):
                trace = ReasoningTrace(
                    triplet=triplet,
                    llm_embedding=llm_embedding[i].detach(),
                    epistemic_uncertainty=epistemic,
                    aleatoric_uncertainty=aleatoric,
                    prediction_mean=pred_mean[i].item(),
                    prediction_std=pred_std[i].item() if sample else 0.0,
                    confidence=confidence,
                    kl_divergence=avg_kl,
                    layer_activations={k: v[i] for k, v in layer_activations.items()}
                )
                self.reasoning_traces.append(trace)
        
        return {
            'prediction_mean': pred_mean,
            'prediction_std': pred_std,
            'epistemic_uncertainty': torch.tensor(epistemic),
            'aleatoric_uncertainty': torch.tensor(aleatoric),
            'total_uncertainty': torch.tensor(total_uncertainty),
            'confidence': torch.tensor(confidence),
            'kl_divergence': torch.tensor(avg_kl),
            'layer_activations': layer_activations
        }
    
    def get_traces(self, last_n: Optional[int] = None) -> List[ReasoningTrace]:
        """Retrieve reasoning traces"""
        if last_n is None:
            return self.reasoning_traces
        return self.reasoning_traces[-last_n:]
    
    def clear_traces(self):
        """Clear stored traces"""
        self.reasoning_traces = []
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        n_batches: int
    ) -> torch.Tensor:
        """
        Compute ELBO loss for Bayesian training
        
        Args:
            predictions: Output from forward pass
            targets: Ground truth labels
            n_batches: Number of batches in dataset (for KL scaling)
        """
        # Negative log likelihood
        pred_mean = predictions['prediction_mean']
        pred_std = predictions['prediction_std'] + 1e-6  # Avoid division by zero
        
        nll = -Normal(pred_mean, pred_std).log_prob(targets).mean()
        
        # KL divergence (scaled by number of batches)
        kl = predictions['kl_divergence'] / n_batches
        
        # ELBO = NLL + KL
        loss = nll + kl
        
        return loss

class LLMKGReasoningSystem:
    """Complete system integrating LLM with Bayesian reasoning"""
    def __init__(
        self,
        llm_model,
        tokenizer,
        bayesian_module: BayesianReasoningModule,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.bayesian_module = bayesian_module.to(device)
        self.device = device
    
    def encode_triplet(self, triplet: Tuple[str, str, str]) -> torch.Tensor:
        """Encode knowledge graph triplet using LLM"""
        head, relation, tail = triplet
        text = f"Head: {head}, Relation: {relation}, Tail: {tail}"
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model(**inputs)
            # Use CLS token or mean pooling
            embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        return embedding
    
    def reason(
        self,
        triplet: Tuple[str, str, str],
        sample: bool = True
    ) -> Dict:
        """
        Perform reasoning on a knowledge graph triplet
        
        Args:
            triplet: (head, relation, tail)
            sample: Whether to use sampling for uncertainty
        
        Returns:
            Dictionary with predictions and uncertainties
        """
        # Encode triplet with LLM
        embedding = self.encode_triplet(triplet)
        
        # Bayesian reasoning
        results = self.bayesian_module(
            embedding,
            triplet=triplet,
            sample=sample,
            store_trace=True
        )
        
        # Format output
        output = {
            'triplet': triplet,
            'prediction': results['prediction_mean'].item(),
            'uncertainty': results['prediction_std'].item(),
            'epistemic_uncertainty': results['epistemic_uncertainty'].item(),
            'aleatoric_uncertainty': results['aleatoric_uncertainty'].item(),
            'confidence': results['confidence'].item(),
            'kl_divergence': results['kl_divergence'].item(),
        }
        
        return output
    
    def batch_reason(
        self,
        triplets: List[Tuple[str, str, str]],
        sample: bool = True
    ) -> List[Dict]:
        """Batch reasoning on multiple triplets"""
        results = []
        for triplet in triplets:
            result = self.reason(triplet, sample=sample)
            results.append(result)
        return results
    
    def get_reasoning_trace(self, triplet_idx: int = -1) -> ReasoningTrace:
        """Get detailed reasoning trace for analysis"""
        traces = self.bayesian_module.get_traces()
        return traces[triplet_idx]
    
    def print_trace(self, trace: ReasoningTrace):
        """Pretty print reasoning trace"""
        print(f"\n{'='*60}")
        print(f"Reasoning Trace for Triplet: {trace.triplet}")
        print(f"{'='*60}")
        print(f"Prediction Mean: {trace.prediction_mean:.4f}")
        print(f"Prediction Std: {trace.prediction_std:.4f}")
        print(f"Epistemic Uncertainty: {trace.epistemic_uncertainty:.4f}")
        print(f"Aleatoric Uncertainty: {trace.aleatoric_uncertainty:.4f}")
        print(f"Confidence: {trace.confidence:.4f}")
        print(f"KL Divergence: {trace.kl_divergence:.4f}")
        print(f"\nLayer Activations:")
        for layer_name, activation in trace.layer_activations.items():
            print(f"  {layer_name}: shape={activation.shape}, "
                  f"mean={activation.mean():.4f}, std={activation.std():.4f}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Mock LLM model for demonstration
    class MockLLM(nn.Module):
        def __init__(self, embedding_dim=768):
            super().__init__()
            self.embedding = nn.Embedding(1000, embedding_dim)
            
        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            return type('Outputs', (), {'last_hidden_state': x})()
    
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            # Simple mock tokenization
            input_ids = torch.randint(0, 1000, (1, 10))
            return {'input_ids': input_ids}
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: {'input_ids': torch.randint(0, 1000, (1, 10))}
    
    # Initialize system
    llm = MockLLM(embedding_dim=768)
    tokenizer = MockTokenizer()
    bayesian_module = BayesianReasoningModule(
        llm_embedding_dim=768,
        hidden_dims=[512, 256, 128],
        n_samples=20
    )
    
    system = LLMKGReasoningSystem(llm, tokenizer, bayesian_module)
    
    # Example triplets from knowledge graph
    triplets = [
        ("Albert Einstein", "won", "Nobel Prize"),
        ("Paris", "is_capital_of", "France"),
        ("Python", "is_used_for", "Machine Learning")
    ]
    
    # Perform reasoning
    print("Bayesian Neural Network Reasoning Results:\n")
    results = system.batch_reason(triplets, sample=True)
    
    for result in results:
        print(f"Triplet: {result['triplet']}")
        print(f"  Prediction: {result['prediction']:.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Epistemic Uncertainty: {result['epistemic_uncertainty']:.4f}")
        print(f"  Aleatoric Uncertainty: {result['aleatoric_uncertainty']:.4f}")
        print()
    
    # Show detailed trace for last triplet
    trace = system.get_reasoning_trace(-1)
    system.print_trace(trace)
