"""VAE-based domain adaptation for radiomics and pathomics.

The module owns the modality encoders, reconstruction/alignment training,
and the survival-prediction wrapper. PyTorch's DataLoader is imported
explicitly to avoid collision with graph-data loaders in analysis scripts.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class DomainAdaptationVAE(nn.Module):
    """VAE for domain adaptation with dual encoders and separate decoders"""
    
    def __init__(self, radiomics_dim, pathomics_dim, hidden_dim=128, latent_dim=32):
        super(DomainAdaptationVAE, self).__init__()
        
        # Encoders for each domain
        self.radiomics_encoder = nn.Sequential(
            nn.Linear(radiomics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.pathomics_encoder = nn.Sequential(
            nn.Linear(pathomics_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Mean and variance layers for latent space
        self.radiomics_mu = nn.Linear(hidden_dim, latent_dim)
        self.radiomics_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.pathomics_mu = nn.Linear(hidden_dim, latent_dim)
        self.pathomics_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Separate decoders for each modality
        self.radiomics_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, radiomics_dim)
        )
        
        self.pathomics_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, pathomics_dim)
        )
        
    def encode_radiomics(self, x):
        h = self.radiomics_encoder(x)
        return self.radiomics_mu(h), self.radiomics_logvar(h)
    
    def encode_pathomics(self, x):
        h = self.pathomics_encoder(x)
        return self.pathomics_mu(h), self.pathomics_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode_radiomics(self, z):
        return self.radiomics_decoder(z)
    
    def decode_pathomics(self, z):
        return self.pathomics_decoder(z)
    
    def forward(self, radiomics, pathomics):
        # Encode radiomics
        radio_mu, radio_logvar = self.encode_radiomics(radiomics)
        radio_z = self.reparameterize(radio_mu, radio_logvar)
        
        # Encode pathomics
        patho_mu, patho_logvar = self.encode_pathomics(pathomics)
        patho_z = self.reparameterize(patho_mu, patho_logvar)
        
        # Decode from radiomics latent space
        radio_recon_from_radio = self.decode_radiomics(radio_z)
        patho_recon_from_radio = self.decode_pathomics(radio_z)
        
        # Decode from pathomics latent space
        radio_recon_from_patho = self.decode_radiomics(patho_z)
        patho_recon_from_patho = self.decode_pathomics(patho_z)
        
        return {
            'radio_mu': radio_mu, 'radio_logvar': radio_logvar,
            'patho_mu': patho_mu, 'patho_logvar': patho_logvar,
            'radio_z': radio_z, 'patho_z': patho_z,
            'radio_recon_from_radio': radio_recon_from_radio,
            'patho_recon_from_patho': patho_recon_from_patho,
            'radio_recon_from_patho': radio_recon_from_patho,
            'patho_recon_from_radio': patho_recon_from_radio
        }

class DomainAdaptationTransformer(BaseEstimator, TransformerMixin):
    """Domain adaptation using dual VAE encoders with separate decoders"""
    
    def __init__(self, radiomics_dim, pathomics_dim, hidden_dim=128, latent_dim=32,
                 epochs=100, batch_size=64, learning_rate=1e-3, beta_kl=1.0,
                 beta_domain=1.0, beta_recon=1.0, beta_cross=0.5, kl_annealing_steps=50,
                 device='cuda'):
        self.radiomics_dim = radiomics_dim
        self.pathomics_dim = pathomics_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_kl = beta_kl
        self.beta_domain = beta_domain
        self.beta_recon = beta_recon
        self.beta_cross = beta_cross
        self.kl_annealing_steps = kl_annealing_steps
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model = None
        self.radiomics_scaler = StandardScaler()
        self.pathomics_scaler = StandardScaler()
        
    def kl_divergence(self, mu, logvar):
        """KL divergence for a single Gaussian distribution (VAE regularization)"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    def kl_divergence_between_distributions(self, mu1, logvar1, mu2, logvar2):
        """KL divergence between two Gaussian distributions (domain alignment)"""
        var1 = logvar1.exp()
        var2 = logvar2.exp()
        
        kl = 0.5 * torch.sum(
            logvar2 - logvar1 + 
            (var1 + (mu1 - mu2).pow(2)) / var2 - 1,
            dim=1
        ).mean()
        return kl
    
    def mmd_between_distributions(self, z1, z2, sigma=None):
        """
        Maximum Mean Discrepancy between two sets of hidden representations.
        
        Args:
            z1: Source domain features [batch_size, feature_dim]
            z2: Target domain features [batch_size, feature_dim]
            sigma: Bandwidth for RBF kernel (if None, use median heuristic)
        
        Returns:
            MMD loss (scalar)
        """
        batch_size = z1.size(0)
        features = torch.cat([z1, z2], dim=0)
        
        # Compute pairwise distances
        xx = torch.mm(z1, z1.t())
        xy = torch.mm(z1, z2.t())
        yy = torch.mm(z2, z2.t())
        
        # Squared L2 distances
        diag_xx = torch.diag(xx).unsqueeze(1).expand_as(xx)
        diag_yy = torch.diag(yy).unsqueeze(1).expand_as(yy)
        
        dist_xx = diag_xx + diag_xx.t() - 2 * xx
        dist_xy = diag_xx + diag_yy.t() - 2 * xy
        dist_yy = diag_yy + diag_yy.t() - 2 * yy
        
        # Clamp for numerical stability
        dist_xx = torch.clamp(dist_xx, min=0)
        dist_xy = torch.clamp(dist_xy, min=0)
        dist_yy = torch.clamp(dist_yy, min=0)
        
        # Median heuristic for bandwidth (if not provided)
        if sigma is None:
            with torch.no_grad():
                all_dists = torch.cat([
                    dist_xx.flatten(), 
                    dist_xy.flatten(), 
                    dist_yy.flatten()
                ])
                sigma = torch.median(all_dists[all_dists > 0]).sqrt()
                sigma = sigma.clamp(min=1e-5)
        
        # RBF kernel: K(x,y) = exp(-||x-y||^2 / (2*sigma^2))
        gamma = 1.0 / (2 * sigma**2 + 1e-8)
        
        k_xx = torch.exp(-gamma * dist_xx)
        k_xy = torch.exp(-gamma * dist_xy)
        k_yy = torch.exp(-gamma * dist_yy)
        
        # MMD: E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
        mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
        
        return mmd

    def mmd_multi_kernel(self, z1, z2, sigmas=None):
        """
        Multi-Kernel MMD with multiple bandwidths.
        """
        if sigmas is None:
            # Common practice: use multiple scales
            sigmas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        mmd_total = 0
        for sigma in sigmas:
            mmd_total += self.mmd_between_distributions(z1, z2, sigma)
        
        return mmd_total / len(sigmas)
    
    def reconstruction_loss(self, recon_x, x):
        """MSE reconstruction loss"""
        return nn.MSELoss()(recon_x, x)
    
    def fit(self, radiomics_data, pathomics_data, verbose=True):
        """Train the domain adaptation model"""
        
        # Normalize data
        radiomics_norm = self.radiomics_scaler.fit_transform(radiomics_data)
        pathomics_norm = self.pathomics_scaler.fit_transform(pathomics_data)
        
        # Convert to tensors
        radiomics_tensor = torch.FloatTensor(radiomics_norm)
        pathomics_tensor = torch.FloatTensor(pathomics_norm)
        
        dataset = TensorDataset(radiomics_tensor, pathomics_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        self.model = DomainAdaptationVAE(
            self.radiomics_dim, self.pathomics_dim,
            self.hidden_dim, self.latent_dim
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            total_recon_loss = 0
            total_cross_recon_loss = 0
            total_kl_loss = 0
            total_domain_loss = 0

            progress = epoch / self.kl_annealing_steps
            kl_weight = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            
            for batch_radio, batch_patho in dataloader:
                batch_radio = batch_radio.to(self.device)
                batch_patho = batch_patho.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_radio, batch_patho)
                
                # Self-reconstruction losses (reconstruct from own latent space)
                recon_loss_radio_self = self.reconstruction_loss(
                    outputs['radio_recon_from_radio'], batch_radio
                )
                recon_loss_patho_self = self.reconstruction_loss(
                    outputs['patho_recon_from_patho'], batch_patho
                )
                
                # Cross-reconstruction losses (reconstruct from other domain's latent space)
                recon_loss_radio_cross = self.reconstruction_loss(
                    outputs['radio_recon_from_patho'], batch_radio
                )
                recon_loss_patho_cross = self.reconstruction_loss(
                    outputs['patho_recon_from_radio'], batch_patho
                )
                
                # Total reconstruction losses
                recon_loss = (recon_loss_radio_self + recon_loss_patho_self) / 2
                cross_recon_loss = (recon_loss_radio_cross + recon_loss_patho_cross) / 2
                
                # KL divergence for each distribution (VAE regularization)
                kl_radio = self.kl_divergence(outputs['radio_mu'], outputs['radio_logvar'])
                kl_patho = self.kl_divergence(outputs['patho_mu'], outputs['patho_logvar'])
                kl_loss = (kl_radio + kl_patho) / 2
                
                # KL divergence between the two distributions (domain adaptation)
                # domain_loss = self.kl_divergence_between_distributions(
                #     outputs['radio_mu'], outputs['radio_logvar'],
                #     outputs['patho_mu'], outputs['patho_logvar']
                # )

                domain_loss = self.mmd_multi_kernel(
                    outputs['radio_z'], outputs['patho_z']
                )
                
                # Total loss with separate weights for self and cross reconstruction
                loss = (self.beta_recon * recon_loss + 
                       self.beta_cross * cross_recon_loss +
                       self.beta_kl * kl_weight * kl_loss + 
                       self.beta_domain * domain_loss)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_cross_recon_loss += cross_recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_domain_loss += domain_loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Loss: {total_loss/len(dataloader):.4f}, "
                      f"Recon: {total_recon_loss/len(dataloader):.4f}, "
                      f"CrossRecon: {total_cross_recon_loss/len(dataloader):.4f}, "
                      f"KL: {total_kl_loss/len(dataloader):.4f}, "
                      f"DomainLoss: {total_domain_loss/len(dataloader):.4f}")
        
        return self
    
    def transform(self, radiomics_data, pathomics_data, return_embeddings='concatenated',
                  return_dataframe=True, index=None):
        """Transform data to aligned latent representations"""
        
        # Normalize
        radiomics_norm = self.radiomics_scaler.transform(radiomics_data)
        pathomics_norm = self.pathomics_scaler.transform(pathomics_data)
        
        # Convert to tensors
        radiomics_tensor = torch.FloatTensor(radiomics_norm).to(self.device)
        pathomics_tensor = torch.FloatTensor(pathomics_norm).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get latent representations
            radio_mu, radio_logvar = self.model.encode_radiomics(radiomics_tensor)
            patho_mu, patho_logvar = self.model.encode_pathomics(pathomics_tensor)
            
            # Use means as embeddings (deterministic)
            radio_z = radio_mu
            patho_z = patho_mu
            
            if return_embeddings == 'concatenated':
                # Concatenate latent representations
                embeddings = torch.cat([radio_z, patho_z], dim=1)
                
            elif return_embeddings == 'mean':
                # Simple mean of the two latent representations
                embeddings = (radio_z + patho_z) / 2
                
            elif return_embeddings == 'radiomics':
                # Use only radiomics latent space
                embeddings = radio_z
                
            elif return_embeddings == 'pathomics':
                # Use only pathomics latent space
                embeddings = patho_z
                
            elif return_embeddings == 'bayesian':
                # Bayesian fusion using product of experts
                # Convert log variance to precision (inverse variance)
                radio_precision = torch.exp(-radio_logvar)
                patho_precision = torch.exp(-patho_logvar)
                
                # Product of experts: precision = sum of precisions, mean = weighted sum by precision
                fused_precision = radio_precision + patho_precision
                fused_mean = (radio_precision * radio_z + patho_precision * patho_z) / (fused_precision + 1e-8)
                
                embeddings = fused_mean
                
            else:
                raise ValueError(f"Unknown return_embeddings: {return_embeddings}")
            
        embeddings_np = embeddings.cpu().numpy()
        
        # Convert to DataFrame if requested
        if return_dataframe:
            if return_embeddings == 'concatenated':
                columns = [f'latent_radio_{i}' for i in range(self.latent_dim)] + \
                         [f'latent_patho_{i}' for i in range(self.latent_dim)]
            elif return_embeddings in ['mean', 'bayesian']:
                columns = [f'fused_latent_{i}' for i in range(self.latent_dim)]
            elif return_embeddings == 'radiomics':
                columns = [f'radio_latent_{i}' for i in range(self.latent_dim)]
            elif return_embeddings == 'pathomics':
                columns = [f'patho_latent_{i}' for i in range(self.latent_dim)]
            else:
                columns = [f'embedding_{i}' for i in range(embeddings_np.shape[1])]
            
            embeddings_np = pd.DataFrame(embeddings_np, columns=columns)
            if index is not None:
                embeddings_np.index = index
        
        return embeddings_np
    
    def fit_transform(self, radiomics_data, pathomics_data, return_embeddings='concatenated'):
        """Fit and transform in one step"""
        self.fit(radiomics_data, pathomics_data)
        return self.transform(radiomics_data, pathomics_data, return_embeddings)
    
    def save(self, filepath):
        """Save the trained transformer"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'radiomics_scaler': self.radiomics_scaler,
            'pathomics_scaler': self.pathomics_scaler,
            'radiomics_dim': self.radiomics_dim,
            'pathomics_dim': self.pathomics_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim
        }
        torch.save(save_dict, filepath)
    
    def load(self, filepath):
        """Load a trained transformer"""
        save_dict = torch.load(filepath, map_location=self.device)
        self.radiomics_dim = save_dict['radiomics_dim']
        self.pathomics_dim = save_dict['pathomics_dim']
        self.hidden_dim = save_dict['hidden_dim']
        self.latent_dim = save_dict['latent_dim']
        
        self.model = DomainAdaptationVAE(
            self.radiomics_dim, self.pathomics_dim,
            self.hidden_dim, self.latent_dim
        ).to(self.device)
        self.model.load_state_dict(save_dict['model_state_dict'])
        self.radiomics_scaler = save_dict['radiomics_scaler']
        self.pathomics_scaler = save_dict['pathomics_scaler']


class DomainAdaptedSurvivalPipeline:
    """Pipeline that combines domain adaptation with survival prediction"""
    
    def __init__(self, domain_adaptor, survival_model, embedding_type='concatenated'):
        self.domain_adaptor = domain_adaptor
        self.survival_model = survival_model
        self.embedding_type = embedding_type
        self.is_fitted = False
    
    def fit(self, radiomics_train, pathomics_train, y_train):
        """Fit domain adaptation and survival model"""
        
        # Fit domain adaptation and get embeddings
        train_embeddings = self.domain_adaptor.fit_transform(
            radiomics_train, pathomics_train, 
            return_embeddings=self.embedding_type
        )
        
        # Train survival model on embeddings
        self.survival_model.fit(train_embeddings, y_train)
        self.is_fitted = True
        
        return self
    
    def predict(self, radiomics_test, pathomics_test):
        """Predict risk scores"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Transform test data
        test_embeddings = self.domain_adaptor.transform(
            radiomics_test, pathomics_test, 
            return_embeddings=self.embedding_type
        )
        
        # Predict with survival model
        return self.survival_model.predict(test_embeddings)
    
    def predict_proba(self, radiomics_test, pathomics_test):
        """Predict survival probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        test_embeddings = self.domain_adaptor.transform(
            radiomics_test, pathomics_test, 
            return_embeddings=self.embedding_type
        )
        
        if hasattr(self.survival_model, 'predict_proba'):
            return self.survival_model.predict_proba(test_embeddings)
        else:
            raise AttributeError("Survival model does not have predict_proba method")
        
    def predict_with_uncertainty(self, radiomics_test, pathomics_test, n_samples=100):
        """Predict with uncertainty estimation using Monte Carlo sampling"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Normalize
        radiomics_norm = self.domain_adaptor.radiomics_scaler.transform(radiomics_test)
        pathomics_norm = self.domain_adaptor.pathomics_scaler.transform(pathomics_test)
        
        # Convert to tensors
        radiomics_tensor = torch.FloatTensor(radiomics_norm).to(self.domain_adaptor.device)
        pathomics_tensor = torch.FloatTensor(pathomics_norm).to(self.domain_adaptor.device)
        
        self.domain_adaptor.model.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Sample from latent distributions
                radio_mu, radio_logvar = self.domain_adaptor.model.encode_radiomics(radiomics_tensor)
                patho_mu, patho_logvar = self.domain_adaptor.model.encode_pathomics(pathomics_tensor)
                
                # Reparameterize to get samples
                radio_z = self.domain_adaptor.model.reparameterize(radio_mu, radio_logvar)
                patho_z = self.domain_adaptor.model.reparameterize(patho_mu, patho_logvar)
                
                # Fuse based on embedding type
                if self.embedding_type == 'concatenated':
                    embeddings = torch.cat([radio_z, patho_z], dim=1)
                elif self.embedding_type == 'mean':
                    embeddings = (radio_z + patho_z) / 2
                elif self.embedding_type == 'bayesian':
                    radio_precision = torch.exp(-radio_logvar)
                    patho_precision = torch.exp(-patho_logvar)
                    
                    # Product of experts: precision = sum of precisions, mean = weighted sum by precision
                    fused_precision = radio_precision + patho_precision
                    embeddings = (radio_precision * radio_z + patho_precision * patho_z) / (fused_precision + 1e-8)
                else:
                    embeddings = radio_z if self.embedding_type == 'radiomics' else patho_z
                
                # Predict
                pred = self.survival_model.predict(embeddings.cpu().numpy())
                predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    def get_latent_representations(self, radiomics_data, pathomics_data):
        """Get the aligned latent representations without survival prediction"""
        return self.domain_adaptor.transform(
            radiomics_data, pathomics_data, 
            return_embeddings=self.embedding_type
        )


