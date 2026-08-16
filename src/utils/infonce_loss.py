import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for Image-Text Retrieval
    
    InfoNCE (Information Noise Contrastive Estimation) uses cross-entropy 
    on the similarity matrix instead of margin-based triplet loss.
    
    Args:
        temperature: Temperature parameter for similarity scaling
        use_memory_bank: Whether to use memory bank for additional negatives
        memory_size: Size of memory bank if enabled
    """
    
    def __init__(self, temperature=0.07, use_memory_bank=True, memory_size=1024):
        super().__init__()
        self.temperature = temperature
        self.use_memory_bank = use_memory_bank
        self.memory_size = memory_size
        
        # Learnable temperature for better optimization
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/temperature)))
        
        # Memory banks for storing past embeddings
        if use_memory_bank:
            self.register_buffer('img_memory_bank', torch.zeros(memory_size, 512))
            self.register_buffer('txt_memory_bank', torch.zeros(memory_size, 512))
            self.memory_ptr = 0
    
    def update_memory_bank(self, img_emb, txt_emb):
        """Update memory banks with current batch embeddings"""
        if not self.use_memory_bank:
            return img_emb, txt_emb
            
        batch_size = img_emb.size(0)
        
        # Get current memory content
        img_mem = self.img_memory_bank.clone()
        txt_mem = self.txt_memory_bank.clone()
        
        # Update memory with current batch
        end_ptr = (self.memory_ptr + batch_size) % self.memory_size
        
        if end_ptr > self.memory_ptr:
            # No wrap-around
            self.img_memory_bank[self.memory_ptr:end_ptr] = img_emb.detach()
            self.txt_memory_bank[self.memory_ptr:end_ptr] = txt_emb.detach()
        else:
            # Wrap-around case
            self.img_memory_bank[self.memory_ptr:] = img_emb[:self.memory_size - self.memory_ptr].detach()
            self.img_memory_bank[:end_ptr] = img_emb[self.memory_size - self.memory_ptr:].detach()
            self.txt_memory_bank[self.memory_ptr:] = txt_mem[:self.memory_size - self.memory_ptr].detach()
            self.txt_memory_bank[:end_ptr] = txt_mem[self.memory_size - self.memory_ptr:].detach()
        
        self.memory_ptr = end_ptr
        
        # Concatenate current batch with memory bank
        img_expanded = torch.cat([img_emb, img_mem], dim=0)
        txt_expanded = torch.cat([txt_emb, txt_mem], dim=0)
        
        return img_expanded, txt_expanded
    
    def forward(self, img_emb, txt_emb):
        """
        Compute InfoNCE loss
        
        Args:
            img_emb: Image embeddings [batch_size, embedding_dim]
            txt_emb: Text embeddings [batch_size, embedding_dim]
        
        Returns:
            InfoNCE loss value
        """
        batch_size = img_emb.size(0)
        
        # 1. Normalize embeddings for cosine similarity
        img_emb = F.normalize(img_emb, p=2, dim=1)
        txt_emb = F.normalize(txt_emb, p=2, dim=1)
        
        # 2. Update memory banks and get expanded embeddings
        img_expanded, txt_expanded = self.update_memory_bank(img_emb, txt_emb)
        
        # 3. Compute similarity matrix with learnable temperature
        # Shape: [batch_size + memory_size, batch_size + memory_size]
        logits = torch.mm(img_expanded, txt_expanded.t()) * self.logit_scale.exp()
        
        # 4. Split into current batch and memory bank portions
        # Current batch logits: [batch_size, batch_size + memory_size]
        current_logits = logits[:batch_size, :]
        
        # 5. Create labels: positives are at diagonal positions (0, 1, 2, ..., batch_size-1)
        labels = torch.arange(batch_size, device=img_emb.device)
        
        # 6. Compute InfoNCE loss (cross-entropy)
        # Image-to-Text direction
        loss_i2t = F.cross_entropy(current_logits, labels)
        
        # Text-to-Image direction (transpose)
        loss_t2i = F.cross_entropy(current_logits.t(), labels)
        
        # 7. Symmetric loss (average of both directions)
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


# Example usage:
if __name__ == "__main__":
    # Create loss function
    loss_fn = InfoNCELoss(temperature=0.07, use_memory_bank=True, memory_size=1024)
    
    # Example embeddings
    batch_size = 32
    embedding_dim = 512
    img_emb = torch.randn(batch_size, embedding_dim)
    txt_emb = torch.randn(batch_size, embedding_dim)
    
    # Compute loss
    loss = loss_fn(img_emb, txt_emb)
    print(f"InfoNCE Loss: {loss.item():.4f}")
