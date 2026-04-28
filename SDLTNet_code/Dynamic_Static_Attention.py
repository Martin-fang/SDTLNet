class Dynamic_Static_Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., alpha=0.5, eps=1e-8):
        """
        Args:
            dim: Input feature dimension
            heads: Number of attention heads
            dim_head: Dimension of each head
            dropout: Dropout probability
            alpha: Fusion coefficient (0~1), controls the weight between dots and Pearson correlation
            eps: Numerical stability term
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.alpha = alpha
        # self._alpha = nn.Parameter(torch.zeros(1)) 
        self.eps = eps
    
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def pearson_corr(self, x):
        """
        Compute Pearson correlation coefficient
        Args:
            x: [B, h, N, D]
        Returns:
            corr: [B, h, N, N]
        """
        B, h, N, D = x.shape
        mean = x.mean(dim=-1, keepdim=True)         # [B,h,N,1]
        x_centered = x - mean                       # Mean centering
        numerator = einsum('b h i d, b h j d -> b h i j', x_centered, x_centered)  # Covariance
        std = torch.sqrt((x_centered**2).sum(dim=-1, keepdim=True) + self.eps)     # [B,h,N,1]
        denominator = torch.matmul(std, std.transpose(-2, -1))                     # [B,h,N,N]
        corr = numerator / (denominator + self.eps)
        return corr

    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        Returns:
            out: [B, N, D]
        """
        b, n, d = x.shape
        h = self.heads
        
        # 1. Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # 2. Dot product attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # [B,h,N,N]
        # 3. Pearson correlation coefficient (based on Q, can also use K)
        pearson = self.pearson_corr(q)  # [B,h,N,N]
        sim = self.alpha * dots + (1 - self.alpha) * pearson  # [B,h,N,N]
#       
        # 5. Softmax to get attention distribution
        attn = self.attend(sim)
        # 6. Weighted sum
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)