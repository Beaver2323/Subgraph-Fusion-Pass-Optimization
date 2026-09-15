


def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    l_x_ = L_x_
    l_y_ = L_y_
    stack = torch.stack([l_x_, l_y_], dim = 1);  l_x_ = l_y_ = None
    return (stack,)
    