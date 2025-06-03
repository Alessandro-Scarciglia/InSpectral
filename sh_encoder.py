# Import modules
import torch
import torch.nn as nn
from profiler import timing_decorator


class SHEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 3,
                 degree: int = 4,
                 out_dim: int = 16,
                 device: str = 'cpu'):
        super(SHEncoder, self).__init__()

        # Attributes
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.degree = degree
        self.out_dim = out_dim

        # Check consistency
        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        # SH Coefficients
        self.C0 = torch.tensor(0.28209479177387814, device=device)

        self.C1 = torch.tensor(0.4886025119029199, device=device)

        self.C2 = torch.tensor([
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ], device=device)

        self.C3 = torch.tensor([
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ], device=device)

        self.C4 = torch.tensor([
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ], device=device)

    @timing_decorator
    def forward(self, input, **kwargs):

        # Bring input to target device
        input = input.to(self.device)

        # Prepare input for encoding
        result = torch.empty((*input.shape[:-1], self.out_dim),
                             dtype=input.dtype,
                             device=self.device)
        x, y, z = input.unbind(-1)

        # First order: 1 element
        result[..., 0] = self.C0

        # Second order: 4 elements
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x

            # Third order: 9 elements
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)

                # Fourth order: 16 elements
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)

                    # Fifth order: 25 elements
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
    
