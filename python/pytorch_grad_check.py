"""
A simple class to check if your manual backwards passes are correct
"""

import torch
from typing import List

class DifferentiableFunction:
    def __init__(self):
        self.context = {}

    def DoForward(self, *inputs) -> List[torch.Tensor]:
        raise NotImplementedError()

    def DoBackward(self, *grad_outputs) -> List[torch.Tensor]:
        raise NotImplementedError()

    def CheckGrad(self, inputs: List[torch.Tensor], rtol=1e-5, atol=1e-8):
        our_forward = self.DoForward(*inputs)
        
        dummy_loss = 0
        diff_forward = []
        for t in our_forward:
            t = t.detach()
            t.requires_grad_(True)
            diff_forward.append(t)
            dummy_loss += t.mean()
        dummy_loss.backward()
        grad_outputs = [t.grad for t in diff_forward]
        our_gradients = self.DoBackward(grad_outputs)


        diff_pt_forward = []
        for t in inputs:
            t = t.detach()
            t.requires_grad_(True)
            diff_pt_forward.append(t)
        
        outputs: List[torch.Tensor] = self.DoForward(*diff_pt_forward)
        pt_dummy_loss = 0
        for t in outputs:
            pt_dummy_loss += t.mean()
        breakpoint()

        pt_dummy_loss.backward()
        pt_gradients = [t.grad for t in diff_pt_forward]

        if len(our_gradients) != len(pt_gradients):
            print("The gradient from PyTorch and the manual backward pass does not have the same number of outputs.")
            return False

        for idx, (our_grad, pt_grad) in enumerate(zip(our_gradients, pt_gradients)):
            if our_grad.shape != pt_grad.shape:
                print(f"The gradient from PyTorch and the manual backward pass do not have the same shape (element {idx}).")
                return False

            if not torch.allclose(our_grad, pt_grad, rtol, atol):
                print(f"The gradients at element {idx} do not match")
                return False
        
        return True  



if __name__ == "__main__":
    class SimpleFunc(DifferentiableFunction):
        
        def DoForward(self, a: torch.Tensor, b: torch.Tensor):
            self.context = {'a': a, 'b': b}
            return [10*a*b + b]

        def DoBackward(self, dL_doutput) -> List[torch.Tensor]:
            a,b = self.context['a'], self.context['b']
            dL_da = dL_doutput[0]*10*b
            dL_db = dL_doutput[0]*10*a + dL_doutput[0]
            return [dL_da, dL_db]

    simple_func = SimpleFunc()
    simple_func.CheckGrad([torch.rand(3), torch.rand(3)])
