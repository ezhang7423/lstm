import torch

from torch.autograd import Variable

dtype = torch.FloatTensor


x = Variable(torch.Tensor([1, 2, 3, 10]))
y = Variable(torch.Tensor([-23]))

w = Variable(torch.ones((4)), requires_grad=True)


def calc_loss():
    y_hat = w @ x

    loss = (y - y_hat) ** 2

    loss.backward()
    print('Y hat:', y_hat.data)
    print('Loss:', loss.data)
    
    return w.grad

def learn(iters):
    for _ in range(iters):
        dldw = calc_loss()
        print('grad:', dldw)
        w.data -= 1e-3 * dldw
        print()
        w.grad.data.zero_()

learn(50)
print(w.data)