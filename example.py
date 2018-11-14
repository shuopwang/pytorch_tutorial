import torch
from torch.autograd import Variable

class my_layer(torch.autograd.Function):
    '''
    定制自己的layer
    '''
    @staticmethod
    def forward(ctx, input):
        '''

        :param ctx: 上下文对象，可用于存储反向计算的信息
        :param input:
        可以使用ctx.save_for_backward方法缓存任意对象用于后向传播
        :return:
        '''
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        '''

        :param ctx: 因为我们在forward中已经用save_for_backward存储了对象，所以可以用saved_tensors对象
        :param grad_output:
        :return:
        '''
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

    