def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + 2 * pad - kernel_size) // stride + 1
