from functorch import jacfwd, jacrev

def jacobian(f, z, mode='fwd'):
    if mode == 'fwd':
        J = jacfwd(f, randomness='same')(z).sum(dim=0).transpose(0, 1)
    else:
        J = jacrev(f)(z).sum(dim=0).transpose(0, 1)
    
    return J
    
