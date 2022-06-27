import numpy as np

def standardize(X_train):
    cache = {'mean':0, 'std':0}
    cache['mean'], cache['std'] = np.nanmean(X_train, axis= 0), np.nanstd(X_train, axis= 0)
    return (X_train-cache['mean'])/cache['std'], cache

def standardize_test(X_test, cache):
    return (X_test-cache['mean'])/cache['std']

def fft_ifft(f, pad_num, thr= None):
    t = np.arange(0, len(f))
    n = len(f) 
    fhat = np.fft.fft(f, n) # Compute the one-dimensional discrete Fourier Transform.
    psd = fhat * np.conj(fhat) # Power spectrum
    indices = np.ones(psd.shape)
    if thr is not None: 
        indices = psd >= thr
    elif 0 < thr <=1: 
        idx = np.argsort(psd)
        cumsum = np.cumsum(psd[idx])
        cumrat = cumsum/cumsum[-1]
        thr_infered = np.min(psd[idx[cumrat > thr]])
        indices = psd >= thr_infered
    fhat = indices * fhat 
    fhat = np.concatenate((fhat, np.zeros((pad_num, ))))
    ffilt = np.fft.ifft(fhat)
    return ffilt.real

def fourier_imputation(f, mask, window= 100, thr= None):
    """
    Imputation using Fourier transform

    # Parameter
    ___________
    f: an univariate time series data
    mask: an indicater vector (True for not missing data)

    # Return
    ________
    f_ret: a complete univariate time series (= without missing values)
    """
    f_ret = f.copy()
    # imputation using fourier transform...
    i = 0
    while i < len(mask): 
        if mask[i] and i+1 < len(mask) and not mask[i+1]:
            idx_str = i
            pad_num = 1
            i += 2
            while i < len(mask) and not mask[i]:
                i += 1
                pad_num += 1
            # print(f"idx_str: {idx_str}, pad_num: {pad_num}")
            n = idx_str+pad_num+1
            f_in = f_ret[:idx_str+1] if idx_str+1 - window < 0 else f_ret[idx_str+1 - window:idx_str+1]
            f_tmp = fft_ifft(f_in, pad_num= pad_num, thr= thr)
            f_ret[idx_str+1:n] = f_tmp[-1-pad_num:-1] 
            # print(idx_str, pad_num, i)
        else: 
            i += 1
    return f_ret