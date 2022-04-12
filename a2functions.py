import pandas as pd
import numpy as np

def generate_dataset1a(n_samples=100):    
    """
    w is treatment, y is outcome, x is covariate
    """   
    w = np.random.binomial(n=1, p=0.4, size=n_samples)
    x = np.random.normal(2, 1, n_samples)    
    y0 =  2 * x
    y1 =  y0 + 1.5
    y = np.where(w == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "w":w})

def estimate_treatmenteffect(df):
    y0 = df[df.w == 0]
    y1 = df[df.w == 1]
    
    delta = y1.y.mean() - y0.y.mean()
    delta_err = np.sqrt(
        y1.y.var() / y1.shape[0] + 
        y0.y.var() / y0.shape[0])
    bias = delta - 1.5
    rmse = np.sqrt(abs(bias))
    
    return {"estimated_effect": delta, "bias":bias, "rmse":rmse,"standard_error": delta_err}

def generate_dataset1b(n_samples=100):    
    """
    w is treatment, y is outcome, x is covariate
    """   
    w = np.random.binomial(n=1, p=0.4, size=n_samples)
    x = np.full(n_samples, 3, dtype=int)    
    y0 =  2 * x
    y1 =  y0 + 1.5
    y = np.where(w == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "w":w})

def generate_dataset2a(n_samples=100):    
    """
    w is treatment, y is outcome, x is covariate, u is confounder
    """   
    u = np.random.uniform(0.2,0.8)
    w = np.random.binomial(n=1, p=u, size=n_samples)
    x = np.random.normal(2, 1, n_samples)    
    y0 =  2 * x
    y1 =  y0 + 1.5
    y = np.where(w == 0, y0, y1) + 2*u + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "w":w, "u":u})

def generate_dataset2b(n_samples=100):    
    """
    w is treatment, y is outcome, x is covariate, u is confounder
    """   
    u = 0.5
    w = np.random.binomial(n=1, p=u, size=n_samples)
    x = np.random.normal(2, 1, n_samples)    
    y0 =  2 * x
    y1 =  y0 + 1.5
    y = np.where(w == 0, y0, y1) + 2*u + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "w":w, "u":u})

def generate_dataset3a(n_samples=100):    
    """
    w is treatment, y is outcome, x is covariate, s is selection bias
    """   
    s = 0.25
    w = np.random.binomial(n=1, p=0.4, size=n_samples)
    x = np.random.normal(2, 1, n_samples)    
    y0 =  2 * x + s
    y1 =  y0 + 1.5
    y = np.where(w == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "w":w})

def generate_dataset3b(n_samples=100):    
    """
    w is treatment, y is outcome, x is covariate, s is selection bias
    """   
    s = np.random.standard_normal()
    w = np.random.binomial(n=1, p=0.4, size=n_samples)
    x = np.random.normal(2, 1, n_samples)    
    y0 =  2 * x + s
    y1 =  y0 + 1.5
    y = np.where(w == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "w":w})

def generate_dataset4(n_samples=100):    
    """
    w is treatment, y is outcome, x is covariate, 
    since the outcome variable is overrepresented at 0, we set the prob of the treatment as 0.1, and the untreated outcome as 0.
    """   
    w = np.random.binomial(n=1, p=0.1, size=n_samples)
    x = np.random.normal(0, 1, n_samples)    
    y0 =  x
    y1 =  y0 + 1.5
    y = np.where(w == 0, y0, y1) + 0.3 * np.random.normal(size=n_samples)
        
    return pd.DataFrame({"x":x, "y":y, "w":w})

def estimate_treatmenteffect_cop(df):
    df = df[df.y > 0]     ##the Conditional-on-Positives (COP) framework
    y0 = df[df.w == 0]
    y1 = df[df.w == 1]
    
    delta = y1.y.mean() - y0.y.mean()
    delta_err = np.sqrt(
        y1.y.var() / y1.shape[0] + 
        y0.y.var() / y0.shape[0])
    bias = delta - 1.5
    rmse = np.sqrt(abs(bias))
    
    return {"estimated_effect": delta, "bias":bias, "rmse":rmse,"standard_error": delta_err}
