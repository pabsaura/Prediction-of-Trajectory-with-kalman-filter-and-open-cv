mport numpy as np
import numpy.linalg as la

def kalman(mu,P,F,Q,B,u,z,H,R):
    # mu, P : estado actual y su incertidumbre.
    # F, Q : sistema dinamico y su ruido.
    # B, u : control model y la entrada.
    # z : observacion.
    # H, R : modelo de observacion y su ruido.
    mup = F @ mu + B @ u; #Estado predicho sin observacion.
    pp = F @ P @ F.T + Q; #Incertidumbre cuando no hay observacion.

    zp = H @ mup #Prediccion respecto al modelo.

    # si no hay observacion solo hacemos prediccion.
    if z is None:
        return mup, pp, zp
    
    epsilon = z - zp #Discrepancia entre la observacion y su prediccion.
    
    k = pp @ H.T @ la.inv(H @ pp @ H.T +R) #Ganancia de Kalman.
    
    new_mu = mup + k @ epsilon; #Nuevo estado actual.
    new_P = (np.eye(len(P))-k @ H) @ pp; #Nueva incertidumbre.
    return new_mu, new_P, zp
