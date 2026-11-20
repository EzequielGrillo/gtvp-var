"""Pruebas rapidas del paquete con datos sinteticos."""
import importlib.util
from pathlib import Path

import numpy as np

from gtvpvar import GTVPVAR

RAIZ = Path(__file__).resolve().parents[1]


def _ejemplo():
    spec = importlib.util.spec_from_file_location("datos_sinteticos", RAIZ / "ejemplos" / "datos_sinteticos.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ejemplo_recupera_regimenes():
    B = _ejemplo().main(n_estimators=50)
    assert B.shape == (2, 2, 3)
    a_bajo, a_alto = np.diag(B[0, :, 1:]), np.diag(B[1, :, 1:])
    assert (a_alto - a_bajo > 0.3).all()


def test_misma_semilla_mismo_resultado_y_guardar_cargar(tmp_path):
    X, S, Y = _ejemplo().simular(T=150)
    kw = dict(n_estimators=10, mtry=0.5, block_size=6, trend_push_column=3, random_state=7)
    m1, m2 = GTVPVAR(**kw).fit(X, S, Y), GTVPVAR(n_jobs=2, **kw).fit(X, S, Y)
    assert np.array_equal(m1.betas_arboles(S), m2.betas_arboles(S))
    m1.guardar(tmp_path / "bosque.npz")
    m3 = GTVPVAR.cargar(tmp_path / "bosque.npz")
    assert np.array_equal(m1.betas_arboles(S), m3.betas_arboles(S))


def test_mrf_una_ecuacion():
    X, S, Y = _ejemplo().simular(T=150)
    m = GTVPVAR(n_estimators=5, mtry=0.5, random_state=0).fit(X, S, Y[:, 0])
    assert m.predict_beta(S[:3]).shape == (3, 1, 3)


def test_matrices_var_con_rezagos():
    from gtvpvar import matrices_var
    y = np.arange(20.0).reshape(10, 2)
    X, Y = matrices_var(y, rezagos=3)
    assert X.shape == (7, 1 + 2 * 3) and Y.shape == (7, 2)
    np.testing.assert_array_equal(Y[0], y[3])
    np.testing.assert_array_equal(X[0], np.r_[1.0, y[2], y[1], y[0]])
    X0, _ = matrices_var(y, rezagos=1, constante=False)
    np.testing.assert_array_equal(X0, y[:-1])


def test_pronosticar_var_p():
    from gtvpvar import matrices_var, pronosticar
    X, S, Y = _ejemplo().simular(T=200)
    y = np.vstack([np.zeros((1, 2)), Y])
    Xv, Yv = matrices_var(y, rezagos=2)
    m = GTVPVAR(n_estimators=5, mtry=0.5, random_state=0).fit(Xv, S[1:], Yv)
    f = pronosticar(m, y, S[-1], h=4, rezagos=2)
    assert f.shape == (4, 2)
    B = m.predict_beta(S[-1][None])[0]
    np.testing.assert_allclose(f[0], B @ np.r_[1.0, y[-1], y[-2]])
    np.testing.assert_allclose(f[1], B @ np.r_[1.0, f[0], y[-1]])
