import sympy as sp
import re

eq_sr = "sin((x + y + 1.652747)*0.34147164)*0.9288274"
expr = sp.sympify(eq_sr)
expr = expr.xreplace({n: round(n, 3) for n in expr.atoms(sp.Number)})
print("PySR:", sp.latex(expr))

eq_sindy = "(x0)' = -13.008 1 +  49.689 x0 + -65.692 x0^2 +  28.937 x0^3"
eq_sindy = eq_sindy.replace("(x0)'", "u'")
eq_sindy = eq_sindy.replace("(x0)''", "u''")
eq_sindy = eq_sindy.replace("x0", "u")
eq_sindy = re.sub(r'(\d+\.\d+)\s+1\b', r'\1', eq_sindy)
if "=" in eq_sindy:
    lhs, rhs = eq_sindy.split("=", 1)
    rhs = rhs.replace("^", "**")
    expr = sp.sympify(rhs)
    expr = expr.xreplace({n: round(n, 3) for n in expr.atoms(sp.Number)})
    print("PySINDy:", f"${lhs.strip()} = {sp.latex(expr)}$")
