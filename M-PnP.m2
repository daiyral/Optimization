
​-- Restart Macaulay interpreter.
polyRing = QQ[lambda, a, b, c, d];

-- The r vector and its transpose.
r = matrix{{a^2 + b^2 - c^2 - d^2}, {2 * b *c + 2 * a * d}, {2 * b * d - 2 * a * c},
  {2 * b * c - 2 * a * d}, {a^2 + c^2 - b^2 - d^2}, {2 * c * d + 2 * a * b},
  {2 * b * d + 2 * a * c}, {2 * c * d - 2 * a *b}, {a^2 + d^2 - b^2 - c^2}}

rT = transpose r;

-- The optimized function.
f = rT * M * r;

-- The constraints function.
phi = a^2 + b^2 + c^2 + d^2 - 1;

-- The LaGrange function.
L = f + lambda * phi;

-- Take all the partial derivatives of the LaGrange function.
dLda = diff(a, L);
dLdb = diff(b, L);
dLdc = diff(c, L);
dLdd = diff(d, L);
dLdlambda = diff(lambda, L);

-- Create the polynomial system.
pols = {dLda, dLdb, dLdc, dLdd, dLdlambda}

-- Generate an ideal from the polynomials. The ideal will be created for the lex ordered polynomial ring.
polsIdeal = ideal pols;



end

-- ------Numeric version using EigenSolver
restart
M = random(QQ^9, QQ^9);
load "M-PnP.m2"
time J = eliminate(lambda, polsIdeal);
degree (JsmallRing = QQ[drop(gens ring polsIdeal,1)])
needsPackage "EigenSolver";
sols = zeroDimSolve sub(J,JsmallRing)
-- (sortSolutions sols / matrix - sortSolutions solsNoLambda / matrix) / norm -- both methods should give the same result

-- ------Numeric version using Msolve
restart
M = random(QQ^9, QQ^9,Height=>1000000)
load "M-PnP.m2"
time J = eliminate(lambda, polsIdeal);
degree (JsmallRing = QQ[drop(gens ring polsIdeal,1)])
needsPackage "Msolve"
sols = msolveRealSolutions(sub(J,JsmallRing),RRi)
-- (sortSolutions sols / matrix - sortSolutions solsNoLambda / matrix) / norm -- both methods should give the same result
