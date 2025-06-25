needsPackage "NumericalLinearAlgebra"
-- Normalize view vectors manually (Macaulay2 doesn't do this automatically)
normalize = v -> v * (1 / norm v)
zeroMatrix := (QQMatrixRing, m, n) -> map(QQMatrixRing^m,QQMatrixRing^n,0)

setupM = (P,V) -> (
    -- Initialize matrices
    QSum := zeroMatrix(QQMatrixRing, 3, 3);
    QCSum := zeroMatrix(QQMatrixRing, 3, 9);
    Qlist := {};
    Clist := {};
    for i from 0 to N-1 do (
	point := P_{i}; -- 3x1 column
	v := V_{i};
	vNorm := normalize v;
	
	Q := id_(QQ^3) - (vNorm * transpose(vNorm));
        Qlist = Qlist | {Q};
	
	-- Build C matrix
	px := point_(0,0);
	py := point_(1,0);
	pz := point_(2,0);

	C := matrix{
	    {transpose point, 0, 0},
	    {0, transpose point, 0},
	    {0, 0, transpose point}
	    };
	Clist = Clist | {C};
	
	QSum = QSum + Q;
	QCSum = QCSum + Q * C;
	);

    -- Solve for T
    T := - QSum^(-1) * QCSum; -- 3x9 matrix

    -- Build M matrix
    M := zeroMatrix(QQMatrixRing, 9, 9);

    for i from 0 to N-1 do (
	D := Clist#i + T;
	M = M + transpose D * Qlist#i * D;
	);

    M
    )
-- Now M and T are ready for symbolic work

-- end
-- restart
load "setupM-PnP.m2"
-- Define the base ring
QQMatrixRing = RR

-- Number of points
N = 4 -- or however many points you want

-- Example data: replace with actual data
P = matrix(QQMatrixRing, {{1,2,3,4}, {2,3,4,5}, {3,4,5,6}}) -- 3xN matrix of 3D points
V = matrix(QQMatrixRing, {{1,1,0,0}, {0,1,1,0}, {0,0,1,1}}) -- 3xN matrix of view vectors

P = random(QQMatrixRing^3,QQMatrixRing^N)
V = random(QQMatrixRing^3,QQMatrixRing^N)

M' = setupM(P,V)
numericalRank M'
M = matrix applyTable(entries M', e->lift(round(6,e),QQ))
load "M-PnP.m2"
needsPackage "Msolve"
sols = msolveRealSolutions(polsIdeal,RRi)
#sols
rur = msolveRUR(polsIdeal);

numgens polsIdeal
polsIdeal / degree
needsPackage "NumericalAlgebraicGeometry"
solsNAG = solveSystem polsIdeal_*;
